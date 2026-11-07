#!/usr/bin/env python3
"""Standard surrogate-model error metrics.

Two families, as customary in the operator-learning / DeepONet
literature (and the arXiv:2504.14180 paper):

  1. One-step (teacher-forced) errors, held-out runs:
       rel-L2 per sample  ||upred - utrue||_2 / ||utrue||_2
       rel-Linf           max_i |upred_i - utrue_i| / ||utrue||_2
     reported for the species vector X and for T separately
     (T as scalar relative error), mean/median/p90 over samples.

  2. Rollout (autoregressive deployment) errors vs a reference run:
       same rel-L2 per cell, per snapshot, plus mean |dT|/T.

Usage:
  one-step : python3 metrics.py --tag prod4 --data ../samples/dataset_prod_v3.npz
  rollout  : python3 metrics.py --tag prod4 --rollout runs/hyb_prod4_g1 \
                --ref runs/di_ref_gpu1 --snaps 1 5 11 51 101
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import StackedDeepONet  # noqa: E402

FLOOR = 1e-30


def boxcox(x, lam):
    if abs(lam) < 1e-8:
        return np.log(np.maximum(x, FLOOR))
    return (np.power(np.maximum(x, FLOOR), lam) - 1.0) / lam


def inv_boxcox(y, lam):
    if abs(lam) < 1e-8:
        return np.exp(y)
    return np.power(np.maximum(1.0 + lam * y, 0.0), 1.0 / lam)


def load_net(ckpt_path, stats_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = StackedDeepONet(ck["n_in"], ck["n_out"],
                          ck["args"].get("n_basis", 32))
    net.load_state_dict(ck["state"])
    net.eval()
    st = dict(np.load(stats_path, allow_pickle=True))
    return net, st


def eval_onestep(net, st, X_in, Y_lbl, dt, batch=8192):
    n_spe = int(st["n_spe"])
    lam = float(st["lam"])
    mu_in, sd_in = st["mu_in"], st["sd_in"]
    mu_out, sd_out = st["mu_out"], st["sd_out"]
    mu_dt, sd_dt = float(st["mu_dt"]), float(st["sd_dt"])

    n = X_in.shape[0]
    Xp = np.empty((n, n_spe))
    Tp = np.empty(n)
    with torch.no_grad():
        for a in range(0, n, batch):
            b = min(a + batch, n)
            tin = np.empty((b - a, X_in.shape[1]))
            tin[:, 0] = np.log10(np.maximum(X_in[a:b, 0], FLOOR))
            tin[:, 1] = np.log10(np.maximum(X_in[a:b, 1], FLOOR))
            tin[:, 2:2 + n_spe] = boxcox(X_in[a:b, 2:2 + n_spe], lam)
            tin[:, 2 + n_spe:] = np.log10(
                np.maximum(X_in[a:b, 2 + n_spe:], FLOOR))
            zin = (tin - mu_in) / sd_in
            zdt = (np.log10(np.maximum(dt[a:b], FLOOR)) - mu_dt) / sd_dt
            z = net(torch.tensor(zin, dtype=torch.float32),
                    torch.tensor(zdt, dtype=torch.float32)).numpy()
            t = z * sd_out + mu_out
            if int(st["target"]) == 1:  # delta
                Xp[a:b] = inv_boxcox(tin[:, 2:2 + n_spe] + t[:, :n_spe], lam)
                Tp[a:b] = np.power(10.0, np.maximum(tin[:, 0], FLOOR)
                                   + t[:, -1])
            else:
                Xp[a:b] = inv_boxcox(t[:, :n_spe], lam)
                Tp[a:b] = np.power(10.0, np.maximum(t[:, -1], FLOOR))

    Xt = np.maximum(Y_lbl[:, :n_spe], 0.0)
    Tt = Y_lbl[:, -1]
    dn = np.linalg.norm(Xt, axis=1)
    dX = np.linalg.norm(Xp - Xt, axis=1)
    rel2 = dX / np.maximum(dn, 1e-30)
    relinf = np.max(np.abs(Xp - Xt), axis=1) / np.maximum(dn, 1e-30)
    relT = np.abs(Tp - Tt) / np.maximum(Tt, FLOOR)
    return rel2, relinf, relT, Xp, Tp


def q(v):
    return (float(np.mean(v)), float(np.median(v)),
            float(np.percentile(v, 90)), float(np.max(v)))


def species_table(net, st, X_in, Y_lbl, dt, names, out_md, roll=None,
                  snaps=(11, 101)):
    """Per-species one-step (teacher-forced, VAL) + rollout errors.

    One-step: |dlgX| dex + relative error per species over held-out
    samples, split by input presence (X_in > 1e-12). Rollout: |dlgX|
    dex per species vs reference run at the requested snapshots.
    Writes a full markdown table + prints the worst rows.
    """
    n_spe = int(st["n_spe"])
    r2, ri, rT, Xp, Tp = eval_onestep(net, st, X_in, Y_lbl, dt)
    Xt = np.maximum(Y_lbl[:, :n_spe], 0.0)
    lgP = np.log10(np.maximum(Xp, 1e-30))
    lgT = np.log10(np.maximum(Xt, 1e-30))
    dex = np.abs(lgP - lgT)
    rel = np.abs(Xp - Xt) / np.maximum(Xt, 1e-30)
    present = X_in[:, 2:2 + n_spe] > 1e-12

    rows = []
    for i, nm in enumerate(names[:n_spe]):
        m = present[:, i]
        v = dex[:, i]
        pv, tv = (v[m], v[~m]) if m.any() and (~m).any() else (v, v)
        rows.append({
            "name": nm, "meanX": float(np.mean(Xt[:, i])),
            "pres": float(m.mean()),
            "med": float(np.median(v)) if v.size else np.nan,
            "p90": float(np.percentile(v, 90)) if v.size else np.nan,
            "max": float(np.max(v)) if v.size else np.nan,
            "med_rel": float(np.median(rel[:, i]))
            if rel.size else np.nan,
            "med_pr": float(np.median(pv)) if pv.size else np.nan,
            "med_ab": float(np.median(tv)) if tv.size else np.nan,
        })

    roll_rows = {}
    if roll:
        import sys as _s
        from pathlib import Path as _P
        _s.path.insert(0, str(_P(__file__).resolve().parents[4]
                               / "visual"))
        from binary_io import binary_io  # noqa: E402

        def load(rd, m):
            f = binary_io(str(_P(rd) / f"pl_{m:05d}.bin"))
            f.open()
            cons = np.frombuffer(f['block_0|hydro_cons_field'], '<f8')
            ni = cons.size // 128
            return cons.reshape(128, ni)

        roll = (roll[0], roll[1])
        for m in snaps:
            cn, cd = load(roll[0], m), load(roll[1], m)
            ns = cn.shape[1] - 5
            Xn = cn[:, 5:] / cn[:, [0]]
            Xd = cd[:, 5:] / cd[:, [0]]
            Xn /= Xn.sum(1, keepdims=True)
            Xd /= Xd.sum(1, keepdims=True)
            dd = np.abs(np.log10(np.maximum(Xn, 1e-30))
                        - np.log10(np.maximum(Xd, 1e-30)))
            for i in range(min(ns, n_spe)):
                roll_rows.setdefault(i, {})[m] = float(
                    np.median(dd[:, i]))

    lines = ["# Species-by-species surrogate errors",
             "",
             f"One-step: held-out samples, n = {len(X_in)}; "
             "`pres` = fraction with X_in > 1e-12.",
             "med/p90/max = |dlgX| [dex]; med_pr = median dex "
             "when present; med_ab = when absent.",
             ""]
    hdr = ("| species | mean X | pres | med dex | p90 dex | "
           "max dex | med rel | med_pr | med_ab |")
    lines.append(hdr)
    lines.append("|" + "---|" * 9)
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["med"])
    for i in order:
        r = rows[i]
        rolltxt = ""
        if i in roll_rows:
            rolltxt = " | ".join(
                f"{roll_rows[i][m]:.2f}" for m in snaps)
        lines.append(
            f"| {r['name']} | {r['meanX']:.2e} | {r['pres']:.2f} | "
            f"{r['med']:.3f} | {r['p90']:.2f} | {r['max']:.1f} | "
            f"{r['med_rel']:.1e} | {r['med_pr']:.3f} | "
            f"{r['med_ab']:.3f} |" + (f" *{rolltxt}*" if rolltxt else ""))
    if roll_rows:
        lines.insert(2, f"Rollout: median |dlgX| dex vs {roll[1]} at "
                        f"t = {list(snaps)} (italic columns).")
    Path(out_md).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_md} ({len(rows)} species)")
    print(f"{'species':>10} {'meanX':>9} {'pres':>5} {'med dex':>8} "
          f"{'p90':>6} {'max':>6} {'med_pr':>7} {'med_ab':>7}")
    for i in order[:18]:
        r = rows[i]
        print(f"{r['name']:>10} {r['meanX']:9.2e} {r['pres']:5.2f} "
              f"{r['med']:8.3f} {r['p90']:6.2f} {r['max']:6.1f} "
              f"{r['med_pr']:7.3f} {r['med_ab']:7.3f}")
    bulk = [r for r in rows if r["meanX"] > 1e-4]
    trac = [r for r in rows if r["meanX"] <= 1e-4]
    for nm, grp in (("bulk (X>1e-4)", bulk), ("trace (X<=1e-4)", trac)):
        if grp:
            md = np.median([r["med"] for r in grp])
            print(f"{nm}: n={len(grp)}  median-of-medians {md:.3f} dex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data")
    ap.add_argument("--rollout")
    ap.add_argument("--ref")
    ap.add_argument("--snaps", type=int, nargs="+",
                    default=[1, 5, 11, 51, 101])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--species", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    models = Path(__file__).resolve().parent.parent.parent.parent \
        / "scratch/donet_tst/fid/models"
    models = Path.home() / "scratch/donet_tst/fid/models"
    ckpt = models / f"ckpt_{a.tag}.pt"
    stats = models / f"stats_{a.tag}.npz"
    net, st = load_net(ckpt, stats)

    if a.data:
        d = dict(np.load(a.data, allow_pickle=True))
        X_in, Y_lbl = d["X_in"], d["Y_lbl"]
        dt, run = d["dt"], d["run"]
        runs = np.unique(run)
        rng = np.random.default_rng(0)
        val = set(rng.choice(runs, max(1, int(len(runs) * a.val_frac)),
                             replace=False))
        mval = np.isin(run, list(val))
        for name, m in (("ALL runs", np.ones(len(run), bool)),
                        ("VAL runs (held out)", mval)):
            r2, ri, rT, _, _ = eval_onestep(net, st, X_in[m], Y_lbl[m],
                                            dt[m])
            m2, md2, p2, x2 = q(r2)
            mi, mdi, pi, xi = q(ri)
            mT, mdT, pT, xT = q(rT)
            print(f"\n== one-step, {name} (n={m.sum()}) ==")
            print(f"  X rel-L2 : mean {m2:.2e}  med {md2:.2e}  "
                  f"p90 {p2:.2e}  max {x2:.2e}")
            print(f"  X rel-Linf: mean {mi:.2e}  med {mdi:.2e}  "
                  f"p90 {pi:.2e}  max {xi:.2e}")
            print(f"  T rel    : mean {mT:.2e}  med {mdT:.2e}  "
                  f"p90 {pT:.2e}  max {xT:.2e}")

        if a.species:
            names = [str(s) for s in d["species"]]
            out = a.out or str(models / f"species_err_{a.tag}.md")
            roll = (a.rollout, a.ref) if (a.rollout and a.ref) else None
            species_table(net, st, X_in[mval], Y_lbl[mval], dt[mval],
                          names, out, roll=roll,
                          snaps=tuple(s for s in a.snaps if s > 1))

    if a.rollout and a.ref:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                               / "visual"))
        from binary_io import binary_io  # noqa: E402

        def load(rd, m):
            f = binary_io(str(Path(rd) / f"pl_{m:05d}.bin"))
            f.open()
            T = np.frombuffer(f['block_0|hydro_T_field'], '<f8')[:128]
            cons = np.frombuffer(f['block_0|hydro_cons_field'], '<f8')
            n_int = cons.size // 128
            cons = cons.reshape(128, n_int)
            return T, cons

        print(f"\n== rollout: {a.rollout} vs {a.ref} ==")
        for m in a.snaps:
            Tn, cn = load(a.rollout, m)
            Td, cd = load(a.ref, m)
            n_spe = cn.shape[1] - 5
            Xn = cn[:, 5:] / cn[:, [0]]
            Xd = cd[:, 5:] / cd[:, [0]]
            Xn /= Xn.sum(1, keepdims=True)
            Xd /= Xd.sum(1, keepdims=True)
            dX = np.linalg.norm(Xn - Xd, axis=1)
            dn = np.linalg.norm(Xd, axis=1)
            r2 = dX / np.maximum(dn, 1e-30)
            rT = np.abs(Tn - Td) / np.maximum(Td, FLOOR)
            m2, md2, p2, x2 = q(r2)
            mT, mdT, pT, xT = q(rT)
            print(f"  t={m:>3}: X rel-L2 med {md2:.2e} p90 {p2:.2e} "
                  f"max {x2:.1e} | T rel med {mdT:.2e} p90 {pT:.2e} "
                  f"max {xT:.2e}")


if __name__ == "__main__":
    main()
