"""Train the stacked DeepONet on snapshot-pair datasets (dataset.py output).

Preprocessing convention (v2 — mirrored by export.py + device kernels):
  inputs : T -> log10 ; rho -> log10 ; X -> Box-Cox(lam) ; F -> log10
           then Z-score every input column
  trunk  : dt -> log10 -> Z-score
  target : --target delta (default, paper-style increments):
             X : BoxCox(X') - BoxCox(X)   (transformed-space increment)
             T : log10(T') - log10(T)
           --target abs (v1):
             X : BoxCox(X') ; T : log10(T')
           then Z-score target columns
  decode (delta): X' = invBoxCox( BoxCox(X) + dX ) ; T' = T * 10^dT
  (Box-Cox: B(x) = (x^lam - 1)/lam, x >= 0; X floored at 1e-30, F at 1e-30)

Loss: weighted L1/Huber on z-scored targets
      + w_sum * || sum_k Xhat_k - 1 ||_1  (de-normalized mass fractions)
      Species weights: 1 where input X > presence-thresh (1e-12),
      w_trace (0.25) below — trace/absent species contribute less.

Schedule (paper-style): phase 1 (batch B, lr, cosine) for epochs//2,
      phase 2 (batch B2, lr/10, cosine) — Adam restarted at the switch.
      Early stopping on val L1 (patience epochs).

Split: by RUN id (10% of runs held out for validation) — never by pair.

Usage:
  python3 train.py --data .../dataset_prod.npz --tag prod \
                   --target delta --epochs 5000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import StackedDeepONet

FLOOR = 1e-30


def boxcox(x, lam):
    x = np.maximum(x, 0.0)
    if abs(lam) < 1e-8:
        return np.log(np.maximum(x, FLOOR))
    return (np.power(np.maximum(x, FLOOR), lam) - 1.0) / lam


def inv_boxcox(y, lam):
    if abs(lam) < 1e-8:
        return np.exp(y)
    return np.power(np.maximum(1.0 + lam * y, 0.0), 1.0 / lam)


def zscore(v):
    return (v - v.mean(0)) / (v.std(0) + 1e-12), v.mean(0), v.std(0) + 1e-12


def t_boxcox(x, lam):
    """torch Box-Cox, same floor semantics as boxcox()."""
    x = x.clamp(min=FLOOR)
    if abs(lam) < 1e-8:
        return torch.log(x)
    return (torch.pow(x, lam) - 1.0) / lam


def t_inv_boxcox(t, lam):
    """torch inverse Box-Cox, arg clamped at 0 ( mass fractions >= 0 )."""
    if abs(lam) < 1e-8:
        return torch.exp(t)
    return torch.pow((1.0 + lam * t).clamp(min=0.0), 1.0 / lam)


def preprocess(X_in, Y_lbl, dt, lam, target="delta"):
    n_spe = Y_lbl.shape[1] - 1
    n_hv = X_in.shape[1] - 2 - n_spe
    bcX = boxcox(X_in[:, 2:2 + n_spe], lam)
    lgT = np.log10(np.maximum(X_in[:, 0], FLOOR))
    tin = np.column_stack([
        lgT,                                                     # T
        np.log10(np.maximum(X_in[:, 1], FLOOR)),                 # rho
        bcX,                                                     # X
        np.log10(np.maximum(X_in[:, 2 + n_spe:], FLOOR)),        # F
    ])
    tdt = np.log10(np.maximum(dt, FLOOR))
    bcY = boxcox(Y_lbl[:, :n_spe], lam)
    lgTY = np.log10(np.maximum(Y_lbl[:, -1], FLOOR))
    if target == "delta":
        tout = np.column_stack([bcY - bcX, lgTY - lgT])
    else:
        tout = np.column_stack([bcY, lgTY])
    zin, mu_in, sd_in = zscore(tin)
    zdt, mu_dt, sd_dt = zscore(tdt[:, None])
    zout, mu_out, sd_out = zscore(tout)
    stats = dict(
        lam=lam, n_spe=n_spe, n_hv=n_hv,
        target=(1 if target == "delta" else 0),
        mu_in=mu_in, sd_in=sd_in, mu_dt=mu_dt[0], sd_dt=sd_dt[0],
        mu_out=mu_out, sd_out=sd_out,
    )
    return zin.astype(np.float32), zdt[:, 0].astype(np.float32), \
        zout.astype(np.float32), stats, tin  # tin = transformed inputs


def decode(zpred, tin, stats):
    """z-scored targets -> physical [X', T'] (both target modes)."""
    n_spe = stats["n_spe"]
    t = zpred[:, :n_spe] * stats["sd_out"][:n_spe] + stats["mu_out"][:n_spe]
    dT = zpred[:, n_spe] * stats["sd_out"][n_spe] + stats["mu_out"][n_spe]
    if stats["target"] == 1:  # delta
        X = inv_boxcox(tin[:, 2:2 + n_spe] + t, stats["lam"])
        T = np.power(10.0, np.maximum(tin[:, 0], FLOOR) + dT)
    else:
        X = inv_boxcox(t, stats["lam"])
        T = np.power(10.0, dT)
    return X, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--target", choices=["delta", "abs"], default="delta")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--batch2", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-sum", type=float, default=0.1)
    ap.add_argument("--loss", choices=["l1", "huber"], default="l1")
    ap.add_argument("--presence-thresh", type=float, default=1e-12)
    ap.add_argument("--w-trace", type=float, default=0.25)
    ap.add_argument("--n-basis", type=int, default=32)
    ap.add_argument("--branch-hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--trunk-hidden", type=int, nargs="+", default=[32])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=300,
                    help="early-stop patience [epochs] on val L1")
    ap.add_argument("--w-early", type=float, default=5.0,
                    help="loss weight multiplier for m <= m-early pairs "
                         "(stiff transient, EMCS-style emphasis)")
    ap.add_argument("--m-early", type=int, default=4)
    ap.add_argument("--w-violent", type=float, default=5.0,
                    help="loss weight multiplier for pairs with large "
                         "label changes (bulk |dX| > 0.05 or |dlgT| > 0.05)")
    ap.add_argument("--w-sep", type=float, default=1.0,
                    help="loss weight multiplier for separatrix-band pairs "
                         "(dataset sep_gain = k_sub(T_dust)*dt with "
                         "|log10(gain)| <= --sep-band); 1.0 = off")
    ap.add_argument("--sep-band", type=float, default=1.0,
                    help="half-width [dex] of the marginal separatrix band "
                         "around k_sub*dt = 1")
    ap.add_argument("--w-eps", type=float, default=1.0,
                    help="loss weight multiplier for the eps output slot "
                         "(drives rollout T accuracy; species slots compete "
                         "with it in the plain L1)")
    ap.add_argument("--w-spx", default=None,
                    help="per-species weight overrides, e.g. "
                         "\"C(d)=8,N(d)=8\" -- output-present pairs "
                         "of the named species get the multiplier")
    ap.add_argument("--w-dump", type=float, default=1.0,
                    help="sample-weight multiplier for DAgger dump "
                         "pairs ( kind 0/1 rows from dnn_dump.bin )")
    ap.add_argument("--dust-sel", default="",
                    help="comma-separated species for the K-step "
                         "dust-dex rejection, e.g. \"C(d),N(d)\"")
    ap.add_argument("--dust-rej", type=float, default=3.0,
                    help="phase-3 checkpoints whose K-step rollout "
                         "dust dex exceeds this ( dex ) are rejected")
    ap.add_argument("--lsum-rej", type=float, default=0.02,
                    help="checkpoint rejection threshold on val_lsum: "
                         "checkpoints with mean |sumX-1| >= this are never "
                         "selected (guard metric, threshold not penalty)")
    ap.add_argument("--rollout", type=int, default=0,
                    help="K-step rollout fine-tune ( phase 3 ): unroll "
                         "windows of K consecutive pairs through the model "
                         "( deployment-faithful decode: X renormalized to "
                         "sum 1, eps = 10^(lg eps + dz) ), loss vs "
                         "reference at every step. 0 = off.")
    ap.add_argument("--init", default=None,
                    help="checkpoint (.pt) to start from ( fine-tune ); "
                         "skips phases 1-2")
    ap.add_argument("--epochs3", type=int, default=300,
                    help="phase-3 ( rollout ) epochs")
    ap.add_argument("--lr3", type=float, default=1e-4,
                    help="phase-3 learning rate")
    ap.add_argument("--batch3", type=int, default=256,
                    help="phase-3 rollout batch ( windows )")
    ap.add_argument("--roll-mix", type=float, default=0.5,
                    help="fraction of rollout batches per phase-3 epoch "
                         "( interleaved with teacher-forced batches to "
                         "avoid one-step forgetting )")
    ap.add_argument("--roll-val", type=int, default=2048,
                    help="number of val windows for the rollout val "
                         "metric")
    ap.add_argument("--roll-tail", action="store_true",
                    help="select phase-3 checkpoints on the p90 of "
                         "the K-step eps error ( the amplifying "
                         "threshold-boundary cells ) instead of "
                         "the median")
    ap.add_argument("--tbptt", type=int, default=0,
                    help="truncated backprop window ( steps ) for "
                         "the rollout loss: unroll K steps for the "
                         "state feedback but detach the graph every "
                         "tbptt steps ( 0 = full BPTT, explodes for "
                         "K > ~16 )")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="grad-norm clip for the phase-3 fine-tune")
    ap.add_argument("--w-stat", type=float, default=0.0,
                    help="weight of the rc-43(a) window-statistics term "
                         "(sorted/quantile L1 match of the terminal-state "
                         "distributions across each rollout batch -- "
                         "well-posed past the amplification horizon where "
                         "the reference trajectory is itself a noise "
                         "draw); 0 = off")
    ap.add_argument("--stat-bins", type=int, default=32,
                    help="number of quantile levels for the window-"
                         "statistics match")
    ap.add_argument("--sel-stat", action="store_true",
                    help="phase-3 checkpoint selection on the val stat "
                         "term ( the rc-43(a) distributional objective ) "
                         "instead of pointwise K-eps; epochs must keep "
                         "eps p90 within --sel-tol of the init baseline")
    ap.add_argument("--sel-tol", type=float, default=0.05,
                    help="allowed absolute eps-p90 increase over the "
                         "init baseline when --sel-stat is on")
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--out", default="~/scratch/donet_tst/fid/models")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    outdir = Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X_in, Y_lbl, dt, run = d["X_in"], d["Y_lbl"], d["dt"], d["run"]
    cell = d["cell"] if "cell" in d.files else np.zeros(len(X_in))
    zin, zdt, zout, stats, tin = preprocess(
        X_in, Y_lbl, dt, args.lam, args.target)
    n_spe = stats["n_spe"]

    # ---- rollout window chains: ( run, cell ) trajectories with
    # ---- consecutive m_pair ( needed by --rollout > 0 )
    m_pair_all = d["m_pair"] if "m_pair" in d.files else np.zeros(len(X_in))
    chains = []          # ( start, length ) into the SORTED index order
    win_runs = []        # run id per chain ( for train/val split )
    if args.rollout > 0:
        srt = np.lexsort((m_pair_all, cell, run))
        r_s, c_s, m_s = run[srt], cell[srt], m_pair_all[srt]
        brk = np.ones(len(srt), bool)
        brk[1:] = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1]) \
            | (m_s[1:] != m_s[:-1] + 1)
        starts = np.flatnonzero(brk)
        lens = np.diff(np.append(starts, len(srt)))
        for s, L, rr in zip(starts, lens, r_s[starts]):
            if L >= args.rollout:
                chains.append((int(s), int(L)))
                win_runs.append(int(rr))
        n_win = sum(L - args.rollout + 1 for _, L in chains)
        print(f"rollout: {len(chains)} chains (>= {args.rollout} steps), "
              f"{n_win} windows of K={args.rollout}")

    runs = np.unique(run)
    rng = np.random.default_rng(0)
    val_runs = set(rng.choice(runs, max(1, int(len(runs) * args.val_frac)),
                              replace=False))
    mval = np.isin(run, list(val_runs))
    mtr = ~mval
    print(f"train {mtr.sum()} / val {mval.sum()} samples "
          f"({len(runs) - len(val_runs)}/{len(val_runs)} runs), "
          f"target={args.target}")

    # per-sample species weights (presence-based); eps slot full weight
    w_spe = np.where(X_in[:, 2:2 + n_spe] > args.presence_thresh,
                     1.0, args.w_trace).astype(np.float32)

    # --w-spx: creation-from-floor species ( e.g. haze dust ) sit below
    # the presence threshold at the INPUT of nearly every pair, so the
    # presence weighting above assigns them w_trace ( 0.25 ) -- and the
    # informative pairs are exactly the ones where they ARE present in
    # the OUTPUT ( creation events ). Override: output-present pairs get
    # the multiplier ( input-present or not ).
    spx_slots = []           # ( name, slot, val-output-present mask )
    if args.w_spx:
        spe_names = [str(s) for s in d["species"]]
        for item in args.w_spx.split(","):
            nm, _, mws = item.partition("=")
            nm, mws = nm.strip(), float(mws)
            j = spe_names.index(nm)
            out_pres = Y_lbl[:, j] > args.presence_thresh
            w_spe[out_pres, j] = np.maximum(w_spe[out_pres, j], 1.0) * mws
            vm = out_pres[mval]        # val rows, order matches vin/vout
            spx_slots.append((nm, j, vm))
            print(f"w-spx: {nm} (slot {j}) x{mws:g}: "
                  f"{int(out_pres.sum())} output-present pairs "
                  f"({int(vm.sum())} val)")

    # per-SAMPLE weights (EMCS-style: emphasize stiff transient + violent
    # changes -- the samples the deployed NN trips sanity guards on)
    m_pair = d["m_pair"] if "m_pair" in d.files else np.zeros(len(X_in))
    kind = d["kind"] if "kind" in d.files else np.full(len(X_in), 2, int)
    early = (m_pair >= 0) & (m_pair <= args.m_early)
    w_samp = np.ones(len(X_in), np.float32)
    if args.w_early != 1.0:
        # dump rows ( m_pair < 0 ) are isolated on-policy pairs, NOT
        # the m=0 initial transient -> no early boost ( --w-dump is
        # their knob )
        w_samp[early] *= args.w_early
    if args.w_dump != 1.0 and bool((kind < 2).any()):
        # DAgger on-policy pairs ( shadow / fallback DI labels ):
        # the states the deployed model actually reaches, which the
        # snapshot distribution under-covers ( threshold band )
        w_samp[kind < 2] *= args.w_dump
        print(f"w-dump: {int((kind < 2).sum())} DAgger pairs "
              f"x{args.w_dump:g}")
    if args.w_violent != 1.0:
        dX = np.abs(Y_lbl[:, :n_spe] - X_in[:, 2:2 + n_spe]).max(axis=1)
        dT = np.abs(np.log10(np.maximum(Y_lbl[:, -1], FLOOR))
                    - np.log10(np.maximum(X_in[:, 0], FLOOR)))
        viol = (dX > 0.05) | (dT > 0.05)
        w_samp[viol] *= args.w_violent
        print(f"sample weights: early(m<={args.m_early}) "
              f"{int(early.sum())} x{args.w_early}, "
              f"violent {int(viol.sum())} x{args.w_violent}")
    # --w-sep (rc-43b): emphasize pairs on the dust-sublimation
    # separatrix -- sep_gain = k_sub(T_dust)*dt from dataset.py, the
    # marginal band |log10(gain)| <= sep-band is where sublimation is
    # marginal on the step timescale and tiny errors flip the branch.
    # Datasets without sep_gain are a no-op (flag prints a note).
    sep_gain = d["sep_gain"] if "sep_gain" in d.files \
        else np.zeros(len(X_in))
    if args.w_sep != 1.0:
        if "sep_gain" in d.files:
            with np.errstate(divide="ignore"):
                sep = (sep_gain > 0) \
                    & (np.abs(np.log10(sep_gain)) <= args.sep_band)
            w_samp[sep] *= args.w_sep
            print(f"w-sep: {int(sep.sum())} separatrix-band pairs "
                  f"(|lg k_sub*dt| <= {args.sep_band:g}) "
                  f"x{args.w_sep:g}")
        else:
            print("w-sep: dataset has no sep_gain array -- no-op "
                  "(rebuild with the current dataset.py to enable)")

    # training weights: eps slot upweighted (--w-eps, drives rollout T);
    # val metric weights: eps slot 1.0 so val_l1 stays comparable across
    # --w-eps settings
    wv = np.concatenate(
        [w_spe, np.ones((len(X_in), 1), np.float32)], axis=1)
    wt = np.concatenate(
        [w_spe, np.full((len(X_in), 1), args.w_eps, np.float32)], axis=1)
    wt = wt * w_samp[:, None]

    dev = args.device if torch.cuda.is_available() or args.device == "cpu" \
        else "cpu"
    tzin = torch.tensor(zin[mtr], device=dev)
    tzdt = torch.tensor(zdt[mtr], device=dev)
    tzout = torch.tensor(zout[mtr], device=dev)
    tw = torch.tensor(wt[mtr], device=dev)
    vin, vdt, vout = zin[mval], zdt[mval], zout[mval]
    vin_tr = tin[mval]  # transformed val inputs (for physical decode)

    net = StackedDeepONet(zin.shape[1], zout.shape[1], args.n_basis,
                           tuple(args.branch_hidden),
                           tuple(args.trunk_hidden)).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"model params: {n_par}  (n_in={zin.shape[1]}, "
          f"n_out={zout.shape[1]}, p={args.n_basis})")

    skip12 = False
    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state"])
        print(f"initialized from {args.init} "
              f"(n_in={ck.get('n_in')}, n_out={ck.get('n_out')})")
        skip12 = True

    sd_out = torch.tensor(stats["sd_out"], device=dev,
                           dtype=torch.float32)
    mu_out = torch.tensor(stats["mu_out"], device=dev,
                          dtype=torch.float32)
    lam = args.lam
    # transformed X per sample (for L_sum decode, train + val)
    tX = torch.tensor(tin[:, 2:2 + n_spe], device=dev)
    trX = tX[torch.tensor(mtr, device=dev)]
    vX = tX[torch.tensor(mval, device=dev)]

    def loss_fn(yp, yt, w):
        if args.loss == "huber":
            e = yp - yt
            ae = e.abs()
            h = torch.where(ae < 1.0, 0.5 * e * e, ae - 0.5)
            return (h * w).sum() / w.sum()
        return ((yp - yt).abs() * w).sum() / w.sum()

    def lsum_fn(yp, xtr):
        if stats["target"] == 1:
            t = yp[:, :n_spe] * sd_out[:n_spe] + mu_out[:n_spe] + xtr
        else:
            t = yp[:, :n_spe] * sd_out[:n_spe] + mu_out[:n_spe]
        # clamp the Box-Cox argument to X in [0, 1] ( mass fractions
        # cannot exceed 1 ): bounds xp and its gradient ( (1+lam*t)^(1/lam
        # - 1) <= 1 ), so a single overshooting species can no longer
        # explode the loss ~1e26 and permanently poison Adam. Same
        # semantics as the deployed commit ( which renormalizes by sumX
        # anyway ).
        xp = torch.clamp(1.0 + lam * t, 0.0, 1.0).pow(1.0 / lam)
        return (xp.sum(1) - 1.0).abs().mean()

    def dust_line(vp):
        """monitor line for --w-spx species: median |dlgX| over val
        pairs where the species is present in the OUTPUT -- for floor
        species these are the creation events, the quantity the
        override is supposed to fix."""
        if not spx_slots:
            return ""
        Xp, _ = decode(vp.cpu().numpy(), vin_tr, stats)
        Xr = Y_lbl[mval]
        parts = []
        for nm, j, vm in spx_slots:
            if not vm.any():
                parts.append(f"{nm} n=0")
                continue
            dx = np.abs(np.log10(np.maximum(Xp[vm, j], FLOOR))
                        - np.log10(np.maximum(Xr[vm, j], FLOOR)))
            parts.append(f"{nm} {np.nanmedian(dx):.2f} dex"
                         f"(n={int(vm.sum())})")
        return "  spx: " + ", ".join(parts)

    def run_phase(n_ep, batch, lr, cosine_ep, best, last_imp):
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cosine_ep)
        N = tzin.shape[0]
        for ep in range(n_ep):
            net.train()
            perm = torch.randperm(N, device=dev)
            tot, nb = 0.0, 0
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                yp = net(tzin[idx], tzdt[idx])
                loss = loss_fn(yp, tzout[idx], tw[idx]) \
                    + args.w_sum * lsum_fn(yp, trX[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
            sched.step()
            if ep % 10 == 9 or ep == n_ep - 1:
                net.eval()
                with torch.no_grad():
                    vp = net(torch.tensor(vin, device=dev),
                             torch.tensor(vdt, device=dev))
                    vl1 = loss_fn(vp, torch.tensor(vout, device=dev),
                                  torch.tensor(wv[mval], device=dev)).item()
                    vls = lsum_fn(vp, vX).item()
                # selection: best val_l1 among sumX-HEALTHY checkpoints
                # ( val_lsum < lsum_rej ). A threshold, not an additive
                # penalty: val_lsum is a guard metric, and an additive
                # w*val_lsum term over-penalizes its small growth vs the
                # val_l1 the rollout actually rides on.
                score = vl1 if vls < args.lsum_rej else np.inf
                log.append(dict(ep=glob_ep[0] + ep, train=tot / nb,
                                val_l1=vl1, val_lsum=vls))
                print(f"ep {glob_ep[0] + ep:5d}  train {tot / nb:.5f}  "
                      f"val_l1 {vl1:.5f}  val_lsum {vls:.4f}"
                      + dust_line(vp)
                      + ("" if np.isfinite(score) else "  [sumX-rej]"))
                if score < best[0] - 1e-5:
                    best = (score, glob_ep[0] + ep,
                            {k: v.clone() for k, v in
                             net.state_dict().items()})
                    last_imp = glob_ep[0] + ep
                elif glob_ep[0] + ep - last_imp > args.patience:
                    print(f"early stop (no val improvement for "
                          f"{args.patience} epochs)")
                    break
        return best

    def run_rollout(best):
        """Phase 3: K-step rollout fine-tune on ( run, cell ) windows.

        Unroll replicates the deployed decode exactly: X' =
        invBoxCox( BoxCox(X) + dX ) renormalized to sum 1; eps' =
        10^(lg eps + dz); rho / flux / dt taken from the reference
        trajectory ( exogenous ). Loss at every step vs the reference
        label, so state-feedback amplification ( rollout runaway ) is
        penalized directly, not just the one-step error. With --w-stat
        > 0 the training loss additionally gets w_stat * stat_term: a
        sorted-quantile ( Wasserstein-1 type ) match of the window-
        TERMINAL state distributions across the batch ( rc-43(a) --
        past the local amplification horizon the reference trajectory
        is itself a noise draw, so the distribution, not the pointwise
        state, is the well-posed target ). Val windows report stat but
        it never enters the selection score."""
        K = args.rollout
        srt = np.lexsort((m_pair_all, cell, run))
        f32 = dict(device=dev, dtype=torch.float32)
        t_tin = torch.tensor(tin[srt], **f32)          # ( n, n_in )
        t_zdt = torch.tensor(zdt[srt], **f32)
        t_out = torch.tensor(zout[srt], **f32)
        t_w = torch.tensor(wt[srt], **f32)
        t_wv = torch.tensor(wv[srt], **f32)
        mu_in_t = torch.tensor(stats["mu_in"], **f32)
        sd_in_t = torch.tensor(stats["sd_in"], **f32)
        S = n_spe

        # P3 dust-dex rejection ( --dust-sel "C(d),N(d)" ): the
        # one-step dust dex is already ~0 ( root-cause 33 ); the
        # deployment failure is K-step compounding through the
        # ignition threshold band, which this measures directly
        # on the val windows ( dust-active subset ).
        spe_names = ([str(s) for s in d["species"]]
                     if "species" in d.files and len(d["species"]) else [])
        dust_j = [spe_names.index(nm.strip())
                  for nm in args.dust_sel.split(",")
                  if nm.strip() and nm.strip() in spe_names] \
            if args.dust_sel else []
        if args.dust_sel:
            print(f"dust-sel: {[(spe_names[j], j) for j in dust_j]} "
                  f"( reject > {args.dust_rej:g} dex, K-step )")

        tr_ch = [(s, L) for (s, L), rr in zip(chains, win_runs)
                 if rr not in val_runs]
        va_ch = [(s, L) for (s, L), rr in zip(chains, win_runs)
                 if rr in val_runs]
        if not va_ch:
            print("rollout: no VAL chains ( val split holds no "
                  "trajectory runs ) -- phase 3 needs a val rollout "
                  "metric; skipped")
            return best
        cs = np.array([c[0] for c in tr_ch])
        cl = np.array([c[1] for c in tr_ch])
        cw = cl - K + 1          # windows per chain
        n_win_tr = int(cw.sum())
        print(f"rollout: {len(tr_ch)} train chains / {n_win_tr} windows; "
              f"{len(va_ch)} val chains")

        rng_w = np.random.default_rng(1)

        def sample_widx(n, chain_set=None):
            if chain_set is not None:
                s_arr = np.array([c[0] for c in chain_set])
                l_arr = np.array([c[1] for c in chain_set])
                w_arr = l_arr - K + 1
                pick = rng_w.choice(len(s_arr), size=n, p=w_arr / w_arr.sum())
                u = (rng_w.random(n) * w_arr[pick]).astype(int)
            else:
                pick = rng_w.choice(len(cs), size=n, p=cw / cw.sum())
                u = (rng_w.random(n) * cw[pick]).astype(int)
            widx = cs[pick] + u if chain_set is None else s_arr[pick] + u
            return torch.tensor(
                widx[:, None] + np.arange(K)[None, :], device=dev)

        # fixed val window set ( reproducible selection metric )
        n_va = min(args.roll_val, max(1, sum(L - K + 1 for _, L in va_ch)))
        vwidx = sample_widx(n_va, va_ch)

        # reference ( teacher ) final X on the val windows: decode the
        # label deltas along the reference trajectory ( needed for the
        # dust dex; no model involved )
        out_va = t_out[vwidx]
        with torch.no_grad():
            Xr_va = t_inv_boxcox(t_tin[vwidx][:, 0, 2:2 + S], lam)
            Xr_va = Xr_va / Xr_va.sum(1, keepdim=True).clamp(min=1e-30)
            for j in range(K):
                dXbc_r = out_va[:, j, :S] * sd_out[:S] + mu_out[:S]
                Xr_va = t_inv_boxcox(t_boxcox(Xr_va, lam) + dXbc_r, lam)
                Xr_va = Xr_va / Xr_va.sum(1, keepdim=True).clamp(min=1e-30)

        def dust_dex_of(Xf):
            """max over --dust-sel species of the median K-step
            |lg X_model - lg X_ref| over dust-active val windows
            ( ref final X > 1e-15 )."""
            if not dust_j:
                return 0.0
            vals = []
            for j in dust_j:
                act = Xr_va[:, j] > 1e-15
                if bool(act.any()):
                    a = torch.clamp(Xf[act, j], min=1e-30)
                    b = torch.clamp(Xr_va[act, j], min=1e-30)
                    vals.append(float(
                        (torch.log10(a / b)).abs().median()))
            return max(vals) if vals else 0.0

        def roll_metrics(dlge):
            """median and p90 of the K-step |deps|/eps error --
            the tail ( p90 ) is what the deployed amplifiers
            ride on; the median hides them."""
            r = np.abs(10.0 ** dlge.detach().cpu().numpy() - 1.0)
            return float(np.median(r)), float(np.percentile(r, 90))

        def unroll(widx, weights):
            B = widx.shape[0]
            tin_w = t_tin[widx]                       # ( B, K, n_in )
            zdt_w = t_zdt[widx]
            out_w = t_out[widx]
            w_w = weights[widx]
            lge = tin_w[:, 0, 0].clone()
            X = t_inv_boxcox(tin_w[:, 0, 2:2 + S], lam)
            X = X / X.sum(1, keepdim=True).clamp(min=1e-30)
            tot = 0.0
            for j in range(K):
                bc = t_boxcox(X, lam)
                tin_j = torch.cat(
                    [lge[:, None], tin_w[:, j, 1:2], bc,
                     tin_w[:, j, 2 + S:]], dim=1)
                z_in = (tin_j - mu_in_t) / sd_in_t
                yp = net(z_in, zdt_w[:, j])
                tot = tot + loss_fn(yp, out_w[:, j], w_w[:, j]) \
                    + args.w_sum * lsum_fn(yp, bc)
                dXbc = yp[:, :S] * sd_out[:S] + mu_out[:S]
                dz = yp[:, S] * sd_out[S] + mu_out[S]
                Xn = t_inv_boxcox(bc + dXbc, lam)
                X = Xn / Xn.sum(1, keepdim=True).clamp(min=1e-30)
                lge = lge + dz
                # TBPTT: the unrolled state keeps feeding forward ( the
                # model still SEES its own K-step drift ), but gradients
                # are cut every tbptt steps: K-step BPTT is a product of
                # K Jacobians through the renormalize/Box-Cox loop and
                # explodes for K > ~16 ( observed: loss 0.067 -> nan in
                # 60 epochs, Adam permanently poisoned ).
                if args.tbptt > 0 and (j + 1) % args.tbptt == 0 \
                        and j < K - 1:
                    X = X.detach()
                    lge = lge.detach()
            # reference final lg eps: start + sum of label increments
            dz_lbl = out_w[:, :, S] * sd_out[S] + mu_out[S]
            dlge = lge - (tin_w[:, 0, 0] + dz_lbl.sum(1))
            # reference ( teacher ) terminal X, decoded along the
            # window from the label deltas -- same recipe as Xr_va,
            # done here so train AND val calls both get it ( rc-43(a)
            # window-statistics term ). Detached: the reference must
            # never carry gradient.
            with torch.no_grad():
                Xr = t_inv_boxcox(tin_w[:, 0, 2:2 + S], lam)
                Xr = Xr / Xr.sum(1, keepdim=True).clamp(min=1e-30)
                for j in range(K):
                    dXbc_r = out_w[:, j, :S] * sd_out[:S] + mu_out[:S]
                    Xr = t_inv_boxcox(t_boxcox(Xr, lam) + dXbc_r, lam)
                    Xr = Xr / Xr.sum(1, keepdim=True).clamp(min=1e-30)
            return tot / K, dlge, X, Xr

        def stat_term(Xm, dlge, Xr):
            """rc-43(a) window-statistics term: L1 match of the
            SORTED terminal-state values ( quantile functions ) across
            the window batch -- a 1-D Wasserstein-1 type distribution
            match. Species ( --dust-sel set, or ALL when unset ) are
            matched in lg10 space ( dex, commensurate across
            species ); eps is matched on the terminal lg-eps error
            dlge directly ( the reference's own error is identically
            zero ). Gradients flow to the model tensors only ( Xr is
            detached; the sort permutation is non-differentiable but
            value-level gradients through the sorted model tensor are
            what we want )."""
            cols = dust_j if dust_j else list(range(S))
            B = Xm.shape[0]
            nb = min(args.stat_bins, B)
            # stat_bins quantile levels of the sorted batch; with
            # B < stat_bins, nb == B = the full sorted vectors
            idx = torch.linspace(0, B - 1, nb,
                                 device=Xm.device).round().long()
            qm = torch.sort(
                torch.log10(Xm[:, cols].clamp(min=1e-30)),
                dim=0).values[idx]
            qr = torch.sort(
                torch.log10(Xr[:, cols].clamp(min=1e-30)),
                dim=0).values[idx]
            qe = torch.sort(dlge, dim=0).values[idx]
            return (qm - qr).abs().mean() + qe.abs().mean()

        # baseline ( pre-fine-tune ) rollout val
        net.eval()
        with torch.no_grad():
            vl, dlge, Xf, Xr_b = unroll(vwidx, t_wv)
            base_roll = vl.item()
            base_med, base_p90 = roll_metrics(dlge)
            base_dust = dust_dex_of(Xf)
            base_stat = stat_term(Xf, dlge, Xr_b).item()
            vp = net(torch.tensor(vin, device=dev),
                     torch.tensor(vdt, device=dev))
            base_vls = lsum_fn(vp, vX).item()
        print(f"rollout val ( pre ): loss {base_roll:.5f}, "
              f"K-step |deps|/eps med {base_med:.4f} "
              f"p90 {base_p90:.4f}, dust {base_dust:.2f} dex, "
              f"stat {base_stat:.4f}")
        # register the init checkpoint as the baseline candidate: the
        # fine-tune may only IMPROVE on it. With a fresh (inf) best the
        # first fine-tune epoch always wins, so a diverging phase 3 can
        # silently return a model WORSE than the one it started from
        # ( observed: init p90 0.206, restored-best p90 0.342 ).
        # If the init itself fails the dust gate it stays at score inf
        # ( weights kept as fallback ) -- any dust-passing epoch then
        # wins, i.e. the fine-tune is ALLOWED to trade eps for dust.
        base_sel = base_p90 if args.roll_tail else base_med
        if args.sel_stat:
            base_sel = base_stat
        if skip12 and best[1] == -1 and best[2] is not None \
                and base_vls < args.lsum_rej \
                and base_dust <= args.dust_rej:
            best = (base_sel, -1, best[2])
            roll_best[0] = base_sel

        opt = torch.optim.Adam(net.parameters(), lr=args.lr3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs3)
        N = tzin.shape[0]
        n_tf = max(1, int((1.0 - args.roll_mix) * N))
        n_rol = max(1, int(args.roll_mix * N / K))
        last_imp = glob_ep[0]
        for ep in range(args.epochs3):
            net.train()
            tot, nb = 0.0, 0
            perm = torch.randperm(N, device=dev)[:n_tf]
            for i in range(0, n_tf, args.batch):
                idx = perm[i:i + args.batch]
                yp = net(tzin[idx], tzdt[idx])
                loss = loss_fn(yp, tzout[idx], tw[idx]) \
                    + args.w_sum * lsum_fn(yp, trX[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               args.grad_clip)
                opt.step()
                tot += loss.item(); nb += 1
            for i in range(0, n_rol, args.batch3):
                widx = sample_widx(min(args.batch3, n_rol - i))
                loss, dlge_t, Xt, Xr_t = unroll(widx, t_w)
                if args.w_stat > 0:
                    # rc-43(a): distribution match on the terminal
                    # states, on TOP of the pointwise per-step loss
                    # ( which handles the short-K fidelity )
                    loss = loss + args.w_stat * stat_term(Xt, dlge_t,
                                                          Xr_t)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               args.grad_clip)
                opt.step()
                tot += loss.item(); nb += 1
            sched.step()
            if ep % 10 == 9 or ep == args.epochs3 - 1:
                net.eval()
                with torch.no_grad():
                    vl, dlge, Xf, Xr_v = unroll(vwidx, t_wv)
                    vp = net(torch.tensor(vin, device=dev),
                             torch.tensor(vdt, device=dev))
                    vls = lsum_fn(vp, vX).item()
                    vstat = stat_term(Xf, dlge, Xr_v).item()
                eps_med, eps_p90 = roll_metrics(dlge)
                ddex = dust_dex_of(Xf)
                eps_sel = eps_p90 if args.roll_tail else eps_med
                if args.sel_stat:
                    # rc-43(a): select on the distributional term,
                    # guarded by the pointwise eps tail staying within
                    # sel-tol of the init baseline -- the pointwise
                    # metric alone rejects exactly the pointwise ->
                    # distributional trade this phase is for ( dagger3
                    # run 1: every epoch rejected, best = init )
                    score = (vstat if vls < args.lsum_rej
                             and ddex <= args.dust_rej
                             and eps_p90 <= base_p90 + args.sel_tol
                             else np.inf)
                else:
                    # NOTE: vstat is REPORTED only -- it never enters the
                    # checkpoint-selection score ( selection semantics
                    # unchanged from pre-rc-43(a) ).
                    score = (eps_sel if vls < args.lsum_rej
                             and ddex <= args.dust_rej
                             else np.inf)
                log.append(dict(ep=glob_ep[0] + ep, roll_train=tot / nb,
                                roll_val=vl.item(), roll_eps=eps_med,
                                roll_eps_p90=eps_p90, val_lsum=vls,
                                dust_dex=ddex, stat=vstat))
                print(f"ep {glob_ep[0] + ep:5d}  roll_train {tot / nb:.5f}"
                      f"  roll_val {vl.item():.5f}  "
                      f"K-eps med {eps_med:.4f} p90 {eps_p90:.4f}"
                      f"  val_lsum {vls:.4f}"
                      f"  dust {ddex:.2f}"
                      f"  stat {vstat:.4f}"
                      + dust_line(vp)
                      + ("" if np.isfinite(score) else "  [rej]"))
                if score < best[0] - 1e-5:
                    best = (score, glob_ep[0] + ep,
                            {k: v.clone() for k, v in
                             net.state_dict().items()})
                    last_imp = glob_ep[0] + ep
                    roll_best[0] = eps_sel
                elif glob_ep[0] + ep - last_imp > args.patience:
                    print(f"early stop ( no rollout-val improvement for "
                          f"{args.patience} epochs )")
                    break
        return best

    log = []
    glob_ep = [0]
    roll_best = [np.nan]   # K-step eps metric of the selected ckpt
    best = (np.inf, -1, None)
    if not skip12:
        n1 = args.epochs // 2
        best = run_phase(n1, args.batch, args.lr, n1, best, -1)
        glob_ep[0] = n1
        if best[2] is not None:
            best = run_phase(args.epochs - n1, args.batch2, args.lr / 10.0,
                             args.epochs - n1, best, glob_ep[0])
    else:
        glob_ep[0] = 0
        best = (np.inf, -1,
                {k: v.clone() for k, v in net.state_dict().items()})

    # ---- phase 3: K-step rollout fine-tune -------------------------
    if args.rollout > 0 and best[2] is not None:
        best = run_rollout(best)
    if best[2] is not None:
        net.load_state_dict(best[2])
        print(f"restored best ckpt (score {best[0]:.5f} @ ep {best[1]})")
        if args.rollout > 0 and np.isfinite(roll_best[0]):
            print(f"selected-ckpt "
                  + (f"stat term {roll_best[0]:.4f}" if args.sel_stat
                     else f"K-step |deps|/eps "
                          f"({'p90' if args.roll_tail else 'median'}) "
                          f"{roll_best[0]:.4f}"))

    # ---- physical validation metrics (decoded, interpretable) ----
    net.eval()
    with torch.no_grad():
        vp = net(torch.tensor(vin, device=dev),
                 torch.tensor(vdt, device=dev)).cpu().numpy()
    Xp, Tp = decode(vp, vin_tr, stats)
    Xr = Y_lbl[mval][:, :n_spe]
    Tr = Y_lbl[mval][:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        dex = np.abs(np.log10(np.maximum(Xp, FLOOR))
                     - np.log10(np.maximum(Xr, FLOOR)))
        dex[Xr <= 1e-10] = np.nan
    relT = np.abs(Tp - Tr) / np.maximum(Tr, 1.0)
    print(f"\n1-step val: median|dlgX| {np.nanmedian(dex):.3f} dex, "
          f"max|dlgX| {np.nanmax(dex):.2f} dex, "
          f"median|dT|/T {np.median(relT):.4f}, "
          f"max|dT|/T {relT.max():.4f}")
    # transient subset (m <= m-early): the stiff states the deployment
    # guards care about
    vmp = m_pair[mval] if "m_pair" in d.files else np.zeros(relT.shape[0])
    m_ear = vmp <= args.m_early
    if m_ear.any():
        Xp_e = np.power(np.maximum(Xp, FLOOR), 1.0)  # keep decode-space
        print(f"transient val (m<={args.m_early}, {int(m_ear.sum())} smp): "
              f"median|dlgX| {np.nanmedian(dex[m_ear]):.3f} dex, "
              f"median|dT|/T {np.median(relT[m_ear]):.4f}, "
              f"frac|sumX-1|>2% "
              f"{np.mean(np.abs(Xp[m_ear].sum(1) - 1) > 0.02):.3f}")
    print("worst species (median dex):")
    for i in np.argsort(-np.nanmedian(dex, 0))[:10]:
        nm = (d["species"][i] if "species" in d else f"sp{i}")
        print(f"  {str(nm):>8s}  {np.nanmedian(dex[:, i]):.3f}")
    if spx_slots:
        Xr_v = Y_lbl[mval][:, :n_spe]
        print("w-spx species (output-present val pairs):")
        for nm, j, vm in spx_slots:
            if not vm.any():
                print(f"  {nm:>8s}  ( no output-present val pairs )")
                continue
            dxs = np.abs(np.log10(np.maximum(Xp[vm, j], FLOOR))
                         - np.log10(np.maximum(Xr_v[vm, j], FLOOR)))
            print(f"  {nm:>8s}  median|dlgX| {np.nanmedian(dxs):.3f} dex "
                  f"(n={int(vm.sum())})")

    torch.save(dict(state=net.state_dict(), args=vars(args),
                    n_in=zin.shape[1], n_out=zout.shape[1]),
               outdir / f"ckpt_{args.tag}.pt")
    np.savez(outdir / f"stats_{args.tag}.npz", **stats,
             presence_thresh=args.presence_thresh,
             w_trace=args.w_trace)
    (outdir / f"trainlog_{args.tag}.json").write_text(json.dumps(log, indent=1))
    print("saved", outdir / f"ckpt_{args.tag}.pt")


if __name__ == "__main__":
    main()
