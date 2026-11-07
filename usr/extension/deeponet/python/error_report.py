#!/usr/bin/env python3
"""Per-species, per-region error report for the DeepONet surrogate.

Compares an all-NN surrogate run against a DI reference run ( same
pars, same snapshot times ), per species, binned over regions:
  - radial quartiles ( from the monotonic rho profile ),
  - reference-T quartiles.
Metrics: |dlgX| in dex ( floor 1e-30, matching training ), |dX| in
mass fraction, and relative T error. Absent species ( X_ref <= 1e-10
in the region ) are excluded from that region's stats and rendered
gray in the heatmap.

Outputs ( <out>/ ):
  err_species_region.md    human tables
  err_species_region.csv   long-format stats
  err_species_region.png   heatmaps + error boxplots
  slot_map.json            species-slot mapping derived from snap 0
"""

import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.environ.get(
    "KRATOS_VISUAL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "visual")))
from binary_io import binary_io

FLOOR = 1e-30
PRE = 'block_0|'


def active_species(par_path):
    spe = None
    for line in open(par_path).read().splitlines():
        s = line.strip()
        if s.startswith('species') and '=' in s and not s.startswith('#'):
            spe = s.split('=', 1)[1].split()
    if spe is None:
        raise RuntimeError(f'no active "species =" line in {par_path}')
    return spe


def species_init(par_paths):
    vals, sec = {}, None
    for p in par_paths:
        for line in open(p).read().splitlines():
            s = line.strip()
            if s.startswith('[') and s.endswith(']'):
                sec = s[1:-1].lower()
                continue
            if sec == 'species_init' and '=' in s and not s.startswith('#'):
                k, v = s.split('=', 1)
                try:
                    vals[k.strip()] = float(v)
                except ValueError:
                    pass
    return vals


def load(path):
    f = binary_io(str(path))
    f.open()
    n_int = int(np.frombuffer(f[PRE + 'hydro_cons_n_int'], '<i4')[0])
    dt = '<f8' if f.hmap[PRE + 'hydro_cons_field'][1] == 8 else '<f4'
    u = np.frombuffer(f[PRE + 'hydro_cons_field'], dt)\
        .reshape(-1, n_int).copy()
    T = np.frombuffer(f[PRE + 'hydro_T_field'], dt).copy()
    return u, T


def renorm(u):
    X = u[:, 5:] / u[:, [0]]
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def sign_flips(profile):
    d = np.diff(profile)
    d = d[d != 0]
    if d.size < 2:
        return 0
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nn-run', required=True)
    ap.add_argument('--ref-run', required=True)
    ap.add_argument('--ovr', default=None,
                    help='override par with species_init ( default '
                         '<ref-run>/ovr.par )')
    ap.add_argument('--snaps', type=int, nargs='+', default=[1, 11])
    ap.add_argument('--out', required=True)
    ap.add_argument('--top', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    par = os.path.join(args.ref_run, 'planet.par')
    ovr = args.ovr or os.path.join(args.ref_run, 'ovr.par')
    spe = active_species(par)
    init = species_init([par, ovr] if os.path.exists(ovr) else [par])

    # ---- species-slot mapping: par order ( verified: x_slot/x_spe is
    # constant = 1/mu_ma at t=0; absent species default to 1e-20 )
    u0, _ = load(os.path.join(args.ref_run, 'pl_00000.bin'))
    n_spe = u0.shape[1] - 5
    if n_spe != len(spe):
        raise RuntimeError(
            f'par has {len(spe)} species, snapshot has {n_spe}')
    xraw = u0[0, 5:] / u0[0, 0]
    xsn = np.array([init.get(n, 0.0) for n in spe])
    r = xraw / np.maximum(xsn, 1e-300)
    ok = xsn > 0
    ambig = (not ok.any()) or (np.max(r[ok]) / np.min(r[ok]) > 1.01)
    if ambig:
        raise RuntimeError(
            'slot order does not match par species order '
            f'( ratio spread {np.max(r[ok]) / np.min(r[ok]):.3g} )')
    names = list(spe)

    # ---- radial axis: pick the ordering along which rho is monotonic
    rho0 = u0[:, 0]
    n = rho0.size
    a = rho0.reshape(2, n // 2).mean(0)      # x fastest
    b = rho0.reshape(n // 2, 2).mean(1)      # y fastest
    x_fast = sign_flips(a) <= sign_flips(b)
    rad = np.arange(n) % (n // 2) if x_fast else np.arange(n) // 2
    outward = rho0[rad == rad.max()].mean() < rho0[rad == 0].mean()
    n_r = 4
    r_reg = rad * n_r // (n // 2)
    r_lab = ['r0 (inner)', 'r1', 'r2', 'r3 (outer)']
    if not outward:
        r_lab = ['r0 (outer)', 'r1', 'r2', 'r3 (inner)']

    rows = []          # long-format CSV rows
    md = []            # markdown
    md.append('# Per-species / per-region surrogate error report\n')
    md.append(f'- NN run: `{args.nn_run}`  ')
    md.append(f'- DI reference: `{args.ref_run}`  ')
    md.append(f'- slot mapping: par order, verified by constant '
              f'x_slot/x_spe = {np.min(r[ok]):.6g} ( 1/mu_ma ) at t=0\n')
    fig_data = {}

    for s in args.snaps:
        uN, TN = load(os.path.join(args.nn_run, f'pl_{s:05d}.bin'))
        uR, TR = load(os.path.join(args.ref_run, f'pl_{s:05d}.bin'))
        XN, XR = renorm(uN), renorm(uR)
        dex = np.abs(np.log10(np.maximum(XN, FLOOR))
                   - np.log10(np.maximum(XR, FLOOR)))
        dxa = np.abs(XN - XR)
        pres = XR > 1e-10

        lgT = np.log10(np.maximum(TR, 1.0))
        qs = np.quantile(lgT, [0.25, 0.5, 0.75])
        t_reg = np.digitize(lgT, qs)
        t_lab = ['T0 (coolest)', 'T1', 'T2', 'T3 (hottest)']

        md.append(f'\n## Snapshot t = {s}\n')
        md.append('| region | cells | mean T [K] | mean rho | '
                  'med |dT|/T | max |dT|/T |')
        md.append('|---|---|---|---|---|---|')
        for kind, regs, labs in (('radial', r_reg, r_lab),
                                 ('T-quartile', t_reg, t_lab)):
            for g in range(4):
                m = regs == g
                if not m.any():
                    continue
                relT = np.abs(TN[m] - TR[m]) / np.maximum(TR[m], 1.0)
                md.append(f'| {labs[g]} ({kind}) | {int(m.sum())} | '
                          f'{TR[m].mean():.4g} | {uR[m, 0].mean():.4g} | '
                          f'{np.median(relT):.3g} | {relT.max():.3g} |')
                rows.append(dict(snap=s, kind=kind, region=labs[g],
                                 species='(T)', med_dex=np.nan,
                                 max_dex=np.nan,
                                 med_dT_rel=float(np.median(relT)),
                                 max_dT_rel=float(relT.max()),
                                 present_frac=np.nan))
                for i in range(n_spe):
                    sel = pres[m, i]
                    if sel.any():
                        med = float(np.median(dex[m, i][sel]))
                        mx = float(dex[m, i][sel].max())
                        frac = float(sel.mean())
                    else:
                        med = mx = np.nan
                        frac = 0.0
                    rows.append(dict(
                        snap=s, kind=kind, region=labs[g],
                        species=names[i], med_dex=med, max_dex=mx,
                        med_dT_rel=np.nan, max_dT_rel=np.nan,
                        present_frac=frac))

        # species table ( radial regions )
        md.append('\n### Species, radial regions '
                  '( median |dlgX| [dex], gray = absent )\n')
        md.append('| species | all | ' + ' | '.join(r_lab) +
                  ' | max dX |')
        md.append('|---|---|---|---|---|---|---|')
        med_all = []
        for i in range(n_spe):
            sel = pres[:, i]
            med_all.append((float(np.median(dex[:, i][sel]))
                            if sel.any() else np.nan, i))
        for _, i in sorted(med_all, key=lambda t: -(t[0] if
                                                    t[0] == t[0] else -1)):
            cells = [f'{np.median(dex[r_reg == g, i][pres[r_reg == g, i]]):.2f}'
                     if pres[r_reg == g, i].any() else ' - '
                     for g in range(4)]
            sel = pres[:, i]
            mall = (f'{np.median(dex[:, i][sel]):.2f}' if sel.any()
                    else ' - ')
            md.append(f'| {names[i]} | {mall} | ' + ' | '.join(cells) +
                      f' | {dxa[:, i].max():.2e} |')
        fig_data[s] = dict(dex=dex, pres=pres, r_reg=r_reg, r_lab=r_lab,
                           t_reg=t_reg, t_lab=t_lab, names=names,
                           TR=TR, TN=TN)

    with open(os.path.join(args.out, 'err_species_region.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    import csv
    with open(os.path.join(args.out, 'err_species_region.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, 'slot_map.json'), 'w') as f:
        json.dump(dict(names=names, ambiguous=bool(ambig)), f, indent=1)

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib missing; tables only')
        return

    s_last = args.snaps[-1]
    d = fig_data[s_last]
    fig, axes = plt.subplots(1, 3, figsize=(13, 10),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.7]))

    def heat(ax, regs, labs, title):
        M = np.full((n_spe, 4), np.nan)
        for i in range(n_spe):
            for g in range(4):
                m = regs == g
                sel = d['pres'][m, i]
                if sel.any():
                    M[i, g] = np.median(d['dex'][m, i][sel])
        im = ax.imshow(M, aspect='auto', cmap='viridis',
                       vmin=0, vmax=np.nanmax(M))
        ax.set_xticks(range(4), labs, rotation=30, ha='right')
        ax.set_yticks(range(n_spe), d['names'], fontsize=4)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label='median |dlgX| [dex]', shrink=0.4)

    heat(axes[0], d['r_reg'], d['r_lab'],
         f'all-NN ( smoke model ) vs DI, t={s_last}: radial regions')
    heat(axes[1], d['t_reg'], d['t_lab'],
         f'reference-T quartile regions')
    m = d['pres'] & (d['dex'] < 1e9)
    data = [d['dex'][d['r_reg'] == g][d['pres'][d['r_reg'] == g]].ravel()
            for g in range(4)]
    bp = axes[2].boxplot(data, tick_labels=d['r_lab'], showfliers=False)
    axes[2].set_yscale('log')
    axes[2].set_ylabel('|dlgX| [dex]')
    axes[2].set_title('error distribution per region\n'
                      '(present species only)', fontsize=9)
    axes[2].tick_params(axis='x', rotation=30)
    for t in axes[2].get_xticklabels():
        t.set_ha('right')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'err_species_region.png'), dpi=160)
    print('wrote', args.out)

    # console summary
    print(f'\n=== t={s_last} top species by median dex '
          f'( overall, radial regions ) ===')
    med_all = [(np.nanmedian(np.where(d['pres'][:, i],
                                      d['dex'][:, i], np.nan)), i)
               for i in range(n_spe)]
    for v, i in sorted(med_all, key=lambda t: -(t[0] if t[0] == t[0]
                                                else -1))[:args.top]:
        cells = []
        for g in range(4):
            mm = d['r_reg'] == g
            sel = d['pres'][mm, i]
            cells.append(f'{np.median(d["dex"][mm, i][sel]):.2f}'
                         if sel.any() else '-')
        print(f'{d["names"][i]:>8s}  all={v:.2f}  r=[{" ".join(cells)}]')
    print('slot mapping ambiguous:', ambig)


if __name__ == '__main__':
    main()
