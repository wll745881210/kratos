#!/usr/bin/env python3
"""Dust / tracer-species accuracy metrics for the DeepONet surrogate.

Compares a hybrid NN run against a DI reference run ( same pars, same
snapshot times ) for species that are CREATED from a near-zero floor
( e.g. C(d), N(d) haze dust ), where per-cell mass fractions are less
meaningful than the integrated inventory.

Per snapshot, per requested species:
  - presence fraction ( X_ref > --pres-thresh ),
  - median / p90 |dlgX| in dex over present cells,
  - mass-weighted inventory  inv = sum( rho * X ) / sum( rho )
    ( both runs ) and its relative error,
  - inventory ratio ( nn / ref ).

Final snapshot additionally dumps a spatial profile ( X along the
monotonic-density ordering, both runs ) for figure use.

Outputs:
  <out>.npz     arrays + species names ( one flat vector per metric,
                ordered [ snap, species ] )
  console table
"""

import argparse, os, re, sys
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


def load(path):
    f = binary_io(str(path))
    f.open()
    n_int = int(np.frombuffer(f[PRE + 'hydro_cons_n_int'], '<i4')[0])
    dt = '<f8' if f.hmap[PRE + 'hydro_cons_field'][1] == 8 else '<f4'
    u = np.frombuffer(f[PRE + 'hydro_cons_field'], dt)\
        .reshape(-1, n_int).copy()
    T = np.frombuffer(f[PRE + 'hydro_T_field'], dt).copy()
    return u, T


def snaps_of(run):
    ss = []
    for f in os.listdir(run):
        m = re.match(r'pl_(\d{5})\.bin$', f)
        if m:
            ss.append(int(m.group(1)))
    return sorted(ss)


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
    ap.add_argument('--spe', nargs='+', default=['C(d)', 'N(d)'])
    ap.add_argument('--pres-thresh', type=float, default=1e-12)
    ap.add_argument('--out', default=None,
                    help='default <analysis dir>/acc_dust_<nn-run>.npz')
    args = ap.parse_args()

    par = os.path.join(args.ref_run, 'planet.par')
    spe = active_species(par)
    idx = []
    for s in args.spe:
        if s not in spe:
            raise RuntimeError(f'species {s} not in {par} species list')
        idx.append(spe.index(s))
    idx = np.array(idx)

    def parse_A(name):
        # atomic mass in amu from the species name ( elements
        # e H He C N O Fe ; charges / (d) markers stripped )
        EL = dict(e=5.49e-4, H=1.00794, He=4.0026, C=12.0107,
                  N=14.0067, O=15.9994, Fe=55.845)
        TOK = re.compile(r'(Fe|He|H|C|N|O|e)(\d*)')
        s = re.sub(r'\(.*?\)', '', name).replace('*', '')
        s = s.replace('+', '').replace('-', '')
        return sum((int(n) if n else 1) * EL[el]
                   for el, n in TOK.findall(s))

    A = np.array([parse_A(s) for s in spe])
    Ai = A[idx]

    ss = [s for s in snaps_of(args.ref_run)
          if os.path.exists(os.path.join(args.nn_run, f'pl_{s:05d}.bin'))]
    if not ss:
        raise RuntimeError('no common snapshots between runs')

    # Kratos cons layout: u_spe = n_i * m_a / rho_0 ( code
    # units ) -- NOT a fraction. Physical mass fraction
    # X_i = u_i * A_i / rho ; the raw field and the mass
    # fraction are both stored. ( number fraction u_i/sum(u)
    # was used here before -- wrong units, root-cause 34. )
    def massfrac(u):
        rho = np.maximum(u[:, [0]], 1e-30)
        return u[:, 5:] * A[None, :] / rho

    res = dict(snaps=np.array(ss), spe=list(args.spe),
               A_amu=Ai,
               pres=np.full((len(ss), len(idx)), np.nan),
               dex_med=np.full((len(ss), len(idx)), np.nan),
               dex_p90=np.full((len(ss), len(idx)), np.nan),
               inv_ref=np.full((len(ss), len(idx)), np.nan),
               inv_nn=np.full((len(ss), len(idx)), np.nan),
               inv_rel=np.full((len(ss), len(idx)), np.nan))
    prof = {}
    for a, s in enumerate(ss):
        uR, TR = load(os.path.join(args.ref_run, f'pl_{s:05d}.bin'))
        uN, TN = load(os.path.join(args.nn_run, f'pl_{s:05d}.bin'))
        XR, XN = massfrac(uR), massfrac(uN)
        for b, i in enumerate(idx):
            xr, xn = XR[:, i], XN[:, i]
            m = xr > args.pres_thresh
            res['pres'][a, b] = m.mean()
            if m.any():
                dex = np.abs(np.log10(np.maximum(xn, FLOOR))
                             - np.log10(np.maximum(xr, FLOOR)))
                res['dex_med'][a, b] = np.median(dex[m])
                res['dex_p90'][a, b] = np.percentile(dex[m], 90)
            # physical inventory: sum( u_i A_i ) / sum( rho )
            iR = float((uR[:, 5 + i] * Ai[b]).sum()
                       / uR[:, 0].sum())
            iN = float((uN[:, 5 + i] * Ai[b]).sum()
                       / uN[:, 0].sum())
            res['inv_ref'][a, b] = iR
            res['inv_nn'][a, b] = iN
            res['inv_rel'][a, b] = (iN - iR) / iR if iR > 0 else np.nan
        if s == ss[-1]:
            n = uR.shape[0]
            nr = n // 2
            # mesh order: x = c % nr ( x fastest ).  The two
            # y-rows are INDEPENDENT radial columns ( per-column
            # radiation rays => row-wise MC draws; guide rc-39 ):
            # keep them separate, ( 2, nr ) profiles, never
            # averaged.
            take = lambda v: v.reshape(2, nr)
            prof = dict(x=np.arange(nr), T_ref=take(TR),
                        T_nn=take(TN))
            for name, i in zip(args.spe, idx):
                prof[f'u_{name}_ref'] = take(uR[:, 5 + i])
                prof[f'u_{name}_nn'] = take(uN[:, 5 + i])
                prof[f'X_{name}_ref'] = take(XR[:, i])
                prof[f'X_{name}_nn'] = take(XN[:, i])

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.ref_run.rstrip('/'))),
        '..', 'analysis',
        f'acc_dust_{os.path.basename(args.nn_run.rstrip("/"))}.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res, **prof)
    print('wrote', out)

    hdr = (f'{"t":>4s} {"species":>6s} {"pres":>5s} {"dex med":>8s} '
           f'{"dex p90":>8s} {"inv ref":>10s} {"inv nn":>10s} '
           f'{"inv rel":>9s}')
    print(hdr)
    print('-' * len(hdr))
    for a, s in enumerate(ss):
        for b, name in enumerate(args.spe):
            print(f'{s:4d} {name:>6s} '
                  f'{res["pres"][a, b]:5.2f} '
                  f'{res["dex_med"][a, b]:8.3f} '
                  f'{res["dex_p90"][a, b]:8.3f} '
                  f'{res["inv_ref"][a, b]:10.3e} '
                  f'{res["inv_nn"][a, b]:10.3e} '
                  f'{res["inv_rel"][a, b]:+9.2e}')


if __name__ == '__main__':
    main()
