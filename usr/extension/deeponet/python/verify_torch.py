"""Offline verification: replicate the CUDA kernel-B forward pass
(torch + numpy) and compare against the solver's printed outputs.

Replicates, in order (see solver.h nn_bat):
  1. cons/flx read from two snapshots (ghost-row fix for rad_flx);
  2. eps = e_g_cgs/rho_cgs with the eos floor repairs;
  3. z-scored inputs [lg eps, lg rho, Box-Cox X_i, lg flx];
  4. StackedDeepONet forward, decode (delta targets);
  5. nn_eos_T (basic_cv path; cv_file absent for the fid network).

Usage:
  python3 verify_torch.py --tag prod_full --run nn_vfull_hyb \
      [--cells 0 16 32 48 64] [--models DIR] [--runs DIR]

Then compare the printed That/sumX values with the '[dnn] nn cN:'
lines of <run>/run.log (set dnn_print = 1 and dnn_force = 1,
dnn_no_fallback = 1 for a forced all-NN run). Agreement to ~5-6
significant digits confirms the native kernels match PyTorch.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/lilew/apps/kratos_donet/visual")
from binary_io import binary_io  # noqa: E402
from model import StackedDeepONet  # noqa: E402

# species table ( chemistry.cpp parse + periodic.cpp masses )
EL = {"e": 5.49e-4, "H": 1.00794, "He": 4.0026, "C": 12.0107,
      "N": 14.0067, "O": 15.9994, "Fe": 55.845}
MA, ME, KB = 1.66054e-24, 9.10938e-28, 1.38065e-16
GAMMA, TMIN, TMAX = 1.66667, 2.7, 1e7


def parse_sp(name):
    s = name.replace("*", "")
    chg = s.count("+") - s.count("-")
    s = s.replace("+", "").replace("-", "")
    s = re.sub(r"\([^)]*\)", "", s)
    if s == "e":
        return ME, 0
    elem = {}
    while s:
        m = re.search(r"[A-Z][a-z]*", s)
        if not m:
            break
        e = m.group(0)
        s = s[m.end():]
        m2 = re.match(r"\d+", s)
        n = int(m2.group(0)) if m2 else 1
        if m2:
            s = s[m2.end():]
        elem[e] = elem.get(e, 0) + n
    mass = -chg * ME + sum(n * MA * EL[e] for e, n in elem.items())
    n_atom = sum(elem.values())
    return mass, n_atom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="prod_full")
    ap.add_argument("--run", default="nn_vfull_hyb")
    ap.add_argument("--models", default="~/scratch/donet_tst/fid/models")
    ap.add_argument("--runs", default="~/scratch/donet_tst/fid/runs")
    ap.add_argument("--cells", type=int, nargs="+",
                    default=[0, 16, 32, 48, 64])
    args = ap.parse_args()

    M = Path(args.models).expanduser()
    R = Path(args.runs).expanduser() / args.run

    st = dict(np.load(f"{M}/stats_{args.tag}.npz", allow_pickle=True))
    ck = torch.load(f"{M}/ckpt_{args.tag}.pt", map_location="cpu",
                    weights_only=False)
    net = StackedDeepONet(ck["n_in"], ck["n_out"],
                          ck["args"].get("n_basis", 32))
    net.load_state_dict(ck["state"])
    net.eval()

    # species from the run's planet.par (first species = line)
    par = (R.parent.parent / "planet.par")
    if not par.exists():
        par = R / "planet.par"
    spe = None
    for ln in par.read_text().splitlines():
        s = ln.split("#")[0].strip()
        if s.startswith("species =") and spe is None:
            spe = s.split("=", 1)[1].split()
    n_spe = len(spe)
    mcgs = np.array([parse_sp(s)[0] for s in spe])
    gm1i = np.array([0.5 * (3 if na <= 1 else
                            (6 if na >= 3 else 5))
                     for _, na in (parse_sp(s) for s in spe)])

    f0, f1 = binary_io(str(R / "pl_00000.bin")), \
        binary_io(str(R / "pl_00001.bin"))
    f0.open()
    f1.open()
    n_int = int(np.frombuffer(
        f0["block_0|hydro_cons_n_int"], "<i4")[0])
    cons = np.frombuffer(
        f0["block_0|hydro_cons_field"], "<f8").reshape(-1, n_int)
    n_hv = int(np.frombuffer(
        f1["block_0|rad_flx_n_int"], "<i4")[0])
    flx = np.frombuffer(
        f1["block_0|rad_flx_field"], "<f8").reshape(-1, n_hv)
    # ghost-aware rows: rad_flx is ghost-padded ( x fastest )
    fnc = np.frombuffer(f1["block_0|rad_flx_n_cell"], "<i4")
    fng = np.frombuffer(f1["block_0|rad_flx_n_gh"], "<i4")
    cnc = np.frombuffer(f0["block_0|hydro_cons_n_cell"], "<i4")
    ncell = cons.shape[0]
    rows = np.empty(ncell, dtype=np.int64)
    r = 0
    for z in range(cnc[2]):
        for y in range(cnc[1]):
            for x in range(cnc[0]):
                rows[r] = (x + fng[0]) + fnc[0] * (
                    (y + fng[1]) + fnc[1] * (z + fng[2]))
                r += 1
    flx = flx[rows]
    RHO0, ENE0 = 1.67262e-14, \
        1.67262e-14 * (6.37814e8 / 3.6e3) ** 2

    # kernel-B input build ( e_g_cgs + floors )
    rho = np.maximum(cons[:, 0], 1e-9)
    p2 = (cons[:, 2:5] ** 2).sum(1)
    e = np.maximum(cons[:, 1] - p2 / (2 * rho), 1e-9 * rho)
    eps = e * ENE0 / (rho * RHO0)
    X = cons[:, 5:] / cons[:, 5:].sum(1, keepdims=True)

    lam = float(st["lam"])
    mu_in, sd_in = st["mu_in"], st["sd_in"]
    zin = np.zeros((ncell, 2 + n_spe + n_hv))
    zin[:, 0] = (np.log10(np.maximum(eps, 1e-30)) - mu_in[0]) / sd_in[0]
    zin[:, 1] = (np.log10(np.maximum(rho, 1e-30)) - mu_in[1]) / sd_in[1]
    for i in range(n_spe):
        v = np.maximum(X[:, i], 1e-30)
        bc = (v ** lam - 1.0) / lam
        zin[:, 2 + i] = (bc - mu_in[2 + i]) / sd_in[2 + i]
    for k in range(n_hv):
        j = 2 + n_spe + k
        zin[:, j] = (np.log10(np.maximum(flx[:, k], 1e-30))
                     - mu_in[j]) / sd_in[j]
    zdt = np.zeros(ncell)  # dt = 1 s grid -> zdt = 0

    with torch.no_grad():
        out = net(torch.tensor(zin, dtype=torch.float32),
                  torch.tensor(zdt, dtype=torch.float32)).numpy()
    mu_out, sd_out = st["mu_out"], st["sd_out"]
    t = out[:, :n_spe] * sd_out[:n_spe] + mu_out[:n_spe]
    dz_e = out[:, n_spe] * sd_out[n_spe] + mu_out[n_spe]
    bc_in = (np.maximum(X, 1e-30) ** lam - 1.0) / lam
    Xh = np.maximum(1 + lam * (bc_in + t), 0) ** (1.0 / lam)
    eps_h = 10.0 ** (np.log10(np.maximum(eps, 1e-30)) + dz_e)

    # nn_eos_T replication ( basic_cv path; no cv_file for fid ).
    # y = number fractions ( n_i / sum(n) ), matching dat.h's
    # y = u_spe / rho ( both proportional to n_i ). NOT Xh/mcgs
    # -- that is a harmonic mean, wrong when electrons abound.
    y = Xh
    ys = np.maximum(y, 0)
    mu = (ys * mcgs[None, :]).sum(1) / (ys.sum(1) + 1e-32)
    ee = eps_h * mu / KB
    cv = (ys * gm1i[None, :]).sum(1) / (ys.sum(1) + 1e-32)
    T = np.clip(ee / cv, TMIN, TMAX)

    cells = args.cells
    print("torch That:", [f"c{c}={T[c]:.2f}" for c in cells])
    print("torch sumX:", [f"c{c}={Xh[c].sum():.4f}" for c in cells])
    print("( compare with '[dnn] nn cN:' lines in",
          str(R / "run.log"), ")")


if __name__ == "__main__":
    main()
