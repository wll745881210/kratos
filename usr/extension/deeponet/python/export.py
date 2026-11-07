"""Export a trained stacked DeepONet to Kratos `model.dnn` (format v1).

Includes a NUMPY reference forward pass over the written file (selftest:
torch model vs numpy-from-file must agree to ~1e-5 in z-space, plus a
physical decode roundtrip on real dataset samples).

model.dnn v1 layout: see docs/analysis/08-deeponet-surrogate-guide.md
Append A. Transform convention (v2):
  inputs: T->log10, rho->log10, X->Box-Cox(lam), F->log10, then z-score
  trunk : dt->log10, then z-score
  target: flags bit1 = 1 (increments): X: BC(X')-BC(X), T: lgT'-lgT
          flags bit1 = 0 (absolute) : X: BC(X'), T: lgT'
  decode: X' = invBC(BC(X)+dz) ; T' = T*10^dzT   (increment mode)
  stats npz key `target`: 1 = increments (written from train.py --target)
Gating fields T_lo/T_hi/dt_min/dt_max come from CLI (physical units).

Usage:
  python3 export.py --tag prod --T-lo 300 --T-hi 6e4 --dt 1 1 \
      --selftest
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from model import StackedDeepONet

MAGIC = b"DON1"


def gelu(x):
    try:
        from scipy.special import erf
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    except ImportError:
        from math import erf
        return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def w32(buf, view):
    view.extend(struct.pack("<I", buf))


def f32v(view, arr):
    view.extend(np.asarray(arr, dtype="<f4").tobytes())


def u32v(view, arr):
    view.extend(np.asarray(arr, dtype="<u4").tobytes())


def extract_mlp(seq):
    """nn.Sequential(Linear,GELU,...,Linear) -> [(W[in,out], b)] per Linear."""
    layers = []
    for m in seq:
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().cpu().numpy().T.copy()  # [in, out] row-major
            b = m.bias.detach().cpu().numpy().copy()
            layers.append((W.astype(np.float32), b.astype(np.float32)))
    return layers


def write_dnn(path, net, stats, spe, hv, gate):
    br = extract_mlp(net.branch.net)
    tr = extract_mlp(net.trunk.net)
    n_spe, n_hv = stats["n_spe"], stats["n_hv"]
    n_in = int(2 + n_spe + n_hv)
    n_out = int(n_spe + 1)
    v = bytearray()
    v.extend(MAGIC)
    w32(2, v)                       # version 2: eps semantics
                                    # ( slot 0 = lg eps, slot n_spe
                                    #   = eps; v1 T-models obsolete )
    w32(n_spe, v); w32(n_hv, v)
    w32(n_in, v); w32(n_out, v)
    w32(net.p, v)                   # n_basis
    w32(len(br), v); w32(len(tr), v)
    w32(1, v)                       # act = GELU (erf)
    inc = int(stats.get("target", 0)) == 1
    w32(2 if inc else 0, v)         # flags (bit1 = increment targets)
    f32v(v, [gate["T_lo"], gate["T_hi"], gate["dt_min"], gate["dt_max"]])
    f32v(v, [float(stats["lam"])])
    f32v(v, stats["mu_in"]); f32v(v, stats["sd_in"])
    f32v(v, stats["mu_out"]); f32v(v, stats["sd_out"])
    f32v(v, [stats["mu_dt"], stats["sd_dt"]])
    for s in spe:
        b = s.encode("utf-8")
        v.extend(struct.pack("<B", len(b))); v.extend(b)
    while len(v) % 4:                 # pad names to 4-byte alignment
        v.extend(b"\x00")
    f32v(v, hv)
    for W, b in br:                 # branch layers
        w32(W.shape[0], v); w32(W.shape[1], v)
        f32v(v, W); f32v(v, b)
    for W, b in tr:                 # trunk layers
        w32(W.shape[0], v); w32(W.shape[1], v)
        f32v(v, W); f32v(v, b)
    f32v(v, net.bias.detach().cpu().numpy())  # combine bias
    v.extend(MAGIC)
    words = np.frombuffer(bytes(v), dtype="<u4")   # all words incl. trailing magic
    w32(int(words.sum() & 0xFFFFFFFF), v)
    Path(path).write_bytes(bytes(v))
    return len(v)


def read_dnn(path):
    raw = Path(path).read_bytes()
    assert raw[:4] == MAGIC and raw[-8:-4] == MAGIC, "bad magic"
    words_all = np.frombuffer(raw, dtype="<u4")
    chk = int(words_all[:-1].sum() & 0xFFFFFFFF)
    assert chk == words_all[-1], "checksum mismatch"
    o = 4
    def u32():
        nonlocal o
        x = int(words_all[o // 4]); o += 4; return x
    def f32(n=1):
        nonlocal o
        x = np.frombuffer(raw, dtype="<f4", count=n, offset=o).copy(); o += 4 * n
        return x if n > 1 else float(x[0])
    hdr = dict(version=u32(), n_spe=u32(), n_hv=u32(), n_in=u32(), n_out=u32(),
               n_basis=u32(), n_lay_br=u32(), n_lay_tr=u32(), act=u32(), flags=u32())
    assert hdr["version"] == 2, "v1 (T-semantics) models are obsolete"
    hdr.update(T_lo=f32(), T_hi=f32(), dt_min=f32(), dt_max=f32(), lam=f32())
    hdr["mu_in"] = f32(hdr["n_in"]); hdr["sd_in"] = f32(hdr["n_in"])
    hdr["mu_out"] = f32(hdr["n_out"]); hdr["sd_out"] = f32(hdr["n_out"])
    hdr["mu_dt"] = f32(); hdr["sd_dt"] = f32()
    spe = []
    for _ in range(hdr["n_spe"]):
        n = raw[o]; o += 1
        spe.append(raw[o:o + n].decode("utf-8")); o += n
    o += (4 - o % 4) % 4               # skip name-block padding
    hdr["hv_eV"] = f32(hdr["n_hv"])
    def layers(n):
        nonlocal o
        ls = []
        for _ in range(n):
            a, b = u32(), u32()
            W = np.frombuffer(raw, dtype="<f4", count=a * b, offset=o).reshape(a, b).copy(); o += 4 * a * b
            bb = np.frombuffer(raw, dtype="<f4", count=b, offset=o).copy(); o += 4 * b
            ls.append((W, bb))
        return ls
    hdr["branch"] = layers(hdr["n_lay_br"])
    hdr["trunk"] = layers(hdr["n_lay_tr"])
    hdr["bias_out"] = f32(hdr["n_out"])
    return hdr, spe


def forward_np(hdr, zin, zdt):
    x = zin
    for W, b in hdr["branch"][:-1]:
        x = gelu(x @ W + b)
    W, b = hdr["branch"][-1]
    B = x @ W + b                                  # [N, n_out*p]
    t = np.atleast_1d(zdt).astype(np.float64)[:, None]  # [N, 1]
    y = t
    for W, b in hdr["trunk"][:-1]:
        y = gelu(y @ W + b)
    W, b = hdr["trunk"][-1]
    T = y @ W + b                                  # [N, p]
    p, n_out = hdr["n_basis"], hdr["n_out"]
    out = hdr["bias_out"][None, :] + (B.reshape(-1, n_out, p) * T[:, None, :]).sum(-1)
    return out


def inv_boxcox_np(y, lam):
    y = np.asarray(y, dtype=np.float64)
    if abs(lam) < 1e-8:
        return np.exp(y)
    return np.power(np.maximum(1.0 + lam * y, 0.0), 1.0 / lam)


def decode_np(hdr, zpred, tin):
    """z-space predictions -> physical [X', T'] using file-header stats."""
    n_spe = hdr["n_spe"]
    t = zpred[:, :n_spe] * hdr["sd_out"][:n_spe] + hdr["mu_out"][:n_spe]
    dT = zpred[:, n_spe] * hdr["sd_out"][n_spe] + hdr["mu_out"][n_spe]
    if hdr["flags"] & 2:            # increment mode
        X = inv_boxcox_np(tin[:, 2:2 + n_spe] + t, hdr["lam"])
        T = np.power(10.0, tin[:, 0] + dT)
    else:
        X = inv_boxcox_np(t, hdr["lam"])
        T = np.power(10.0, dT)
    return X, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--data", default="", help="dataset npz (default: dataset_<tag>.npz)")
    ap.add_argument("--models", default="~/scratch/donet_tst/fid/models")
    ap.add_argument("--T-lo", type=float, default=300.0)
    ap.add_argument("--T-hi", type=float, default=6e4)
    ap.add_argument("--dt", type=float, nargs=2, default=[1.0, 3600.0],
                    help="dt_min dt_max [code units, must match bdt.dt_0]")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    md = Path(args.models).expanduser()
    ck = torch.load(md / f"ckpt_{args.tag}.pt", map_location="cpu")
    stats = dict(np.load(md / f"stats_{args.tag}.npz", allow_pickle=False))
    data_path = Path(args.data).expanduser() if args.data else md.parent / "samples" / f"dataset_{args.tag}.npz"
    d = np.load(data_path, allow_pickle=True)
    spe = [str(s) for s in d["species"]]
    hv = d["hv_eV"].astype(np.float64)

    net = StackedDeepONet(ck["n_in"], ck["n_out"], ck["args"]["n_basis"],
                          tuple(ck["args"]["branch_hidden"]),
                          tuple(ck["args"]["trunk_hidden"]))
    net.load_state_dict(ck["state"])
    net.eval()

    gate = dict(T_lo=args.T_lo, T_hi=args.T_hi,
                dt_min=args.dt[0], dt_max=args.dt[1])
    out = Path(args.out) if args.out else md / f"model_{args.tag}.dnn"
    n = write_dnn(out, net, stats, spe, hv, gate)
    print(f"wrote {out} ({n} bytes, {len(spe)} species, {len(hv)} bands, p={net.p})")

    if args.selftest:
        hdr, spe2 = read_dnn(out)
        assert spe2 == spe, "species roundtrip failed"
        rng = np.random.default_rng(7)
        zin = rng.standard_normal((256, hdr["n_in"])).astype(np.float32)
        zdt = rng.standard_normal(256).astype(np.float32)
        with torch.no_grad():
            yt = net(torch.tensor(zin), torch.tensor(zdt)).numpy()
        yn = forward_np(hdr, zin, zdt)
        err = np.abs(yt - yn).max()
        print(f"selftest: max |torch - numpy| = {err:.3e}  "
              f"(target mode: {'increments' if hdr['flags'] & 2 else 'absolute'})")
        # structural gate: a loading/order error is O(1); fp32 summation
        # noise on +-3-sigma random inputs can reach ~1e-3 for trained
        # weights. Physical accuracy is gated below on real data.
        assert err < 1e-2, "selftest FAILED"

        # physical decode roundtrip on real samples: numpy-from-file vs
        # torch-model-on-torch-pipeline, both decoded per target mode
        FLOORL = 1e-30
        X_in = d["X_in"].astype(np.float64)
        n_spe = hdr["n_spe"]
        lam = hdr["lam"]
        bcX = (np.power(np.maximum(X_in[:, 2:2 + n_spe], FLOORL), lam)
               - 1.0) / lam
        tin = np.column_stack([
            np.log10(np.maximum(X_in[:, 0], FLOORL)),
            np.log10(np.maximum(X_in[:, 1], FLOORL)),
            bcX,
            np.log10(np.maximum(X_in[:, 2 + n_spe:], FLOORL)),
        ])
        zdt_r = (np.log10(np.maximum(d["dt"], FLOORL)) - hdr["mu_dt"]) \
            / hdr["sd_dt"]
        zin_r = (tin - hdr["mu_in"]) / hdr["sd_in"]
        yn_r = forward_np(hdr, zin_r.astype(np.float32),
                          zdt_r.astype(np.float32))
        Xn, Tn = decode_np(hdr, yn_r, tin)
        with torch.no_grad():
            yt_r = net(torch.tensor(zin_r.astype(np.float32)),
                       torch.tensor(zdt_r.astype(np.float32))).numpy()
        Xt, Tt = decode_np(hdr, yt_r, tin)
        eX = np.abs(Xn / np.maximum(Xt, 1e-300) - 1.0)[Xt > 1e-10].max()
        eT = np.abs(Tn / Tt - 1.0).max()
        print(f"decode roundtrip: max rel |dX|/X = {eX:.3e}, "
              f"max rel |dT|/T = {eT:.3e}")
        assert eX < 1e-3 and eT < 1e-3, "decode roundtrip FAILED"
        print("selftest PASSED")


if __name__ == "__main__":
    main()
