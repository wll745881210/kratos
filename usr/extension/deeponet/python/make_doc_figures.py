#!/usr/bin/env python3
"""Documentation figures for the Kratos DeepONet chemistry surrogate.

Generates all figures referenced by docs/analysis/12-deeponet-usage.md
and docs/analysis/13-deeponet-performance.md. Data sources are frozen
artifacts only (snapshots, run logs, training JSONs, stats npz); no
simulation is ever re-run.

Usage:
    python3 make_doc_figures.py            # all figures
    python3 make_doc_figures.py fig_speedup fig_tail
    python3 make_doc_figures.py --list

Outputs (PDF + PNG) to <repo>/docs/analysis/figs/deeponet/.
Run from any directory; paths are resolved from constants below.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ----------------------------------------------------------------------
# frozen data locations
# ----------------------------------------------------------------------
SCR = Path("/mnt/scratch/donet_tst/fid")
RUNS = SCR / "runs"
MODELS = SCR / "models"
ANALYSIS = SCR / "analysis"
HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[3] / "docs" / "analysis" / "figs" / "deeponet"
COVERAGE = SCR / "samples" / "coverage_prod_full.png"

# ----------------------------------------------------------------------
# house style
# ----------------------------------------------------------------------
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.grid": False,
        "savefig.bbox": "tight",
    }
)

C_DI = "#333333"        # direct integration (reference)
C_NN = "#B32134"        # NN surrogate (house red)
C_ALT = "#1B5299"       # secondary NN variant (house blue)
C_GREY = "#8C8C8C"


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=180)
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print("wrote", FIGS / f"{name}.png")


# ----------------------------------------------------------------------
# shared loaders (frozen artifacts)
# ----------------------------------------------------------------------

def read_snapshot(path):
    """Load one snapshot via the dataset module (cons + T + flux)."""
    sys.path.insert(0, str(SCR / "samples"))
    import dataset as ds

    return ds.load_snap(path), ds


def load_acc(run):
    d = np.load(ANALYSIS / f"acc_{run}.npz")
    return d


def parse_run_log(run):
    """Per-cycle speed + [dnn] nn/FB counts from a kratos run.log."""
    log = RUNS / run / "run.log"
    cyc, spd = [], {}
    nn, fb = {}, {}
    with open(log) as fh:
        cur = None
        for line in fh:
            m = re.match(r"^cycle\s*=\s*(\d+),.*Speed\s*=\s*([0-9.e+-]+)", line)
            if m:
                cur = int(m.group(1))
                spd[cur] = float(m.group(2))
                continue
            if line.startswith("[dnn] nn "):
                nn[cur] = nn.get(cur, 0) + 1
            elif line.startswith("[dnn] FB"):
                fb[cur] = fb.get(cur, 0) + 1
    t = np.array(sorted(spd))
    return dict(
        cycle=t, speed=np.array([spd[c] for c in t]),
        nn=np.array([nn.get(c, 0) for c in t]),
        fb=np.array([fb.get(c, 0) for c in t]),
    )


# ======================================================================
# usage-doc figures
# ======================================================================

def fig_architecture():
    """Stacked DeepONet: branch (state) x trunk (lg dt) -> basis merge."""
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="black", fs=11):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, style="-|>", lw=1.2):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle=style, lw=lw, color="black"),
        )

    # branch path
    box(0.2, 4.35, 2.2, 0.85, "state $x$\n$[\\lg\\varepsilon, \\lg\\rho, X_i, \\lg F_b]$\n(61)", fs=10)
    box(3.0, 4.35, 2.0, 0.85, "branch MLP\n512 $\\to$ 256", fs=10)
    box(5.6, 4.35, 2.1, 0.85, "coefficients\n(50 $\\times$ 32)", fs=10)
    arrow(2.4, 4.775, 3.0, 4.775)
    arrow(5.0, 4.775, 5.6, 4.775)

    # trunk path
    box(0.2, 0.4, 2.2, 0.85, "$\\lg \\Delta t$\n(trunk input)", fs=10)
    box(3.0, 0.4, 2.0, 0.85, "trunk MLP\n32", fs=10)
    box(5.6, 0.4, 2.1, 0.85, "basis\n(32)", fs=10)
    arrow(2.4, 0.825, 3.0, 0.825)
    arrow(5.0, 0.825, 5.6, 0.825)

    # merge
    box(8.2, 2.4, 1.5, 0.85, "inner\nproduct\n(32)", fs=10)
    arrow(7.7, 4.775, 8.55, 3.25)
    arrow(7.7, 0.825, 8.55, 2.4)

    # output
    box(8.2, 3.6, 1.5, 1.1, "decode\n$X_i'$, $\\varepsilon'$", fc="#F2E6E9", fs=10)
    arrow(8.95, 3.25, 8.95, 3.6, style="-|>")

    # annotations
    ax.text(1.3, 5.45, "branch (stacked per output)", ha="center", fontsize=10, color=C_NN)
    ax.text(1.3, 0.15, "trunk", ha="center", fontsize=10, color=C_ALT)
    ax.text(5.0, 2.75, "575,442 parameters total", ha="center", fontsize=10, style="italic")
    ax.text(4.2, 1.7, "$\\hat y = \\sum_{k=1}^{32} \\beta_k(x)\\, \\tau_k(\\lg \\Delta t)$",
            ha="center", fontsize=12)
    ax.text(0.2, 2.4, "inputs\n($\\in \\mathrm{I\\!R}^{61}$)", ha="left", va="center", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save(fig, "fig_architecture")


def fig_pipeline():
    """End-to-end workflow: offline training + online deployment."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, style="-|>"):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle=style, lw=1.4, color="black"),
        )

    # offline row
    ax.text(0.2, 7.6, "offline (host, PyTorch)", fontsize=11, color=C_DI, weight="bold")
    box(0.3, 5.9, 2.6, 1.2, "sweep.py\nKratos CPU (FP64 DI)\n126 runs $\\times$ 102 snaps", fc="#EDEDED")
    box(3.6, 5.9, 2.6, 1.2, "dataset.py\npair $m \\!\\to\\! m\\!+\\!1$\n1.58M samples")
    box(6.9, 5.9, 2.6, 1.2, "train.py\nphases 1--2 + K=32\nTBPTT rollout", fc="#F2E6E9")
    box(10.2, 5.9, 2.6, 1.2, "export.py\nmodel.dnn (v2)\n+ selftest")
    arrow(2.9, 6.5, 3.6, 6.5)
    arrow(6.2, 6.5, 6.9, 6.5)
    arrow(9.5, 6.5, 10.2, 6.5)

    # online row
    ax.text(0.2, 4.6, "online (GPU, native FP32 kernels)", fontsize=11, color=C_NN, weight="bold")
    box(0.3, 2.0, 2.6, 2.2, "probe (DI)\none solve() attempt\nbudget: dnn_probe_iter", fc="#EDEDED")
    box(3.6, 2.0, 2.6, 2.2, "nn_pack\\ queue\natomicAdd slot\ncell indices only")
    box(6.9, 2.0, 2.6, 2.2, "nn_bat kernel\nbuild inputs from $u$/flx\nforward, guards", fc="#F2E6E9")
    box(10.2, 2.0, 2.6, 2.2, "commit $X_i$, $\\varepsilon$\nexact $u[\\mathrm{ene}]$ update\nor cleanup DI")
    arrow(2.9, 3.1, 3.6, 3.1, style="-|>")
    arrow(6.2, 3.1, 6.9, 3.1)
    arrow(9.5, 3.1, 10.2, 3.1)

    # feedback arrows
    arrow(11.5, 5.9, 11.5, 4.2, style="-|>")
    ax.text(11.7, 5.1, "model.dnn\n(uploaded once)", fontsize=9, ha="left")
    arrow(1.6, 2.0, 1.6, 1.2, style="-|>")
    ax.text(1.8, 1.45, "probe converged $\\Rightarrow$ commit DI (non-stiff cells: DI is faster and exact)",
            fontsize=9, ha="left")
    arrow(11.5, 2.0, 11.5, 1.2, style="-|>")
    ax.text(11.7, 1.45, "guard trip $\\Rightarrow$ exact DI re-solve (fallback)", fontsize=9, ha="left")
    save(fig, "fig_pipeline")


# ======================================================================
# performance-doc figures
# ======================================================================

def fig_speedup():
    """(a) wall-clock bars over model variants; (b) per-cycle speed."""
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), constrained_layout=True
    )

    variants = [
        ("DI (reference)", 205.4, C_DI),
        ("v5", 137.6, C_GREY),
        ("v6b", 38.1, C_GREY),
        ("v7", 199.2, C_GREY),
        ("v8", 29.8, C_ALT),
        ("prod\\_full", 33.7, C_NN),
    ]
    names = [v[0] for v in variants]
    walls = np.array([v[1] for v in variants])
    cols = [v[2] for v in variants]
    x = np.arange(len(variants))
    a.bar(x, walls, color=cols, edgecolor="black", lw=0.6)
    for xi, w in zip(x, walls):
        a.text(xi, w + 3, f"{205.4 / w:.1f}$\\times$", ha="center", fontsize=11)
    a.axhline(205.4, color=C_DI, ls=":", lw=1.0)
    a.set_xticks(x, names, rotation=45, ha="right")
    a.set_ylabel("wall-clock, 100 cycles [s]")
    a.set_yscale("log")
    a.set_ylim(10, 400)

    di = parse_run_log("di_ref_gpu1")
    vf = parse_run_log("nn_vfull_hyb")
    b.semilogy(di["cycle"][1:], di["speed"][1:] + 1e-2, color=C_DI, label="DI")
    b.semilogy(vf["cycle"][1:], vf["speed"][1:] + 1e-2, color=C_NN,
               label="hybrid (prod\\_full)")
    b.set_xlabel("cycle")
    b.set_ylabel("chemistry speed [cell/s]")
    b.set_ylim(1e0, 1e4)
    b.legend(loc="lower right")
    save(fig, "fig_speedup")


def fig_accuracy():
    """Deployment error vs time: median/max |dT|/T and outlier count."""
    v8 = load_acc("nn_v8_hyb")
    vf = load_acc("nn_vfull_hyb")
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11.0, 4.0), constrained_layout=True
    )
    a.semilogy(vf["t"], vf["dT_med"], color=C_NN, label="prod\\_full")
    a.semilogy(v8["t"], v8["dT_med"], color=C_ALT, ls="--", label="v8")
    a.set_xlabel("$t$")
    a.set_ylabel("median $|\\Delta T|/T$")
    a.set_ylim(5e-4, 1e0)
    a.legend(loc="upper right")

    b.plot(vf["t"], vf["n02"], color=C_NN, label="prod\\_full")
    b.plot(v8["t"], v8["n02"], color=C_ALT, ls="--", label="v8")
    b.set_xlabel("$t$")
    b.set_ylabel("cells with $|\\Delta T|/T > 0.2$")
    b.set_ylim(0, 32)
    b.legend(loc="upper right")
    save(fig, "fig_accuracy")


def fig_profiles():
    """Radial T profiles at t = 1, 10, 101: DI vs hybrid."""
    sys.path.insert(0, str(SCR / "samples"))
    import dataset as ds

    base = RUNS
    tsel = [1, 10, 101]
    fig, axes = plt.subplots(
        len(tsel), 2, figsize=(11.0, 7.6), constrained_layout=True,
        sharex=True,
    )
    snaps_ref = sorted((base / "di_ref_gpu1").glob("pl_*.bin"))
    snaps_nn = sorted((base / "nn_vfull_hyb").glob("pl_*.bin"))
    times = [ds.load_snap(s)[1]["time"] for s in snaps_ref]
    x = np.arange(128) / 128.0  # normalized radius (2 blocks x 64)
    for row, t in enumerate(tsel):
        i = times.index(float(t))
        Tr = np.concatenate([b["T"] for b in ds.load_snap(snaps_ref[i])[0]])
        Tn = np.concatenate([b["T"] for b in ds.load_snap(snaps_nn[i])[0]])
        axp = axes[row, 0]
        axp.plot(x, Tr, color=C_DI, label="DI")
        axp.plot(x, Tn, color=C_NN, ls="none", marker=".", ms=3.5,
                 label="hybrid")
        axp.set_ylabel(f"$T$ at $t={t:.0f}$ [K]")
        axp.set_yscale("log")
        if row == 0:
            axp.legend(loc="upper right")
        axe = axes[row, 1]
        dT = (Tn - Tr) / Tr
        axe.plot(x, dT, color=C_NN, lw=1.0)
        axe.axhline(0, color="black", lw=0.6)
        axe.axhline(0.2, color="black", lw=0.6, ls="--")
        axe.axhline(-0.2, color="black", lw=0.6, ls="--")
        axe.set_ylabel("$\\Delta T/T$")
        if row == len(tsel) - 1:
            for c in (axes[row, 0], axes[row, 1]):
                c.set_xlabel("$x$ (block 0 | block 1)")
    save(fig, "fig_profiles")


def fig_tail():
    """eps-ratio evolution of the worst threshold cells (honest limit)."""
    sys.path.insert(0, str(SCR / "samples"))
    import dataset as ds

    base = RUNS
    cells = [95, 30, 31, 94, 91, 26]
    snaps_ref = sorted((base / "di_ref_gpu1").glob("pl_*.bin"))
    snaps_nn = sorted((base / "nn_vfull_hyb").glob("pl_*.bin"))
    rho0, ene0 = ds._units_of_run(base / "nn_vfull_hyb")
    n = min(len(snaps_ref), len(snaps_nn))
    ts, ratio = [], {c: [] for c in cells}
    for i in range(1, n):
        ur = np.concatenate([b["u"] for b in ds.load_snap(snaps_ref[i])[0]])
        un = np.concatenate([b["u"] for b in ds.load_snap(snaps_nn[i])[0]])
        er = ds._eps_cgs(un, ene0, rho0) / ds._eps_cgs(ur, ene0, rho0)
        ts.append(ds.load_snap(snaps_ref[i])[1]["time"])
        for c in cells:
            ratio[c].append(er[c])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    styles = ["-", "-", "--", "--", ":", ":"]
    for c, st in zip(cells, styles):
        ax.plot(ts, ratio[c], st, lw=1.6,
                color=C_NN if c >= 64 else C_ALT,
                label=f"c{c} (b{c // 64}x{c % 64})")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\varepsilon_{\\mathrm{NN}}/\\varepsilon_{\\mathrm{DI}}$")
    ax.set_yscale("log")
    ax.legend(loc="lower left", ncol=2, fontsize=10)
    save(fig, "fig_tail")


def fig_dust_profiles(nn_run="nn_vfull_hyb", snap="pl_00101.bin"):
    """Radial profiles of C(d) / N(d) straight from the Kratos
    snapshots: raw species fields ( n_i * m_a / rho_0, code
    units ) and mass fractions ( u_i * A_i / rho ), DI vs the
    NN-involved hybrid run. x = radial cell index ( cell % 64 ).
    The two y-rows are INDEPENDENT radial columns ( per-column
    rays => row-wise MC draws, guide rc-39 ): solid = row 0,
    dotted = row 1."""
    EL = dict(e=5.49e-4, H=1.00794, He=4.0026, C=12.0107,
              N=14.0067, O=15.9994, Fe=55.845)
    TOK = re.compile(r"(Fe|He|H|C|N|O|e)(\d*)")

    def parse_A(s):
        s = re.sub(r"\(.*?\)", "", s).replace("*", "")
        s = s.replace("+", "").replace("-", "")
        return sum((int(n) if n else 1) * EL[e]
                   for e, n in TOK.findall(s))

    parse_A_fig = parse_A
    sys.path.insert(0, str(HERE / "python"))
    from acc_dust import active_species, load

    spe = active_species(str(RUNS / "di_ref_gpu1" / "planet.par"))
    A = np.array([parse_A(s) for s in spe])
    iC, iN = spe.index("C(d)"), spe.index("N(d)")
    uD, TD = load(str(RUNS / "di_ref_gpu1" / snap))
    uN, TN = load(str(RUNS / nn_run / snap))
    x = np.arange(64)                        # radial index

    fig, axs = plt.subplots(
        3, 2, figsize=(9.4, 8.6), constrained_layout=True
    )
    # (row 0) temperature for context + dust window
    a = axs[0, 0]
    for r, ls, lw in ((0, "-", 1.8), (1, ":", 1.1)):
        a.semilogy(x, TD[r * 64:(r + 1) * 64], color=C_DI, lw=lw, ls=ls)
        a.semilogy(x, TN[r * 64:(r + 1) * 64], color=C_NN,
                   lw=lw * 0.8, ls=ls)
    a.axhspan(1700.0, 1950.0, color=C_NN, alpha=0.12, lw=0)
    a.text(1, 1780.0, "dust window (1700--1950 K)",
           fontsize=8, va="center", color=C_NN)
    a.set_ylabel("$T$ [K]")
    a.legend(("DI row 0", "NN row 0", "DI row 1", "NN row 1"),
             loc="upper left", fontsize=8)
    axs[0, 1].axis("off")

    for row, (idx, Am, name) in enumerate(
            ((iC, A[iC], r"\mathrm{C(d)}"),
             (iN, A[iN], r"\mathrm{N(d)}")), start=1):
        # (left) the raw Kratos field: n_i * m_a / rho_0
        a = axs[row, 0]
        for r, ls, lw in ((0, "-", 1.8), (1, ":", 1.1)):
            a.semilogy(x, uD[r * 64:(r + 1) * 64, 5 + idx],
                       color=C_DI, lw=lw, ls=ls)
            a.semilogy(x, uN[r * 64:(r + 1) * 64, 5 + idx],
                       color=C_NN, lw=lw * 0.8, ls=ls)
        a.set_ylabel(
            rf"$n_{{\mathrm{{{name}}}}} m_a / \rho_0$")
        if row == 2:
            a.set_xlabel("radial cell $x$ (inner $\\to$ outer)")
        # (right) mass fraction u_i A_i / rho
        b = axs[row, 1]
        for r, ls, lw in ((0, "-", 1.8), (1, ":", 1.1)):
            b.semilogy(x, uD[r * 64:(r + 1) * 64, 5 + idx] * Am
                       / uD[r * 64:(r + 1) * 64, 0],
                       color=C_DI, lw=lw, ls=ls)
            b.semilogy(x, uN[r * 64:(r + 1) * 64, 5 + idx] * Am
                       / uN[r * 64:(r + 1) * 64, 0],
                       color=C_NN, lw=lw * 0.8, ls=ls)
        b.set_ylabel(rf"$X_{{{name}}} = u_i A_i / \rho$")
        if row == 2:
            b.set_xlabel("radial cell $x$ (inner $\\to$ outer)")
    save(fig, "fig_dust_profiles")


def fig_radial_profiles():
    """Radial-column profiles of T, C(d), N(d) straight from the
    Kratos snapshots: DI vs NN hybrid, three representative cases
    ( fid; matched MC pair 324 = in-band; pair 322 = lottery loss ).
    Raw fields as written by the code: T from hydro_T_field,
    species = n_i * m_a / rho_0. Solid = t=101, dashed = t=11.
    The two y-rows are independent radial columns ( guide
    rc-39 ): each panel overlays row 0 ( opaque ) and row 1
    ( alpha 0.45, same styles )."""
    from acc_dust import active_species, load

    SAMPLES = SCR / "samples"
    spe = active_species(str(RUNS / "di_ref_gpu1" / "planet.par"))
    iC, iN = spe.index("C(d)"), spe.index("N(d)")
    cols = (
        ("fid ($T_0 = 900$ K)", RUNS / "di_ref_gpu1",
         RUNS / "nn_dagger_hyb"),
        ("MC 324 ($T_0 = 1869$ K, $f = 0.33$)",
         SAMPLES / "run_0324_di", SAMPLES / "run_0324"),
        ("MC 322 ($T_0 = 1227$ K, $f = 3.15$)",
         SAMPLES / "run_0302", SAMPLES / "run_0322"),
    )
    x = np.arange(64)
    fig, axs = plt.subplots(
        3, 3, figsize=(12.6, 9.2), constrained_layout=True
    )
    for j, (title, di, hyb) in enumerate(cols):
        uD, TD = load(str(di / "pl_00101.bin"))
        uN, TN = load(str(hyb / "pl_00101.bin"))
        _, TD1 = load(str(di / "pl_00011.bin"))
        _, TN1 = load(str(hyb / "pl_00011.bin"))
        # T panel
        a = axs[0, j]
        for r, al in ((0, 1.0), (1, 0.45)):
            s0, s1 = r * 64, (r + 1) * 64
            a.semilogy(x, TD[s0:s1], color=C_DI, lw=1.8, alpha=al)
            a.semilogy(x, TN[s0:s1], color=C_NN, lw=1.4, alpha=al)
            a.semilogy(x, TD1[s0:s1], color=C_DI, lw=0.8, ls="--",
                       alpha=0.55 * al)
            a.semilogy(x, TN1[s0:s1], color=C_NN, lw=0.7, ls="--",
                       alpha=0.55 * al)
        a.axhspan(1700.0, 1950.0, color=C_NN, alpha=0.12, lw=0)
        if j == 0:
            a.set_ylabel("$T$ [K]")
            a.legend(("DI (reference)", "NN hybrid",
                      "DI, $t=11$", "NN, $t=11$"),
                     loc="upper left", fontsize=8)
        a.set_title(title, fontsize=10)
        # C(d) / N(d) panels (raw Kratos field)
        for row, idx in ((1, iC), (2, iN)):
            a = axs[row, j]
            for r, al in ((0, 1.0), (1, 0.45)):
                s0, s1 = r * 64, (r + 1) * 64
                yD = np.maximum(uD[s0:s1, 5 + idx], 1e-30)
                yN = np.maximum(uN[s0:s1, 5 + idx], 1e-30)
                a.semilogy(x, yD, color=C_DI, lw=1.8, alpha=al)
                a.semilogy(x, yN, color=C_NN, lw=1.4, alpha=al)
            if j == 0:
                nm = "C(d)" if row == 1 else "N(d)"
                a.set_ylabel(rf"$n_{{\mathrm{{{nm}}}}} m_a / \rho_0$")
            if row == 2:
                a.set_xlabel("radial cell $x$ (inner $\\to$ outer)")
    save(fig, "fig_radial_profiles")


def fig_dust():
    """Dust (C(d), N(d)) accuracy: inventory, dex error, spatial miss."""
    dvf = np.load(ANALYSIS / "acc_dust_nn_vfull_hyb.npz")
    d8 = np.load(ANALYSIS / "acc_dust_nn_v8_hyb.npz")
    t = dvf["snaps"].astype(float)
    iC = list(dvf["spe"]).index("C(d)")
    iN = list(dvf["spe"]).index("N(d)")
    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(11.6, 3.9), constrained_layout=True
    )
    # (a) mass-weighted inventory vs t (the science curve)
    a.semilogy(t, dvf["inv_ref"][:, iC], color=C_DI, lw=1.8,
               label="DI (reference)")
    a.semilogy(t, dvf["inv_nn"][:, iC], color=C_NN, lw=1.4,
               label="NN prod\\_full")
    a.semilogy(t, d8["inv_nn"][:, iC], color=C_ALT, lw=1.2, ls="--",
               label="NN v8")
    a.set_xlabel("$t$")
    a.set_ylabel("mass-weighted $X_{\\mathrm{C(d)}}$")
    a.legend(loc="lower right", fontsize=9)
    # (b) per-cell dex error vs t (both dust species)
    a2 = b
    a2.plot(t, dvf["dex_med"][:, iC], color=C_NN, lw=1.4,
            label="C(d) med")
    a2.plot(t, dvf["dex_p90"][:, iC], color=C_NN, lw=1.0, ls="--",
            label="C(d) p90")
    a2.plot(t, dvf["dex_med"][:, iN], color=C_ALT, lw=1.4,
            label="N(d) med")
    a2.plot(t, dvf["dex_p90"][:, iN], color=C_ALT, lw=1.0, ls="--",
            label="N(d) p90")
    a2.set_xlabel("$t$")
    a2.set_ylabel("$|\\Delta\\lg X|$ [dex]")
    a2.set_ylim(0, 25)
    a2.legend(loc="upper left", fontsize=9)
    # (c) final-snapshot spatial profile, mesh order, per-row
    xs = dvf["x"]
    a3 = c
    for r, ls, lw in ((0, "-", 1.8), (1, ":", 1.1)):
        a3.semilogy(xs, dvf["X_C(d)_ref"][r], color=C_DI, lw=lw,
                    ls=ls)
        a3.semilogy(xs, dvf["X_C(d)_nn"][r], color=C_NN,
                    lw=lw * 0.7, ls=ls)
    a3.set_xlabel("radial cell $x$ (inner $\\to$ outer)")
    a3.set_ylabel("$X_{\\mathrm{C(d)}}$ at $t=101$ (mass frac)")
    a3.legend(("DI row 0", "NN row 0", "DI row 1", "NN row 1"),
              loc="lower right", fontsize=8)
    save(fig, "fig_dust")


def fig_dust_window():
    """Why dust is missed: the NN thermal trajectory skips the
    1700-1950 K dust window. Final snapshot (t=101), per cell."""
    dvf = np.load(ANALYSIS / "acc_dust_nn_vfull_hyb.npz")
    xs = dvf["x"]
    dust = (dvf["X_C(d)_ref"] > 1e-15).any(0)  # either column
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(9.0, 3.6), constrained_layout=True
    )
    # (a) temperature profile vs the dust window, per column
    for r, ls, lw in ((0, "-", 1.8), (1, ":", 1.1)):
        a.semilogy(xs, dvf["T_ref"][r], color=C_DI, lw=lw, ls=ls)
        a.semilogy(xs, dvf["T_nn"][r], color=C_NN, lw=lw * 0.8,
                   ls=ls)
    a.axhspan(1700.0, 1950.0, color=C_NN, alpha=0.12, lw=0)
    a.text(1, 1780.0, "dust window (1700--1950 K)",
           fontsize=9, va="center", color=C_NN)
    a.scatter(xs[dust], dvf["T_ref"][0][dust], s=18,
              facecolor="none", edgecolor=C_NN, lw=1.0, zorder=5,
              label="dust cells (ref, either column)")
    a.set_xlabel("radial cell $x$ (inner $\\to$ outer)")
    a.set_ylabel("$T$ at $t=101$ [K]")
    a.legend(("DI row 0", "NN row 0", "DI row 1", "NN row 1",
              "dust cells (ref)"), loc="upper right", fontsize=8)
    # (b) the yield those cells produce
    for r, ls, lw in ((0, "-", 1.8), (1, ":", 1.1)):
        b.semilogy(xs, dvf["X_C(d)_ref"][r], color=C_DI, lw=lw,
                   ls=ls)
        b.semilogy(xs, dvf["X_C(d)_nn"][r], color=C_NN,
                   lw=lw * 0.8, ls=ls)
    b.scatter(xs[dust], dvf["X_C(d)_ref"][0][dust], s=18,
              facecolor="none", edgecolor=C_NN, lw=1.0, zorder=5)
    b.set_xlabel("radial cell $x$ (inner $\\to$ outer)")
    b.set_ylabel("$X_{\\mathrm{C(d)}}$ at $t=101$")
    b.legend(("DI row 0", "NN row 0", "DI row 1", "NN row 1"),
             loc="upper right", fontsize=8)
    save(fig, "fig_dust_window")


def fig_ensemble():
    """P5 statistical dust gate: matched MC-condition ensemble.

    Per-run C(d)/N(d) inventory ratio (hybrid / pure-DI, same
    ovr.par) over n=5 conditions, with the DI ensemble's own
    spread shown. The deliverable is the DISTRIBUTION: per-run
    dust is a lottery draw in both the hybrid and the reference
    ( guide rc-35/37 ).
    """
    # matched pairs: (DI run, HYB run)
    pairs = [(301, 321), (302, 322), (323, 323), (324, 324),
             (325, 325)]
    samples = SCR / "samples"
    sys.path.insert(0, str(samples))
    from acc_dust import active_species  # noqa: F401
    EL = dict(e=5.49e-4, H=1.00794, He=4.0026, C=12.0107,
              N=14.0067, O=15.9994, Fe=55.845)
    TOK = re.compile(r"(Fe|He|H|C|N|O|e)(\\d*)")

    def parse_A(s):
        s = re.sub(r"\\(.*?\\)", "", s).replace("*", "")
        s = s.replace("+", "").replace("-", "")
        return sum((int(n) if n else 1) * EL[e]
                   for e, n in TOK.findall(s))

    spe = active_species(str(SCR / "planet.par"))
    n_hyd = len(spe)
    A = np.array([parse_A(s) for s in spe])
    idx = {s: i for i, s in enumerate(spe)}
    jC, jN = idx["C(d)"], idx["N(d)"]

    def inv(rd):
        sys.path.insert(0, "/home/lilew/apps/kratos_donet/visual")
        from binary_io import binary_io  # noqa: E402

        f = binary_io(str(samples / rd / "pl_00101.bin"))
        f.open()
        k = "block_0|"
        n_int = int(np.frombuffer(f[k + "hydro_cons_n_int"], "<i4")[0])
        cons = np.frombuffer(
            f[k + "hydro_cons_field"], "<f8"
        ).reshape(-1, n_int)
        f.close()
        rho = cons[:, 0]
        u = cons[:, 5:5 + n_hyd]
        return (u[:, jC] * A[jC]).sum() / rho.sum(), \
            (u[:, jN] * A[jN]).sum() / rho.sum()

    rc, rn, ic, inn = [], [], [], []
    for di, hy in pairs:
        cdi, ndi = inv(f"run_{di:04d}") if di in (301, 302) \
            else inv(f"run_{di:04d}_di")
        chy, nhy = inv(f"run_{hy:04d}")
        rc.append(chy / cdi)
        rn.append(nhy / ndi)
        ic.append(cdi)
        inn.append(chy)
    rc, rn = np.array(rc), np.array(rn)
    ic, inn = np.array(ic), np.array(inn)

    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11.0, 4.0), constrained_layout=True
    )
    xx = np.arange(len(pairs))
    a.axhspan(1e-2, 1e2, color="0.92", zorder=0,
               label="acceptance band ($[10^{-2}, 10^2]$)")
    a.scatter(xx, rc, color=C_NN, s=30, zorder=5,
              label="C(d) ratio")
    a.scatter(xx, rn, color=C_ALT, s=30, zorder=5, marker="s",
              label="N(d) ratio")
    a.axhline(1.0, color=C_DI, lw=1.0)
    a.set_yscale("log")
    a.set_xticks(xx)
    a.set_xticklabels([f"{di}/{hy}" for di, hy in pairs],
                      rotation=20, fontsize=8)
    a.set_xlabel("matched pair (DI run / HYB run)")
    a.set_ylabel("inventory ratio  HYB / DI")
    a.set_title(
        f"median C(d) {np.median(rc):.2f}"
        f"  ({np.mean((rc > 1e-2) & (rc < 1e2)):.0%} in band)"
        f"   N(d) {np.median(rn):.2f}"
        f"  ({np.mean((rn > 1e-2) & (rn < 1e2)):.0%} in band)",
        fontsize=10,
    )
    a.legend(loc="lower left", fontsize=8)
    # (b) the raw ensemble: both draws over matched conditions
    b.semilogy(xx, ic, "o", color=C_DI, ms=6, label="DI C(d)")
    b.semilogy(xx, inn, "o", color=C_NN, ms=6, label="HYB C(d)")
    b.set_yscale("log")
    b.set_xticks(xx)
    b.set_xticklabels([f"{di}/{hy}" for di, hy in pairs],
                      rotation=20, fontsize=8)
    b.set_xlabel("matched pair")
    b.set_ylabel("mass-weighted $X_{\\mathrm{C(d)}}$ at $t=101$")
    b.set_title(
        "the reference itself spans "
        f"{np.log10(ic.max() / min(ic.min(), 1e-300)):.1f} dex",
        fontsize=10,
    )
    b.legend(loc="lower left", fontsize=8)
    save(fig, "fig_ensemble")


def fig_training():
    """Rollout fine-tune curves for v8 / prod_full (+ supervised v2)."""
    j8 = json.load(open(MODELS / "trainlog_prod8.json"))
    jf = json.load(open(MODELS / "trainlog_prod_full.json"))
    jv2 = json.load(open(MODELS / "trainlog_prod_v2.json"))
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11.0, 4.0), constrained_layout=True
    )
    # supervised phase-1/2 of the v2 base training
    a.semilogy([e["ep"] for e in jv2], [e["val_l1"] for e in jv2],
               color=C_GREY, label="prod\\_v2 (supervised)")
    a.set_xlabel("epoch")
    a.set_ylabel("val $L_1$")
    a.legend(loc="upper right")

    b.plot([e["ep"] for e in j8], [e["roll_eps_p90"] for e in j8],
           color=C_ALT, ls="--", marker=".", label="v8")
    b.plot([e["ep"] for e in jf], [e["roll_eps_p90"] for e in jf],
           color=C_NN, marker=".", label="prod\\_full")
    b.axhline(0.1367, color=C_ALT, lw=0.8, ls=":")
    b.axhline(0.2061, color=C_DI, lw=0.8, ls=":")
    b.text(60, 0.141, "v8 init (p90 0.137)", fontsize=9, color=C_ALT)
    b.text(60, 0.219, "v8 pre-fine-tune (0.206)", fontsize=9, color=C_DI)
    b.set_xlabel("epoch (phase 3, K=32 rollout)")
    b.set_ylabel("$K$-step $|\\Delta\\varepsilon|/\\varepsilon$ p90")
    b.set_ylim(0, 0.24)
    b.legend(loc="upper right")
    save(fig, "fig_training")


def fig_routing():
    """Per-cycle nn-eval and fallback counts (probe routing at work)."""
    vf = parse_run_log("nn_vfull_hyb")
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    cyc = vf["cycle"][1:]
    ax.bar(cyc, vf["nn"][1:], color=C_NN, edgecolor="none",
           label="NN commits (of 8 sampled)")
    ax.bar(cyc, vf["fb"][1:], bottom=vf["nn"][1:], color=C_GREY,
           edgecolor="none", label="fallbacks (exact DI re-solve)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("sampled events per cycle")
    ax.set_ylim(0, 9)
    ax.legend(loc="upper right")
    save(fig, "fig_routing")


def fig_coverage():
    """Copy the frozen coverage figure into the docs tree."""
    import shutil

    FIGS.mkdir(parents=True, exist_ok=True)
    shutil.copy(COVERAGE, FIGS / "fig_coverage.png")
    print("copied", FIGS / "fig_coverage.png")


# ======================================================================
# registry
# ======================================================================

FIGURES = {
    "fig_architecture": fig_architecture,
    "fig_pipeline": fig_pipeline,
    "fig_speedup": fig_speedup,
    "fig_accuracy": fig_accuracy,
    "fig_profiles": fig_profiles,
    "fig_tail": fig_tail,
    "fig_dust": fig_dust,
    "fig_dust_profiles": fig_dust_profiles,
    "fig_radial_profiles": fig_radial_profiles,
    "fig_dust_window": fig_dust_window,
    "fig_ensemble": fig_ensemble,
    "fig_training": fig_training,
    "fig_routing": fig_routing,
    "fig_coverage": fig_coverage,
}


def main():
    if "--list" in sys.argv:
        for k in FIGURES:
            print(k)
        return
    keys = [a for a in sys.argv[1:] if not a.startswith("--")]
    for k in keys or FIGURES:
        FIGURES[k]()


if __name__ == "__main__":
    main()
