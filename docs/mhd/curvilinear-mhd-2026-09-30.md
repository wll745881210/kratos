# Curvilinear MHD — implementation & debugging notes (2026-09-30)

Spherical-polar and cylindrical MHD support: integrators, coordinate-singularity
boundary conditions, SMR compatibility, rotating frames, and the `fixd` inflow
boundary.  Commits (main, oldest → newest):

| Commit | Content |
|---|---|
| `5059289` | sph MHD integrator, `dx_cfl`, CT geometry hooks |
| `36f55e8` | hydro eos `dt` uses `dx_cfl` (was `dx_f`) |
| `e3de35c` | pole BC (`pol`), pole-safe CT, Option-C θ-momentum (hydro + MHD) |
| `e8f9855` | cylindrical hydro+MHD integrators, `sfc0[2]` fix, rotating frame |
| `81397a7` | SMR × curvilinear (geometry halo, restriction fixes, emf ordering) |
| `a0962e4` | cylindrical rotation-axis (`axs`) support |
| `e90149d` | `fixd` Dirichlet inflow BC; BC semantics documented |

Test pgens (untracked, `usr_ext/` is its own repo): `usr_ext/std_tst/mhd_sph_tst`,
`usr_ext/std_tst/mhd_cyl_tst`, `usr_ext/std_tst/hydro_cyl_tst`.

---

## 1. Architecture

### 1.1 Integrator hook pattern

Curvilinear integrators are CRTP leaves on `dynamics::mhd::integrator::rk2_t`
(resp. hydro) that override small hooks rather than whole kernels:

- `edg_f(ax, i, geo)` — edge length for the CT curl.  sph: axis-1 edges ×`r`,
  axis-2 edges ×`r·sin θ_f` (with `|sin θ_f| < 1e-6 → 0` pole snap);
  cyl: axis-1 (φ) edges ×`r_f`.  Cartesian default = `dx_f` — **bit-identical**.
- `sfc_thin(a, i, geo)` — face area used for thin axes (`n_dim < 3`).
- `flux_to_du_*` — finite-volume divergence; carries the metric rescaling.
- `src_geo(...)` — geometric source, appended after the flux loop.

`src/mesh/geometry/geometry.h::dx_cfl(a, i)` returns the CFL cell extent:
uniform-Cartesian fast path `dx0[a]` (bit-identical), else `V / max(A_l, A_r)`.
Both eos `dt` reductions use it (hydro `eos/adiabatic.h:154`,
mhd `eos/adiabatic.h:118`).  Using the coordinate `dx_f` instead is a real bug in
curvilinear geometries: at small radii the θ-arc `r dθ` is ~10× smaller than the
coordinate width and the run SIGFPEs on the CFL.

### 1.2 Momentum-equation consistency (Option C)

The θ-momentum r-flux may be rescaled to the *tensor* form
(`flux × r_f/r_c`, so `Σ r·ρv_θ` telescopes = slice angular momentum), in which
case the geometric source must contain **only** the `ρv_φ² cotθ/r`
(− `B_φ² cotθ/r`) term: the `−ρv_rv_θ/r` (+ magnetic) part is already carried by
the rescaled flux.  Keeping both (as the pre-2026-09 code did) double-counts
`−T_{rθ}/r` and produces a non-convergent `sin 2θ / r` momentum error.

Chosen: **Option C** (keep the r³ tensor flux, drop the component-form source
terms) in both hydro and MHD sph integrators; conserves discrete slice angular
momentum exactly and passes the uniform-Cartesian-vector probes to the same
truncation level as the component form.  The MHD `src_geo` keeps the
cell-centered pointwise form for everything else:

```
d_mom_r  += dt·( 2 p* ·cm0 + ρ(v_θ²+v_φ²)/r − (B_θ²+B_φ²)/r )
d_mom_θ  += dt·( p* θ-moment + (ρv_φ²−B_φ²)cotθ/r )          (Option C)
d_mom_φ   = 0     (the φ flux rescaling transports total angular momentum)
```

with `cm0 = 3(r_r²−r_l²) / (2(r_r³−r_l³))` = exact `⟨1/r⟩` moment and
`p* = p + B²/2`.  Athena++ face-averages some source terms; kratos keeps them
cell-centered, which avoids their known geometric-source issue by construction.

Cylindrical: `d_mom_r += dt/r_c·(p* + ρv_φ² − B_φ²)` with `r_c = (r_l+r_r)/2`
(the exact `⟨1/r⟩` moment in cyl), plus the φ-momentum r-face rescale
`x_fc(0, i[0]+j) / r_c` in `flux_to_du_cyl` (discrete angular momentum).

### 1.3 CT at coordinate singularities

- **Zero-area faces**: `intg_field` skips the emf update when the face area is
  `<= 0` (the pole/axis faces; their B is gauge).  Bit-neutral elsewhere.
- **Ring emf averaging**: on the axis/pole all φ cells share one physical edge,
  so the emf there must be φ-independent — `pole_emf_k` (sph) and
  `cyl_axis_emf_k` (cyl) replace `ee` on the ring by its φ-mean.
- **Ordering matters**: the averaging must run **after** the SMR E-field edge
  exchange (`act_flx`), otherwise the exchange overwrites the averaged values
  and divB leaks at fine-coarse pole edges.  Both kernels are therefore launched
  from an `intg_flx` override (post-exchange, pre-CT-update), *not* from
  `calc_flx`.
- Discrete divB is conserved to the float-storage floor through blasts crossing
  the pole/axis (see §4).

### 1.4 Boundary functor mechanics

**The functors operate on the primitive `b_d.hyd.w`, not conserved `u`.**
The staging (`mhd/boundary/phys.h`) copies `hyd.w` into per-thread column
buffers; a Dirichlet functor must therefore write `(rho, pre, vel)`, not
`(rho, ene, mom)` — writing conserved values into the primitive stage produced
a few-% boundary pressure error that took a dump-probe to diagnose.

Staging lattices (MHD): the normal B component comes from the **face** lattice,
tangential components from the **cell** lattice (they differ only on the right
side: `n_ceff−i` vs `n_ceff−i−1`).  Functors set signs per component; the base
writes ghosts `i = −n_gh..−1`.  `pin_bnd_face = true` additionally writes
`bf = 0` on the boundary face row itself (needed at the pole/axis: c2p reads
that face, and tangential-direction BCs copy interior values into its corner
columns — BC execution is axis-major, so a later pole/axs pass wins the corner
slots deterministically).

Boundary kind semantics (now documented at the enroll sites):

| Kind | Meaning |
|---|---|
| `ref` | reflect; flips normal momentum |
| `out` | zero-gradient **+ inward-momentum clamp** — outflow-only by design; using it where flow enters gives O(1e-1) density errors (the old "inflow artifact") |
| `fre` | free zero-gradient extrapolation — lets inflow through |
| `fixd` | Dirichlet: ghost = `[init]` ambient medium (`rho0, pre0, vel0, b0`), filled by the keeper at init |
| `pol` | spherical pole (θ sides only): v_r/B_r even, v_θ/v_φ/B_θ/B_φ flip; pins B_θ=0 on the pole face |
| `axs` | cylindrical axis (r sides only): v_z/B_z even, v_r/v_φ/B_r/B_φ flip; pins B_r=0 on the axis face |
| `per` | periodic (handled by the mesh, not the functors) |

`pol`/`axs` use the same-φ wedge continuation — **exact for axisymmetric
fields**; tilted-field exactness would need a φ+π remap (cross-block; deferred,
the `rebin_t` machinery is the production route for non-axisymmetric polar runs).

### 1.5 Rotating frame (`omega_0`)

Lab-frame round-trip in `flux_to_du_*`: augment the φ-momentum flux with
`ω·lever·F(ρ)` (lever = `r_f` on r-faces), apply the metric rescale, add the
`ene` cross terms, then transform the updated conserved state back
(`u[mom_φ] -= res·ωr`, `u[ene] += res·(ωr)²/2` …).  The centripetal source uses
the **lab** velocity `v_φ + ω r`.  Validated by the counter-rotating test
(frame +ω, gas −ω r ⇒ lab-static): machine-exact on CPU and GPU, B frozen.

### 1.6 SMR × curvilinear (`81397a7`)

- **Geometry halo**: the SMR restriction stencils reach `2·order` rows normal to
  an interface (into the fine block's own ghost halo for thin blocks) and the
  tangential boundary face rows.  `sfc`/`vol` are now allocated with ghost
  padding and filled analytically; `xf`/`xv` carry one extra slot so the
  boundary ghost *face* coordinate exists.  Dump format unchanged — `read()`
  extrapolates the extra slot for legacy dumps.  Multigrid coarsening loops
  shifted to logical indices (its curvilinear path was and remains incomplete).
- **`to_coarser` face-B restriction fixes**: right-side off-by-one (read a ghost
  face row), tangential boundary receiver rows now skip the component that owns
  no such face row, and zero-area (pole) footprint rows divide safely.
- Cartesian SMR gate (`cvf8`) bit-identical after all of the above.

---

## 2. Grid & geometry facts

- `geo.surface(i, a)` = `sfc0[a] > 0 ? sfc0[a] : sfc[a](i)`; `sfc0 > 0` is the
  constant-area fast path.  The old cylindrical meshgen set `sfc0[2] = 1` in the
  uniform branch → all z-faces reported area 1.0 (fixed to −1).
- sph areas: r-face `r_f²·|cos θ_l − cos θ_r|·dφ`; θ-face `(dr²/2)·sin θ_f·dφ`
  (with `|sin θ_f| < 1e-6 → 0` pole snap); φ-face `(r_r²−r_l²)/2·dθ`.
  Volume `(r_r³−r_l³)/3·|cos θ_l−cos θ_r|·dφ`.
- **float π**: `sin(3.14159274f) = −8.74e-8` — exact `== M_PI` comparisons fail
  in P1 (and `sin(π_double) = +1.22e-16` fails the `sfc > 0` guard in P2).  Use
  tolerance snaps on both faces and edges.
- BC / comm kernels require `n_ceff[a] % min(n_ceff[a], bnd_thread_limit) == 0`
  on the in-plane axes.  The phys path throws
  `"Indivisible stride, boundary::phys_base_t."`; the comm path has **no guard**
  and crashes with a TBB recursive `std::out_of_range`.
  Workaround: `bnd_thread_limit = 8` divides typical block sizes.
- SMR regions: `reg.x` is stored float32 in P1 — write `3.14159`, not full π, in
  `[refine_region_*]` or the binary search throws "Incorrect SMR region".

## 3. Debugging playbook (hard-won)

- **Binary provenance is the #1 trap.**  Default `ARCH=CUDA`; a CUDA/HIPCPU
  switch without `make clean` links a franken-binary that runs and produces
  plausible garbage on GPU.  A failed link *deletes* `bin/kratos`.  Verify GPU
  binaries with `cuobjdump --list-elf bin/kratos` (must list a `.cubin`).
  Work in a symlink-farm workspace (`~/apps/kratos_mhd_coord`,
  `/tmp/opencode/kbuild_cpu`), never build inside the Seafile-synced repo, never
  touch `~/apps/kratos_mhd` (the other session's tree).  A repo `bin/kratos`
  shim forwarding to the workspace binary prevents stale-binary accidents.
- **Single build dir = single config.**  Changing `USRDIR`/`PRECISION` needs no
  clean (only `usr.cpp` / affected TUs rebuild), but ARCH does.  Snapshot
  binaries (`cp bin/kratos /tmp/.../kratos_<tag>`) before switching.
- **pgen/binary mismatch is silent**: pars for a different pgen "run" if the
  mode numbers exist — check the make link line's `usr.o` path, or the dump
  record width (`nint` = 8 MHD vs 5 hydro) as a tell.
- **Dumps**: host dumps are **double** regardless of PRECISION; `bf` is a padded
  `(n+5)³` box with interior faces at offset +2 (per-component face boxes:
  `(n0+1,n1,n2)`, …); `dt` is stored float32 in P1 by design; `mesh_tree`
  contains one run-to-run nondeterministic byte (exclude from bit-gates).
- **divB checks need the exact code areas** — a `sin θ_c·dθ` approximation for
  the r-face area injects a fake 3e-6 "divergence" at init.
- **`t_lim` float-ulp crumbs (P1)**: the final clamped dt is stored as float, so
  `t` lands ~6e-8 short and the next 2–3 cycles take geometrically shrinking
  "crumb" steps before terminating.  Benign; recognize the signature
  (`dt ~ 1e-10 → 1e-17` at `t ≈ t_lim`) instead of chasing a "CFL collapse".
- **gdb on TBB/GPU-CUDA**: `catch throw` lands in the `dat_3d` bounds check with
  the offending index in scope; DEBUG HIPCPU builds + a minimal repro are the
  fast path (user workflow: big tests on CUDA, debug minimal cases on CPU).
- **init dump artifact**: base `mhd_t::init_cond` sets `bf` and `ene` but not
  `u[bcc]`; dump 0 shows `bcc = 0` until the first `cons2prim` fills it from the
  faces.  Cosmetic; pgens should set `bcc` themselves.

## 4. Validation summary (CUDA P1 on GPU1 unless noted)

| Test | Result |
|---|---|
| sph uniform / rotating frame | machine-exact (mom ~1e-9) |
| sph toroidal-B_φ pinch | mom_r exact to −dt·B²/r (1.3e-4 rel) |
| sph uniform B_z (pole domain) | interior frozen ~1e-5..1e-8; convergent after Option C |
| sph blast crossing pole | divB conserved at floor (8e-9 CPU P2; 1e-5 GPU P1 = reader floor) |
| sph SMR, pole-touching region | divB 2.11e-6 constant over 416 cyc (was 2.7e-5 before the emf-ordering fix) |
| sph linear wave battery (mhd_lw) | see §5 note |
| cyl uniform / B_z / pinch / v_z | machine-exact incl. exact φ-seam; divB ≡ 0 |
| cyl blast (annular + on-axis) | divB 6.8e-16 → 2.9e-7 (P1 floor); mom_φ = 0 exactly |
| cyl field loop / Alfvén | loop advects at exact v=1; Alfvén interior L1 rate 2.60× |
| cyl counter-rotating frame | machine-exact, GPU bit-matches CPU |
| Cartesian gates (cvf8 SMR, alf2d thin-z, sod) | bit-identical |
| `fixd` inflow (uniform v_z) | drho 3.3e-8 (vs 4.4e-2 with `out`, 2.7e-8 with `fre`) |

## 5. Known limitations / open items

- **r = 0 in spherical: deliberately unsupported.**  The through-origin
  continuation `(r,θ,φ) → (−r, π−θ, φ+π)` is a non-local 2D remap the
  per-column BC staging cannot express, and the CFL collapses as `dr → 0`
  anyway.  Use `r_min > 0` with a physical inner BC (planet_atm pattern).
- Tilted (non-axisymmetric) fields at the pole/axis are wedge-approximate;
  exactness needs a φ+π source remap or φ-ring rebinning (`rebin_t`).
- Pole/axis emf averaging is **block-local** in φ; full-φ multi-block pole caps
  need the rebin-style cross-block variant.
- Comm path lacks the divisibility guard (§2); add the phys-style throw.
- Curvilinear multigrid (self-gravity in sph/cyl): feasible — geometry tables
  now cover the halo; zero-area faces self-close the Poisson operator at
  singularities; anisotropy near the axis will likely demand line smoothers.
  Deferred until a physics driver needs it.
- `dode_tst`'s own `sph_pol_t` zeroes mom_φ at the pole (vs flip in `pole_t`) —
  a production choice, different semantics, left alone.
