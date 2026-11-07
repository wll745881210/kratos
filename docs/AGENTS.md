# AGENTS.md — Kratos

**Kratos** is a multi-physics astrophysical simulation framework designed for heterogeneous computing
devices (GPUs). It solves compressible fluid dynamics (hydro/MHD), non-equilibrium thermochemistry,
ray-tracing radiation, self-gravity (multigrid Poisson), and tracer particles on block-structured AMR
meshes.  C++17, CUDA/HIP/MUSA, optional MPI.

> **Papers**: Wang (2025), ApJS, 277:63 — fundamentals & hydro.
> Wang (2025, draft) — thermochemistry & reacting flow.
> Yang & Wang (2026, draft) — polarized Monte Carlo radiative transfer.

## Build

GNU Make only (no CMake). The single binary is `bin/kratos`. You **must** specify which
problem generator (pgen) to compile via `USRDIR`.

```bash
make USRDIR=usr_ext/bin_bh DEBUG=0 ARCH=CUDA MPI=0 PRECISION=1
```

| Variable     | Options                          | Notes                            |
|--------------|----------------------------------|----------------------------------|
| `USRDIR`     | `usr` (default), `usr_ext/<pgen>` | Problem generator directory — **required** to compile a specific pgen |
| `ARCH`       | `CUDA` (default), `HIP`, `HIPCPU`, `MUSA` | `HIPCPU` = HIP-CPU runtime (g++-10). |
| `DEBUG`      | `0` (default), `1`               | Debug adds `-g -O0`, defines `__GPU_DEBUG__`/`__CPU_DEBUG__`, force-includes `debug_macros.h`. |
| `MPI`        | `0` (default), `1`               | Links `-lmpi`.                 |
| `PRECISION`  | `1` (default), `0`, `2`          | `0`=all float, `1`=float/float safe, `2`=all double. See `src/types.h`. |

- `make clean` removes `obj/` and `bin/`.
- `make remake` = `clean` then `all`.
- Sources are **auto-discovered** from `src/`, `$(USRDIR)/`, and `usr/extension/`. No source
  lists to maintain. Only one pgen can be compiled at a time.

## Entry point and plugin model

`src/main.cpp:11` calls `prob::run(argc, argv)`. The default (`src/user/probgen.cpp:18`) is a
**weak symbol** — users **must** supply a strong override or the binary will print an error and
exit.

Each problem/test lives under `usr_ext/` or `usr/` and provides its own `prob::run()` that creates a
`mesh::mesh_t`, enrolls modules (device, comm, cycle, meshgen, physics), reads input, and launches
`mesh.evolve()`.  Core sources under `src/` are not meant to be modified for problem setup.

**Module container:** `mesh::mesh_t` is a container that holds only geometry — it has no awareness
of which physics modules are enrolled. Users enroll exactly the modules they need. With only the hydro
module, Kratos is a hydro code. With only radiation modules, it is a pure radiative transfer code
(see `usr_ext/pol_rad`). A single binary can implement any subset of the available physics.

Usage: `bin/kratos <param_file> [override_file ...]`

## Parameter files

INI-like format. Sections are `[section]` headers, followed by `key = value` lines:

```
[mesh]
x_min         =   0   -1   -1
x_max         =   1    1    1
n_cell_global = 512   2    1

[dynamics]
gamma         = 1.4

[boundary]
kinds = fre out out out out out
```

Internally, the `input` class (`src/io/args/input.h`) stores entries as `section|key` → value,
so programmatic access uses `args.get<T>("section", "key")` or `args("section", "key", var)`.
Lines starting with `#` are comments. See `usr_ext/std_tst/sod/sod.par` for a complete example.

## Key source directories

| Directory                  | Purpose                                           |
|----------------------------|---------------------------------------------------|
| `src/`                     | Core library (mesh, device, comm, io, modules)    |
| `src/device/`              | GPU abstraction (CUDA/HIP/MUSA)                   |
| `src/comm/`                | MPI/serial communication                          |
| `src/io/`                  | Config parser, binary I/O                         |
| `src/mesh/`                | Block-structured AMR mesh, cycle, module base     |
| `src/modules/`             | Physics: dynamics (hydro/MHD), chemistry, multigrid, particle |
| `src/modules/dynamics/mhd/nonideal/` | **The non-ideal MHD line**, moved back here 2026-11-06, in two subdirectories: `explicit_sts/` (the explicit / super-time-stepping half — `nonideal.h`/`.cpp` with the `need_ni_ee` seam, `implicit.h`, `eta_tensor.h`) and `edge_mg/` (the edge-centred EMF implicit step, `ee_method = 5`, matrix-free device MG; its test-only reference (oracle) lives in `test/edge_mg_ref/`), and inside `edge_mg/` a `kernels/` subdirectory — the only place `__global__` may appear for the solver. Its gates and pars stay in `usr_ext/mhd_ee/`. |
| `usr/extension/`           | Reusable extensions: radiation, chem_therm, hydro proxy |
| `usr_ext/`                 | Problem configurations (each subdir provides `prob::run()`) |
| `visual/`                  | Python postprocessing scripts (`hydro_data.py`, `slice_plot.py`, etc.) |
| `backup/`                  | Archived code from previous versions (not compiled) |
| `docs/`                    | Documentation, layered by intent: `methods/` (paper drafts, PDF), `mhd/` (MHD & non-ideal topic docs), `history/` (archived process records; the live state of the non-ideal line lives in `mhd/nonideal/`), `analysis/` (cross-module architecture reports), `radiative_transfer/` (the RT line: `scheme-comparison.md` (DDMC vs FLD/M1 survey + roadmap), `fld-stage1-plan.md` (the staged implementation plan), `fld-method.md` (the shipped gray-FLD MG module's method + validation), `fld-ppd-integration-plan.md` (the staged plan for wiring the FLD into the ppd solver's IR reprocessing + dust temperature)), plus the repo-level guides below |
| `docs/methods/method_0.pdf`        | Published method paper: Wang (2025), ApJS, 277:63           |
| `docs/methods/method_1.pdf`        | Method paper draft: thermochemistry & reacting flow (2025)  |
| `docs/methods/method_polrad.pdf`   | Method paper draft: polarized MCRT (Yang & Wang 2026)      |
| `docs/Chemistry.tm`        | Chemistry variable conversion notes (TeXmacs format) |
| `docs/mhd/curvilinear-mhd-2026-09-30.md` | Spherical/cylindrical MHD: integrators, pole/axis BCs, SMR, debugging notes |
| `docs/mhd/pole-rebin.md` | **The polar φ-rebin of the MHD state** (full-polar-domain spherical runs): the φ-group average of the conserved state plus the DIVERGENCE-PRESERVING face-field rebuild (group-averaged r/θ faces, linear φ-face ramp), the `[mesh] pole_rebin_*` knobs, the axis-only gating, and the sp32 validation numbers (2026-10-17) |
| `docs/mhd/nonideal/`      | **The implicit non-ideal MHD line** (subdirectory): the entry points `STATUS.md` (current state + open problems) and `dead-ends.md` (the falsified-hypothesis index), the method / user / SMR-transfer docs, and the literature surveys |
| `docs/mhd/nonideal/STATUS.md` | **The single entry point for the line**: the support matrix (what runs and what does not), the isolation criteria to use instead of chaotic-state absolutes, the open problems **P1–P7**, the knob/trap table, and the doc map (2026-10-21) |
| `docs/mhd/nonideal/dead-ends.md` | **The falsified-hypothesis index**: every abandoned approach (the 2.5D/thin-axis family, `mg_cc_exact`, the item-0 boundary surgery, the PPD+`n_φ=1` blow-up) as one assumption → outcome + evidence + date, plus the methodology traps M1–M5 (2026-10-21) |
| `docs/mhd/nonideal/edge-emf-charter.md` | **The CHARTER for the edge-EMF route** (the unknown moved from face-centred B to edge-centred E, so that `δB = C·E` ⇒ `D·δB = 0` by construction; 2026-10-26): the gates P0′→P5 with written-down thresholds, **the RESERVE section that freezes the face-centred stack** (freeze list + golden regression), **the retention/isolation of the divergence-cleaning projection** (kept even for the edge route, because error still accumulates), the do-not list, and the anti-drift rules + deviation log. **Authoritative for the edge work**; `emf-unknown-route-plan.md` is now only its derivation archive |
| `docs/mhd/nonideal/feec-single-operator-mg-plan.md` | The FEEC single-operator (exact-row) plan |
| `docs/mhd/nonideal/implicit-nonideal-mg-method.md` | **The implicit non-ideal MHD (curl–curl MG) implementation document** — the design, BC framework, validation evidence, and limitations (2026-10-07) |
| `docs/mhd/nonideal/nonideal-userguide.md` | **Operational user guide for the implicit non-ideal MHD (curl–curl MG) module**: quickstart, pgen/provider integration recipe (chemistry plug-in point), parameter reference, support matrix, debug workflow |
| `docs/mhd/nonideal/smr-interface-transfers.md` | **The 2:1 block-interface transfer reference**: lattice alignment (cells/faces/edges, ghost vs slave slots), restriction/prolongation (state, flux reflux, CT field, MG V-cycle), the ni path's coefficient fills (rfill/xfill) and slave-face rules (ctrow/slv/corner) — every algorithm with file:line citations (2026-10-11) |
| `docs/mhd/nonideal/nonideal-placement-decision.md` | The 2026-10-11 decision to keep `src/modules/dynamics/mhd/nonideal/` in the src tree (the `fld_t` reverse-coupling audit; deferred pending the provider-interface narrowing) |
| `docs/mhd/nonideal/literature/` | The literature-survey corpus behind the line (migrated from the scratch `reference/`, 2026-10-11): `non-ideal-implicit.md` (operator structure/singularity), `non-ideal-mixed.md` (explicit-implicit spatial mixing), `divfree_mg_survey.md`, `bc_noniterative_survey.md`, `bc_mg_boundary_survey.md` (+`_review`), `magnetic_diffusion_physical_bc_survey.md` (+`_review`), `face_centered.md`, `curvilinear_smr.md`, `curvilinear_mg.md`, `hall-implicit-survey.md` (H0), `hall-second-order-survey.md` (H4), `implicit-tensor-nonideal-curvilinear-survey.md` (the P1 gap survey: per-code time integrators/coordinate support, STS-vs-implicit verdict, tensor curl–curl preconditioners, degenerate-axis/metric-identity evidence, IMEX alternatives — primary sources + Athena++ source check, 2026-10-21), **`ct-implicit-tensor-diffusion-survey.md`** (**the Python-side algorithm charter**: the two cross-validating routes — face-centred CT implicit tensor-η diffusion + MG ／ cell-centred implicit tensor-η diffusion + potential projection + MG — with the verified cell complex and metric formula, the literature to port per ingredient, the validation ladder, and the literature gaps; **spherical is the target**; 2026-10-28) |
| `docs/mhd/nonideal/history/` | The fix/diagnosis records: `sph_smr_regression_bisect.md` (the spherical-SMR bisect, §7.45), `diagnoses_reply.md` (the explosion diagnoses) |
| `docs/history/`            | **Archive**: the per-session process chronicles of the interface/race fixes were taken off the reading path on 2026-10-21 (they live on in git — recover one with `git log --diff-filter=D -- <path>`);  what sits here now is `agents-changelog-archive.md` (the AGENTS.md changelog moved out whole) and `fix-geo-loc-fix-2026-07-27.md` (the geo-loc fix note) |
| `docs/README-USER.md`      | User guide: framework, module interactions, pgen composition |
| `docs/analysis/`           | Codebase analysis reports: architecture, conventions, doxygen plan |
| `docs/STYLE_GUIDE.md`      | **The authoritative C++ typesetting/style guide** (column limit, parenthesis spacing, `for`/`while`/`if` **and `if`/`else` brace placement**, `__device__` conventions, naming, include order). `docs/analysis/04-coding-conventions.md` is an analysis report and **defers to this file** |
| `usr_ext/bin_bh/`          | Binary star system pgen with analytic gravity, sinks, outflows |
| `usr_ext/pol_rad/`         | Pure radiation pgen: polarized MCRT (no hydro enrolled) |
| `usr_ext/planet_atm_sph/`  | Multi-module: hydro + chemistry + radiation in spherical coordinates |

**Documentation discipline.**  A newly verified fact goes into the line's `STATUS.md`;  a
falsified hypothesis gets one row in `dead-ends.md`;  narrative and process records go to
`docs/history/`.  Append-style corrections that leave a superseded claim in place are not
acceptable — correct or delete the claim where it lives.

**Code style** is governed by `docs/STYLE_GUIDE.md` — authoritative for the column limit,
parenthesis/angle-bracket spacing, control-flow braces (`if`/`else` braces go on **their own
line**, and `else` **never** shares the closing-brace line), **§ 3.4** (every block's brace on its
own line; a control-flow body is **never** on the header line, even for one statement),
**§ 5.2** (`__forceinline__` only on small accessors, not on large device functions),
**§ 6.3** (where a function body lives: `__device__`/`__host__ __device__`/`__global__` stay in the
header;  a `__host__`-only non-template body is declared in the header and defined in the matching
`.cpp` — one `.cpp` per header,  **no size exemption**;  a `__host__` template body stays in the
header and is placed by length:  ≤ 15 lines in-class,  16–120 below the class,  > 120 in a
`*_impl.h`),
`__device__` conventions, naming and include order.  `docs/analysis/04-coding-conventions.md` is a
report and defers to it.

## Test / verification

No test runner. `test/sod/sod.cpp` is a standalone problem that overrides `prob::run()`. To
compile a test you must link it against the core sources — there is no test Makefile target. Tests
under `usr_ext/std_tst/` serve the same purpose (sod, kh, bow_shock, linear_wave, etc.).

### Debug workflow for long-run / large-scale tests

1. **Run production tests with CUDA builds first** (`make USRDIR=<pgen> MPI=1 -j12`).
   Do not use `ARCH=HIPCPU DEBUG=1` for long runs or many-cell grids — the HIP-CPU runtime at `-O0`
   is orders of magnitude slower.
2. **On problems/bugs** (NaN, illegal memory access, unexpected aborts): first set
   `except_ferr = 1` in the parameter file's `[cycle]` section (traps floating-point
   errors / device faults at the source), then clean-rebuild the same pgen with the debug
   CPU target, `make clean && make ARCH=HIPCPU DEBUG=1 USRDIR=<pgen> -j8` (always
   `make clean` when switching `ARCH`), then debug under GDB started with `catch throw`
   and `break abort` — this catches C++ exceptions (e.g. `std::runtime_error` from input
   validation) and hard aborts (CUDA/HIP device-side errors surfaced as `abort`) at the
   throwing frame.
3. **After the check**, if longer / higher-resolution runs are needed, switch back with
   `make clean && make ARCH=CUDA USRDIR=<pgen> -j12` (again: always `make clean` on an
   `ARCH` switch).
4. **`CUDA Error: invalid argument`** (and every other `cuda_api` failure): do NOT switch
   to `HIPCPU` — stay on `ARCH=CUDA`, clean, rebuild with `DEBUG=1`, and run under GDB
   with `catch throw` / `break abort`.  In a release build the wrapper
   (`src/device/cuda/cuda.cpp`) prints the STALE return value of the API call;  with
   `DEBUG=1` it synchronizes and re-reads `cudaGetLastError()`, so the message and the
   GDB stop belong to the call that actually failed.  `ARCH=HIPCPU DEBUG=1` only
   instruments the CPU side (no GPU there) and cannot localize a CUDA API failure.
   (Workspace build notes: the farms' `BUILD_GUIDELINES.md`.)

## Binary I/O

Snapshots use a custom binary format defined in `src/io/binary/binary_io.h`. The Python reader is
`visual/binary_io.py`; the high-level API is `visual/hydro_data.py` (`get_dd()`, `get_last_dd()`).
Default output file suffix is `.bin` (set in `src/mesh/cycle/cycle.cpp:242`).

## Namespace conventions

`mesh::`, `device::`, `comm::`, `dynamics::`, `chemistry::`, `multigrid::`, `particle::`,
`prob::`, `utils::`, `type::`.

## Device dispatch convention (no runtime type resolution on the device)

Kratos resolves policy types **on the host, once**, and instantiates one kernel per
derived type.  The canonical shape is `src/modules/dynamics/prototypes/integrator.h`
plus `src/modules/dynamics/hydro/integrator/integrator.h`:

- the `__global__` kernel takes the **policy objects as parameters**
  (`flux_kernel< dat_T, bdt_T, rec_T, rie_T, eos_T, itg_T >`), so the policy is a
  compile-time template argument — the kernel body never branches on a runtime tag;
- the **launch function lives in the CRTP base**
  (`hydro::integrator::base_t< derived_T >::calc_flx`, `integrator.h:289-308`): it does
  `const auto& f = get_self();  dev.launch( flux_kernel< dat_t >, ..., f );`,
  i.e. **the derived object passes itself into the kernel template**;
- derived types supply **only `__device__` policy functions**
  (`reconstruct` / `riemann` / `add_flux` / `flux_to_du` in `sph.h` / `cyl.h` / `rk2.h`);
- the runtime choice (if any) is a **host-side virtual** on a plain abstract base
  (`dynamics::integrator::base_t`, `prototypes/integrator.h:37-62`) or a host-side
  factory — never a `switch` / `if` on a type tag inside device code.

Consequences to respect:

- the policy object is copied **by value** into the kernel: keep it trivially
  copyable (POD scalars only); put every pointer and buffer in a POD view struct
  (e.g. `ctx_t`) that is itself trivially copyable;
- "two backends" means **two derived types + one common base**, not one type with an
  `if` inside; shared pieces (vector ops, dot, residual, convergence test, warm
  start) exist exactly once, in the base.

**The second half of the convention — the data side.**  A device backend must not
carry per-step host round-trips.  The non-ideal line's `ee_method = 5`
(`src/modules/dynamics/mhd/nonideal/edge_mg/edge_mg_mf.h`) is the worked example:  topology, the
prolongation operators, the per-level blocks and the per-dof cell counts are built
**once** (a host `hier_t::build` -- which used to be `mg_t::build` -- hands the device its
numbering,  its prolongations and its per-dof cell counts), and a step is a
sequence of kernels over device-resident buffers, ending in at most O(1) scalars read
back (`it`, `res`, `dt`).  Its gate (`usr_ext/mhd_ee/gate_mf.cpp`, `test_mode 15`)
does not trust a hand-written byte counter:  it poisons the host mirrors with a
sentinel before the step and asserts they are untouched afterwards (see
`dead-ends.md` M81).

**A gate has to be able to see the error it is named after.**  The boundary load
(`assemble_load_cell`) used to have no criterion that could see a small *multiplicative*
error: the v4 family's convergence rate passed at 1.225, and gate A1's load-identity
residual stayed bit-identical under both a `1.001` scale and a normal-sign flip --
because A1's ladder loads the *reference* assembly, so the production path was never in
its sight.  A1 now compares the production assembly against P4.2's independent
orthogonal-frame formula **cell by cell** (**2.6e-16** on 320 and 5888 wall cells) and
judges the load identity against a band of `10 h^2` (the measured order is `h^2.4`);
both mutations go red (`mhd/nonideal/STATUS.md` (93),  `dead-ends.md` M84).

## Index conventions

Hydrodynamic variable indices are always qualified with `idx::`:
```cpp
u[ idx::rho ]       // mass density
u[ idx::ene ]       // total energy density
u[ idx::mom + a ]   // momentum density (a = 0,1,2)
```
This signals to readers that these are internal hydrodynamic indices, not arbitrary array offsets.

## Memory Placement

`const_list_t<T>` (`src/utilities/types/var_list.h`) stores reaction data on GPU with two modes:

- **Constant memory** (`on_const=true`, default): `f_malloc_const` + `f_cc`. Low-latency reads
  but limited to ~64 KB total. Used by default for all reaction sub-lists.
- **Global memory** (`on_const=false`): `f_malloc` + `f_cp`. Unlimited capacity. Toggled via
  `reac_on_const = false` in `[chemistry]` when databases exceed the constant budget.

All reaction sub-lists (mol_cool, atom_cool, phot_reac, dust_reac) share the same policy —
set in `reaction::list::std_t` and propagated to `usr/extension/chem_therm/reaction/reaction_list.cpp`.
The `free()` method on both `const_list_t` and `reaction::list::std_t` cleans up device memory
when `on_const=false`.

## Pole Rebin

The pole rebinning kernel (`src/mesh/geometry/pole_rebin.h`) uses a 2D block grid (n_bl_x × n_bl_y)
with thread count capped by `pole_rebin_threadlim` (parameter in `[mesh]`, default 32). Controls
GPU occupancy during the coarsening step that merges cells near θ=0,π in spherical coordinates.

## Product Deduplication

`base_t::setup_pars()` (`src/modules/chemistry/reaction/reaction.cpp`) merges duplicate product
species entries in `i_chem`/`f_chem` after stoichiometric reduction. Reactant multiplicities
are preserved since rate coefficients depend on them. `n_chem` is set twice: before `reduce_reac()`
for correct reduction, and after product dedup for correct memory allocation.

## Rate Coefficient Conversion (CGS → INI prefactor)

**Reference:** `docs/Chemistry.tm` §1.1

### The full conversion chain

Kratos uses three distinct representations of species abundance:

| Variable | Definition | Units | Used by |
|---|---|---|---|
| `n_i` | Number density (CGS) | cm⁻³ | Physics databases (UMIST, KIDA, VENOT) |
| `ỹ_i` | Reduced number density = `n_i·m_A/ρ₀` | dimensionless | Hydro advection (what the hydro solver transports) |
| `y_i` | Molar abundance = `n_i·m_A/ρ` | dimensionless | Chemistry ODE solver (what `rate()` consumes) |

where `ρ₀` is the code-unit mass density (user-defined CGS value) and `ρ` is the local
physical mass density. The conversion `ỹ_i → y_i` is done by dividing by the
code-unit density `ρ̃ ≡ ρ/ρ₀` (see `src/modules/chemistry/solver/dat.h:154`).

### INI prefactor conversion

The `rate()` method in `src/modules/chemistry/reaction/reaction.h` automatically
applies `ρ^(R−1)` to the rate expression (where R = number of reactants). The INI
`parameters` prefactor `Ã` must already account for the CGS-to-Kratos unit conversion.

**Conversion rule:**

```
Ã = A · m_A^−(R−1) = A · N_A^(R−1)   (CGS units)
```

| R (reactants) | Reaction type        | Ã (INI prefactor) | Example |
|:---:|---|:---:|---|
| 1 | Unimolecular (dissociation) | Ã = A (no change) | CH4 → CH3 + H: A stays in s⁻¹ |
| 2 | Bimolecular                 | Ã = A · N_A       | OH + H → H2O: multiply A by N_A |
| 3 | Termolecular                | Ã = A · N_A²      | H + H + M → H2 + M: multiply A by N_A² |

where `m_A = 1.67262e-24 g` (atomic mass unit, `phys::cgs::mp` in `src/utilities/phys/constants.h:16`)
and `N_A = 1/m_A ≈ 6.022e23` in CGS.

**Critical pitfall:** Unimolecular (R=1) reactions receive **no** N_A factor. If you
multiply a dissociation rate by N_A, the resulting rate is inflated by ~6e23, causing
extreme numerical stiffness and solver step-size collapse. This is the most common
error when porting databases to Kratos format.

**For Troe/pressure-dependent reactions** (implemented in
`usr/extension/chem_therm/reaction/reaction.h`, `kind = troe`): the unified rate
coefficient k(T,n_tot) uses n_tot = ρ/(μ·m_A) internally. The INI entry uses **11**
parameters: `A₀ b₀ C₀ A∞ b∞ C∞ α T₁ T₂ T₃ μ̄`, where A₀ (low-P limit, cm⁶/s equivalent)
and A∞ (high-P limit, s⁻¹) follow the same N_A conversion rule as termolecular and
unimolecular reactions respectively. The rate_type must be set to `TROE`.

## Gotchas

- `debug_macros.h` is **force-included** via `-include` in the Makefile — it's always present even
  if not explicitly `#include`d.
- The `usr_ext/` directory is its **own git repository** (it has its own `.git/`, the parent
  `.gitignore` ignores it, and `git submodule status` reports nothing) — changes there must be
  committed separately, and a `git worktree` of the parent does **not** contain it.
- `float_min` in `src/types.h` is intentionally set to `1e-24f` (or `1e-28f` for PRECISION=2)
  rather than `std::numeric_limits::min()`.
- Default snapshot suffix is `.bin` (not `.dat`), defined in `src/mesh/cycle/cycle.cpp`.
- `device::base_t::sync_all_streams()`'s base part only drains the (never-populated)
  `stream_m`/`event_m` maps, but the derived classes append the real barrier
  (`cuda_t::sync_all_streams` adds `cudaDeviceSynchronize()`, likewise hip/musa) — so the
  virtual call IS a full device barrier. The per-stream `sync_stream(s)` /
  `mod_base_t::sync_streams(mesh)` are the lighter-weight alternatives.
- The `input` class is the only core type without the `_t` suffix — it's `class input`, not `input_t`.
- `shared_from_this()` on `mod_base_t` is **deleted** — use `weak_from_this()` only, to avoid
  ownership cycles.
- The `USRDIR` variable selects which pgen to compile. Only **one** pgen per build — the binary
  is always `bin/kratos`.

## Changelog

Only the two newest entries stay here;  the full changelog (2026-07-13 … 2026-10-18, every Stage A–N record) is archived at `docs/history/agents-changelog-archive.md`.

### 2026-11-06 (later still) — the edge step's host code lives in `driver_t`, and every host body left its header

The 1,387-line `edge_mg.cpp` was 73% free functions (1,014 lines: a 299-line `solve_step`, a
173-line diagnostic, a traffic meter, a bridge and 212 lines of dead code).  They are now members
of one type: `driver_t` (`edge_mg_driver.h/.cpp`, 206 + 951 lines) owns the step state, the
traffic counters, the complex/complex setup, the A7.1 bridge and the nine stage methods;
`solver_t` holds a `driver_t * p_dr` and forwards, so `edge_mg.cpp` is 253 lines of `par_t` and
`solver_t`.  115 further host definitions left 12 headers for a `.cpp` of their own (one header,
one `.cpp`); the `edge_mg` headers went 9,491 -> 5,970 lines.  Dead code went with it:
`edge_mg_vector.h`, `ic0_t`, `dense_chol_t` and seven unreferenced functions.  The rule is now
`STYLE_GUIDE.md` **§6.3** -- device-visible bodies stay in the header (requirement: every TU and
every kernel instantiation needs them), host non-templates go to the module `.cpp`, host
templates stay in the header and are tiered by body length (<=15 lines in class, 16-120 below the
class, >120 in a `*_impl.h`); `AGENTS.md`'s Code style section points at it.  Verification:
`make USRDIR=usr_ext/mhd_ee ARCH=CUDA PRECISION=2` RC=0 with no warnings and
**25 gates, 0 not passing**; M9.5 method 5 is `2.751e-13` with 0 reads / 0 writes and `mf-cyc`
and the MF.6 iteration counts are bit-identical.  The only moving numbers were last digits of
comparison lines, and those move partly **run to run**: the device operator scatters with
`atomicAdd` (A4.31), so a pair of identical runs already differs there (2.634 vs 2.659e-16 on
`mf-agg`); the rest is the TU change (the bodies are now compiled in their own TU).  A gate for
this kind of move must use the numbers that are stable on one binary -- M9.5's 0/0, the iteration
counts, `mf-cyc` -- never the atomic-scattered comparison lines (`mhd/nonideal/STATUS.md` (100)).

The same evening the per-level ladder left the module too.  `driver_t::check_hierarchy` used to
print `mf-agg` / `mf-gal` / `mf-lam` / `mf-cyc` on every hierarchy build, which also cost a
host-device read-back of `lam`; it now measures only the two scalars and throws, prints nothing on
a healthy build and reads no scalar back.  The report is `usr_ext/mhd_ee/ee_audit.cpp`, installed
before the first step through `driver_t::audit( )` (a function pointer, null in production) and
`solver_t::driver( )`; the nine gates that print it are `gate_e2e.cpp` and `gate_mf.cpp`.  The
contraction is passed in, so the V-cycle still runs once (`mhd/nonideal/STATUS.md` (101)).

### 2026-11-06 (later) — H1 finished for the solver's PCG, and the polar axis had a singular coarse block

The per-step scalar recursion is now **on the device**.  `mf_t::pcg` computes `alpha`, `beta`,
`res` and the breakdown flag in one-thread kernels and reads **8 floats** per iteration; the
dot products go through `real_dot_dev` (two-stage reduction, the scalar never leaves the device),
and `omega` -- with `lambda` from a device power iteration -- is a device scalar too
(`STATUS.md` (90) section 2).  Step time: 54 ms (8x16x32) / 94 ms (8x16x128), from 170 / 299 --
and the coarse level's nested iteration limit turned out to be pure rate: `ee_coarse_iter` = 10,
`ee_coarse_tol` = 1e-10 are new `[nonideal]` knobs (100 iterations cost 3x and change the
contraction by nothing).

That change then made the polar axis fail, which means it had been broken since P5: a coarse
cell whose local edges the compaction dropped has an **all-zero row and column** in the aggregated
block (the prolongation has no column for it), so `blk_blk_k`'s Cholesky hit a zero pivot and every
coarsest solve returned NaN.  Nothing saw it: the contraction stayed finite at 4.75e-02, because the
old `pcg` broke on a bad `pAp` *before* multiplying it in, so the coarse correction was just zero --
and the old contraction band was 1.0.  The fix is a unit row for the dropped edges plus a positive-
pivot guard, and the usable lesson is `0 * NaN = NaN`: a breakdown now leaves the state exactly
untouched (`dead-ends.md` M86).  Suite: 24 gates, 0 not passing; a5/8x16x128 numbers bit-identical.

**And the last host-side scalars went with it.**  K6's face solve (`M2 rep = M12^T a`) had its own
`pcg_dev_t` -- the final client of `edge_mg_krylov.h` -- so it now runs the same device scalar
recursion (the four generic ops moved to `edge_mg_dev_vec.h`) and that header is **deleted**: the
whole line has two solves left, `mf_t::pcg` and `rep_dev_solve`, both reading **8 floats per
iteration** and computing nothing on the host (`STATUS.md` (94)).  The bridge also got a **warm
start** (the previous step's rep): 9 iterations -> 1 on a steady step, and the 0-iteration path is
now gated -- `K6.12`/`K6.13` start from the device's and the host's solution and demand **zero**
iterations, because a self-correcting CG hides a wrong initial residual completely: flipping that
residual's sign left G1 at **3.3e-16** and moved only the iteration count (`dead-ends.md` M88).
A new gate `mf_wb1` puts the mode-1 tail in the suite (25 gates, 0 not passing) -- it used to be
exercised by hand only.

**The line moved back into the trunk** (2026-11-06, `STATUS.md` (96)):  after P4 finished and the
host stack was deleted,  `usr_ext/mhd_ee/ni/` became `src/modules/dynamics/mhd/nonideal/`
(29 files,  15,782 lines,  filenames unchanged).  `usr_ext/ppd/ni_link.cpp` and
`usr_ext/ni_ab/ni_link.cpp` -- three-line text includes that existed only because `USRDIR`
compiles nothing but the pgen -- are **gone**;  `find src` picks the module up for every pgen.
Verified:  `mhd_ee` (PREC2) 25 gates green,  `ni_ab` (PREC2),  `ppd` (PREC1),  `pol_rad` and
`sod` (PREC1) all build.  `edge_mg/kernels/` now exists too -- `ni_ie_kernels.h` (the implicit step's eta / grad / M12 /
`M2` / writeback kernels,  10 of them) and `ni_mg_kernels.h` (the matrix-free MG's 22) --
so **`__global__` appears nowhere else in `edge_mg/`**;  the five `__host__ __device__ inline`
geometry helpers that the kernels differentiate with moved with them,  `wb_face_t` moved into
`edge_mg_data.h`,  and the header-defined `ee_div_k` had to become `static` (a non-template
`__global__` in a shared header is emitted per TU and nvlink refuses the duplicate).  The explicit
half of the line (`nonideal.h`,  `implicit.h`) still carries 11 kernels;  that is a separate move.
The rule's boundary is written down in `STATUS.md` (97).

**And the module directory now has two halves, not one plus leftovers.**  The explicit /
super-time-stepping side (`nonideal.h`,  `nonideal.cpp`,  `implicit.h`,  `eta_tensor.h`) moved into
`nonideal/explicit_sts/`,  so `nonideal/` reads as two routes -- `explicit_sts/` and `edge_mg/` --
plus the solver's `edge_mg/kernels/`.  File contents unchanged;  what changed is one more `../` in the
relative includes and six external include sites (`STATUS.md` (98)).  **And the test oracle left the library.**  `edge_mg_ref*` -- the reference side the
gates compare against (`ref_*`,  ~1,750 lines;  `probe_*` until P4.3b) -- was still in
`src/modules/dynamics/mhd/nonideal/edge_mg/`,  so `find src` compiled it into *every* pgen's binary
while no production line calls it (production keeps its own `prod_rhs_m12` /
`prod_rhs_m12_t`).  It now lives in `test/edge_mg_ref/` -- a directory no build auto-discovers --
and the two pgens that need it (`mhd_ee`'s gates,  `ni_ab`'s comparison legs) each compile it
through a two-line TU (`edge_mg_ref_tu.cpp`).  Its thirteen declarations left the production
interface header `edge_mg_iface.h` for `edge_mg_ref.h` (`STATUS.md` (99)):  25 gates green,
`mf-cyc` and `M9.5` bit-identical,  `ppd` builds with no mention of the file.
Those twelve `__global__`
definitions still live inline in their headers;  moving them into an `explicit_sts/kernels/` is a
separate decision,  because their template parameters are the dynamics module's own types
(`bdt_T`,  `itg_T`,  `eta_T`) and `implicit.h` declares before it defines.
