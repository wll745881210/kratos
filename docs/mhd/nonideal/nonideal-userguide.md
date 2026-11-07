# Implicit non-ideal MHD (curl–curl multigrid):  the user's guide

> Audience:  an agent or user wiring the implicit non-ideal MHD module into a
> larger application ( thermochemistry solvers,  protoplanetary-disk
> accretion / wind launching,  … ),  or running it standalone.   This is the
> *operational* companion to `docs/mhd/nonideal/implicit-nonideal-mg-method.md` ( the
> algorithm / derivation document )  —  here we answer "which knob,  which
> file,  which command",  not "why".   State as of 2026-10-11;  the
> chronicle for how it got here is
> `~/scratch/non_ideal_tst/status-smr-curvilinear-2026-10-05.md`.

---

## 1.  What the module is — and is not

**Is**:  an implicit,  unconditionally-stable magnetic-diffusion solve of
`( I + dt · R · curl curl ) B* = B^n` on the face-centred CT field,  for
Ohmic + ambipolar diffusivity ( scalar or anisotropic,  uniform or
spatially varying ),  on Cartesian / cylindrical / spherical grids,  on
single-level and 2:1-SMR meshes,  with selectable physical-boundary edge
BCs.  The CT-adjoint defect-correction keeps the discrete divergence
EXACTLY preserved at the fixed point ( `div B* = div B^n` to roundoff ).
Time schemes:  backward Euler / Crank–Nicolson / BDF2.

**Is not** ( read this list before planning science runs ):

1. **No implicit Hall.**  The Hall term is interface-present in the
   *explicit* chain only ( `nonideal.h`'s `eta_h` / `hall_l` );  the
   implicit operator carries no `eta_h` at all — the provider's Hall
   output is dropped at the Picard freeze.  For protoplanetary-disk wind
   launching this is the **primary physics gap**:  Hall is dispersive,  not
   parabolic — it cannot ride this curl–curl preconditioner as-is and
   needs its own treatment ( explicit sub-cycling or a dedicated solver ).
   Plan for Hall explicitly before promising disk-wind results.
2. **Energy feedback on the implicit path:  implemented**  ( 2026-10-11 ).
   The heating is NOT a source term:  it is the magnetic-energy sink
   delivered as the diffusion-Poynting-flux divergence
   `u[ene] += -dt div( E* x Bbar )`  with the same edge E\* as the field
   update and  Bbar  the  ( B^n + B* ) / 2  face average.
   ⚠ 2026-10-24:  the switch stays 1 by default -- the deposit is
   REQUIRED ( § 3.5 gives the algebra:  without it the gas absorbs the
   transport term `dt div S` as spurious heat ).   A numerical caveat
   does apply:  the stencil's `surface/geo.volume` is ill-conditioned at
   the polar axis,  where `p` was measured reaching 1e34 / inf.

3. **MPI is untested** for the face-aware comm and the new fills ( all
   development batteries are single-rank ).  Single-node GPU only.
4. Not a moving-mesh / unstructured solver;  block-structured 2:1 SMR only.

Two implicit solvers exist and are mutually exclusive per run ( a
run-time guard throws ):

* **Solver 1** `mg_diffusion_t`  ( `[nonideal] mg_diffusion` ):  cell-centred
  vector-Laplacian BE multigrid.  Older production path;  div B preserved
  only to the Laplacian's discretization error.  Use only if you
  specifically need it;  not covered further here.
* **Solver 2** `mg_cc_diffusion_t`  ( `[nonideal] mg_cc` ):  the face-centred
  curl–curl solver — **the recommended path**,  the subject of this guide.

**The one-solver rule is only partially guarded.**  The guard covers
`mg_diffusion` ↔ `mg_cc` ( and the var-mode `eta_*` knobs )  — but
`mg_cc = 1` combined with a live EXPLICIT diffusion chain
( `[nonideal] eta_o / eta_a / hall_l ≠ 0`,  or `mode = 1` / `sts` )
is NOT guarded outside var mode and would apply the diffusion TWICE.
When running mg_cc, keep the explicit chain inert:  `mode = 0` and
the `eta_*` knobs zero ( § 3.3 ).

---

## 2.  Ten-minute quickstart

Everything below assumes the reference pgen `usr_ext/std_tst/mhd_ni_tst`
( the validation harness AND the template for your own pgen ).

```sh
# 1. Build ( CUDA release;  USRDIR is mandatory;  always make clean when
#    switching ARCH / DEBUG / USRDIR ):
cd ~/apps/kratos_mhd_cuda
make clean && make USRDIR=usr_ext/std_tst/mhd_ni_tst \
     ARCH=CUDA DEBUG=0 MPI=0 -j8

# 2. GPU discipline:  GPU0 is occupied by other programs — always:
export CUDA_VISIBLE_DEVICES=1

# 3. Run the analytic benchmark ( T1:  cylindrical annulus eigenmode,
#    essential BC,  one cycle,  exact Bessel answer ):
cd ~/scratch/non_ideal_tst/runs/mhd_ni/ni_d4/run_bc
mkdir -p /tmp/t1run && cp t1_16.par /tmp/t1run/ && cd /tmp/t1run
~/apps/kratos_mhd_cuda/bin/kratos t1_16.par
#    Expect:  "MG with N iters, defect = X" converging below ~1e-6,
#    exit 0,  a t1_16_00001.bin written.

# 4. Probe convergence speed per V-cycle ( the diagnostic switch ):
#    add  print_iter = 1  under [multigrid] in the par file,  rerun.

# 5. Run the full regression gate ( 26 tests,  ~minutes,  single rank ):
cd ~/scratch/non_ideal_tst/runs/mhd_ni
CUDA_VISIBLE_DEVICES=1 bash batt_cleanup.sh check   # default REF = main_v14
```

Test-writing conventions ( enforced by the battery;  follow them in your
own tests too ):

* **One driving cycle per test**:  `n_cycle_lim = 1`  with
  `final_output = 1` in `[cycle]` — the cycle-1 dump `<prefix>_00001.bin`
  is the post-step state.  ( The runner then executes a degenerate second
  cycle with `dt → 0`;  harmless. )  Long runs are NOT the debug tool —
  convergence speed is judged from the per-V-cycle / per-pass defect
  sequences,  not from end states.
* `except_ferr = 1` under `[cycle]` in any diagnostic run ( traps FP /
  device faults at the source ).
* `[multigrid] print_info = 1` prints the per-solve defect summary;
  `print_iter = 1` prints the defect after EVERY V-cycle.
* A "1 iteration / defect-0" print = unseeded solve;  a 300-cap print
  above tolerance = stalled solve.  The attainable defect floor is
  problem-dependent — a small final defect is NOT by itself a failure
  criterion;  the *decay sequence* is.

Reference .par files to copy from ( all under
`~/scratch/non_ideal_tst/runs/mhd_ni/` ):

| Need | File |
|---|---|
| Cartesian uniform-eta eigenmode, no SMR | `ni_cond/g_ok.par` |
| Cyl. analytic decay, essential BC ( T1 ) | `ni_d4/run_bc/t1_16.par` |
| Cyl. variable ambipolar + SMR + essential BC | `vareta/c_amb_smr2_bc1.par` |
| Spherical variable-eta smoke | `vareta/s_uni_var.par` |
| Spherical essential-BC eigenmode | `vareta/sph_t25_16.par` |
| Boundary-BC A/B/C comparison battery | `ni_d4/run_bc/bc{0,1,2}*.par` |

---

## 3.  Integration wiring ( for the thermochemistry / application agent )

This section is the checklist for embedding the implicit solver in a new
pgen with your own diffusivity provider.  The reference implementation of
every item is `usr_ext/std_tst/mhd_ni_tst/usr.cpp`;  line numbers below
point there ( they drift — grep the symbols ).

### 3.1  Module enrollment

Enrollment block ( `usr.cpp:1441` `run()` — the reference pgen;  the
mg_cc minimal set is 3 modules + device + meshgen ):

```cpp
mesh.enroll_device    < device_t > (  );
mesh.enroll_mgen      < mesh::meshgen::cyl_t > (  ); // base / cyl_t / sph_t
mesh.enroll_module    < mhd_ni_t< itg_t > > (  );   // the MHD module
mesh.enroll_module    < mg_cc_t > (  );             // solver 2  <-- this guide
// ( mg_ni_t — solver 1 — is a fourth enroll in the reference pgen;
//   drop it if you only use mg_cc )
```

Enrollment order matters:  the MG module's `step()` runs AFTER the ideal
MHD step each cycle ( the mesh walks modules in enrollment order ),  and
reads `dt` from the cycle ( `mesh.p_cyc->dt()` ) — the same dt the EOS
step produced.  The ideal integrator carries no diffusion hook ( Stage
W3 ):  split-sweep treatments for the *explicit* chain are invoked from
the pgen's `mhd_ni_t::step` override ( `usr.cpp:292` ) via
`ni_t::diffuse`;  the **mg_cc path needs no such call** — its module
step does everything.  With `mode = 0` ( default ) `diffuse` is a no-op.

`itg_t` picks the geometry:  `rk2_t<>` ( Cartesian ),
`integrator::cyl_t<>`, `integrator::sph_t<>` — paired with the matching
meshgen.

### 3.2  The proxy scaffolding ( copy verbatim,  rename )

`mg_cc_diffusion_t` does not own block data;  it views the MHD module's
data through a proxy you define in the pgen.  Copy
`prx_base_t` + `prx_cc_t` ( `usr.cpp:34`–`167` ):  weak-ptr module
handles,  `d_mhd` / `d_mg_cc` accessors,  and `bdt_cc_t` — the
block-data view that shallow-copies the MHD face field plus the MG-side
scratch ( `mgd`,  `r_arr`,  `e_arr`,  `src0`,  `b_prev`,  `b_iter`,
`ifc_ax`,  `xfc_ax` ).  The second proxy in the file,  `prx_nimg_t` /
`prx_mgni_t`,  exists for solver 1 — drop it if you only enroll mg_cc.

The mg_cc type is then:

```cpp
using mg_cc_t = mg_cc_diffusion_t< prx_cc_t, my_eta_t >;
```

### 3.3  The diffusivity provider ( the chemistry plug-in point )

`nonideal.h` defines the provider contract;  the placeholder
`chem_eta_t` ( `nonideal.h:195` )  is the reserved wiring point for the
chemistry module — it currently throws on `read`.  Implement your own
struct following:

```cpp
struct my_eta_t
{
    //  Plain-data members only:  the provider is copied BY VALUE into
    //  the kernels — it must stay trivially copyable.  No device
    //  pointers,  no std:: containers.
    type::float_t  eta_o, eta_a;      //  whatever pars you need

    __host__ void read( const input & args )
    {
        eta_o = args.get< type::float_t >( "mysec", "eta_o", 0 );
        //  ...  set  on = ( any diffusivity > 0 );
    };

    //  The device evaluation:  eo / eh / ea = Ohmic / Hall / ambipolar
    //  diffusivity at CELL  i  ( the local composition / ionisation
    //  state is read from  b_d.hyd.w  if needed ).   eh is DROPPED by
    //  the implicit operator ( no implicit Hall ) — return 0 there or
    //  feed the explicit chain only.
    template< class bdt_T, class idx_T, class f2_T >
    __device__ __forceinline__ void eval
    ( const bdt_T & b_d, const idx_T & i,
      f2_T & eo, f2_T & eh, f2_T & ea )                 const
    {
        //  e.g.  const auto * w = b_d.hyd.w.at( i );
        //        evaluate from  w[ idx::rho ],  species,  temperature ...
        eo = ...;   eh = 0;   ea = ...;
        return;
    };
};
```

Wiring rules ( enforced by run-time guards in
`mg_cc_diffusion_t::read`,  `implicit_cc.h:1686` )  —  note the
**two-provider pattern** the reference pgen uses:

* The mg_cc module carries YOUR provider ( template arg 2 ):
  `mg_cc_diffusion_t< prx_cc_t, my_eta_t >`,  enabled with
  `[nonideal] mg_cc_eta_var = 1` + the provider's own pars.
  The uniform knobs ( `mg_cc_eta`,  `mg_cc_eta_o/_a`,  `_z`
  profiles )  must then be ZERO — the guard throws on any mixture.
* The MHD module's integrator + EOS keep an INERT provider
  ( `hall_len_eta_t` / `const_eta_t` with all `eta_*` knobs zero ):
  the explicit coupled chain then adds no diffusion EMF and the
  `ni_eos_t` cap is relaxed by the `mg_cc` flag itself.  Do NOT give
  the MHD module a live provider while mg_cc runs — outside var mode
  nothing throws and the diffusion would be applied twice ( the
  explicit EMF + the implicit solve ).
* `[nonideal] prof_eta = 1` ( which swaps the MHD module's provider to
  `prof_eta_t` )  exists ONLY for explicit / STS *reference* runs —
  mg_cc off,  the explicit chain seeing the same variable η.
* `mg_cc_exact = 1` ( the FEEC single-operator path )  with
  `eta_var` is NOT audited and throws — keep the defect-correction path.
* The anisotropy DIRECTION ( b̂ )  is handled internally:  frozen from
  the field at the freeze ( `mg_cc_bhat` / the anisotropic paths ).  The
  provider returns scalars only.

Semantics of the freeze ( Picard ):  once per cycle,  inside the MG
module's step,  `freeze_r_arr` ( `implicit_cc.h:1962` )  refreshes the
primitive state ( `w` — your provider reads fresh ρ / composition
there ),  evaluates the provider per cell,  writes the frozen R
coefficients into `r_arr` / `e_arr`,  and restricts them to the coarse
levels.  The coefficients are then FROZEN for the whole solve —
nonlinearity is resolved at first-order Picard accuracy per step ( fine
for `dt`-small evolution;  revisit if your chemistry makes η vary
strongly within a step ).

A one-shot host-side SVD audit of the frozen R ( positive-
semidefiniteness )  runs on the first var-mode step and throws if your
provider produces negative eigenvalues — surface densities / ionisation
fractions can do that;  clamp in the provider ( e.g.  floor η_O + η_A
perpendicular at a tiny positive value ).

Reference provider if you want a cheap variable-η before chemistry is
ready:  `prof_eta_t` ( `nonideal.h:151` ) — density power law
`η = η_ref ( ρ / ρ_ref )^α` via the `prof_*` pars.  The reference pgen
bakes it into its `mg_cc_t` instantiation;  enable with
`[nonideal] mg_cc_eta_var = 1` ( the `prof_eta` switch stays 0 — that
one is the MHD-module side,  for explicit reference runs ).

### 3.4  The dt handshake ( EOS )

`ni_eos_t` ( `nonideal.h:1296` )  wraps the adiabatic EOS and adds the
parabolic dt cap — which is RELAXED when `mg_cc` is on ( the implicit
solve is unconditionally stable;  only the Hall cap would remain,  and
there is none on the implicit path ).  Enroll it ( it is part of the
module's internal `enroll` in § 3.3 ),  and the EOS-derived dt then
flows:  cycle dt → MHD ideal step → mg_cc module step reads
`mesh.p_cyc->dt()`.  `dt` still comes from the EOS step — a full skip of
the MHD module would freeze `t` / leave `dt` dangling ( that is why the
isolation switch keeps `cons2prim` ).

The `[prob] mhd_off = 1` isolation switch ( this pgen only,
`usr.cpp:282`–`345` )  runs `cons2prim` + `act_bnd` but skips the ideal
flux / update — the field evolves ONLY through the implicit solver.
Use it to bisect "is a problem in the ideal step or the implicit solve".

### 3.5  Energy feedback / heating ( MUST implement for chemistry )

**The energy-budget principle ( verified against Athena++'s design and our
explicit chain ):  the non-ideal heating is NOT a source term — it is the
magnetic-energy sink, delivered in flux form.**  The pair

```
induction:   B*  = B^n  -  dt C E*            ( the CT update )
energy:      u[ene]  +=  - dt div( E* x B )    ( the DIFFUSION Poynting flux )
```

with the SAME edge field  `E*`  conserves the total energy exactly ( both
updates telescope the same faces ).  CONFIRMED 2026-10-24 against the
local Athena++ tree:  its conserved `IEN` carries the magnetic energy and
`FieldDiffusion::AddPoyntingFlux` adds the diffusive Poynting flux to the
`IEN` flux ( `src/field/field_diffusion/energy.cpp:108-128` ) -- exactly
this pair.  The algebra:  a diffusive step changes the field energy by
transport AND dissipation,  `dm = -dt div S - dt eta J^2`;  leaving
`u[ene]` alone hands the gas `dt div S + dt eta J^2` ( the transport as
spurious heat ),  whereas the deposit leaves it exactly `dt eta J^2` and
moves the total only by the boundary outflow.  Locally the thermal gain is
`+ dt E*.J = dt ( eta_O J^2 + eta_A |J_perp|^2 )` — the magnetic energy
loss minus the boundary Poynting outflow — emerging as the DIFFERENCE of
the two flux divergences, never as an added term.  Adding a separate
`+ dt eta J^2` source on top of the flux ( or instead of it ) is either a
double count or a non-conservative discretization.  This mirrors viscous
hydrodynamic heating exactly ( kinetic sink = thermal gain via the stress
flux ), and it is exactly how Athena++ does it
( `FieldDiffusion::PoyntingFlux` + `AddPoyntingFlux`,
`src/field/field_diffusion/energy.cpp` — the flux  `S = E_oa x B`  is added
to the energy flux; there is no heating source anywhere ).

What exists today:

* **Explicit chain — complete.**  `ni_ene_k`  ( `nonideal.h:721` )  already
  implements the recipe:  the face-centred  `S_a = E_dif x B`  from the edge
  emfs ( averaged onto faces )  and the cell-centred B,  applied as
  `u[ene] += -dt div(S)`  ( the  `s==0 ? +s_a : -s_a`  face loop ),  plus the
  `heat[ 0 ] / heat[ 1 ]`  DIAGNOSTIC accumulation
  ( `dt eta_O J^2`  /  `dt eta_A |J_perp|^2`  —  analysis only,  never added
  to the energy ).  The identity "Ohmic heating = magnetic-energy loss" is
  the test_mode-1 validation.
* **Implicit path — IMPLEMENTED**  ( 2026-10-11,  `[nonideal] mg_cc_ene`,
  default ON ).   After the solve converges  ( bf = B^n,  mgd.fld = B* ):
  the fixed - point edge emf  E*  is evaluated per edge  ( the exact row's
  own  `eta_edge_full` / `cur_edge_full`  machinery in var mode;  the
  uniform  eo J + ea (J.z)  otherwise;  sign - flipped to the physical E —
  the MG row's  `+dt curl`  convention is the negative of the CT  `ee` ),
  staged into  `e_dif`,  completed at the boundary / interface rows by the
  MHD keeper's  `act_flx`  edge exchange  ( through  `ee`,  swapped back ),
  and applied as  `u[ene] += -dt_eff div( E* x Bbar )`  with  Bbar  the
  ( B^n + B* ) / 2  face average.   CN  ( `mg_cc_time = 1` )  uses the
  trapezoid  E = ( E^n + E^{ n+1 } ) / 2  ( the  E^n  end from a  bf
  snapshot into  b_iter ).   BDF2 keeps the BE - pair form  ( th = 1 ).

**Validation**  ( the annulus ring,  one cycle,  16x32x16 cyl,  bc = 1 ):

| configuration | dMagE | dEtot | capture |
|---|---|---|---|
| uniform η, BE | -1.7565e-3 | +1.6906e-3 | 0.9625 |
| uniform η, CN | -1.8253e-3 | +1.7573e-3 | 0.9627 |
| var η, CN | -1.8252e-3 | +1.7573e-3 | 0.9628 |
| ene off ( A/B ) | -1.7565e-3 | 0 | 0 |

The 3.7 % residual is the boundary Poynting outflow  ( the same stencil
family as the explicit chain's  test_mode-1 identity ).   The  off  leg
leaves  B*  BIT-IDENTICAL  ( the pass never reads  u  back ).

The efficiency-study switch  ( Athena++ has NO built-in one — verified in
the public tree and the local copy:  `PoyntingFlux` is unconditional when
the EOS is non-barotropic and the coefficients are nonzero;  groups patch
it out or subtract a compensating term of identical size —  the two are
mathematically identical ):  `[nonideal] mg_cc_ene`  —  1 = on,  the
default for production  ( energy - conserving );  0 = off —  the dissipated
magnetic energy LEAVES the system,  exactly the with/without-heating A/B
experiment for disk-wind efficiency studies.

Gate note:  the feedback changes  u[ ene ]  →  the EOS  dt  →  all
downstream defect sequences drift at the  1e-7  trajectory level  ( the
§ 7.65 lesson );  the baseline was re-frozen as  **main_v15**  ( PASS;
the analytic  t1 / st25 / divB / dBn  metrics stayed within tolerance ).

Until you enable  `mg_cc_ene`  understanding this,  note that OLD runs
( pre-2026-10-11 )  had the magnetic-energy sink lost from the total —
acceptable for pure-B-diffusion tests  ( the whole prior validation
battery ),  NOT for a heated disk wind.

---

## 4.  Geometry,  boundary,  and SMR support matrix

Verified state ( gate `main_v14`,  2026-10-10/11;  details in the
chronicle § 7.58–7.65 ):

| Configuration | uniform η | var η_O | var η_A ( aniso ) | Notes |
|---|---|---|---|---|
| Cartesian,  single-block | ✓ | ✓ | ✓ | exact divB |
| Cartesian,  multi-block / SMR | ✓ | ✓ | ✓ | gate g_smr / t3_smr |
| Cylindrical,  single-block | ✓ | ✓ | ✓ | corner-free base + curvature;  zebra line smoother |
| Cyl.,  multi-block same-level | ✓ | ✓ | ✓ | exact ≤ 2 ifc axes ( V7/V9 );  3-axis blocks exact via mode-2 vertex comm ( V11 ) |
| Cyl.,  SMR ( cross-level ) | ✓ | ✓ | ✓ | field:  Stage-K exact transfer;  coefficients:  A3 xfill;  residual:  fre×pair corner columns 2–3e-3 ( B2,  was 0.2 ) |
| Spherical,  single-block | ✓ | ✓ | ✓ | clean to 1e-6 incl. essential BC eigenmode |
| Spherical,  SMR | ✓ | ✓ | ✓ | per-block divB drift 1e-6 .. 3.4e-6 ( clean ) |

Ambipolar here means the anisotropic `η_A` with b̂ frozen from the
field ( Stage V4/V10 ).  The linear z-profiles ( `mg_cc_eta_o_z` /
`_a_z` )  are **Cartesian-only** ( guard rail throws on curvilinear );
use the variable provider there.

Practical SMR guidance ( from § 7.59 ):  on cylindrical grids the
residual drift concentrates at corner / seam columns of blocks whose
three axes are all interfaces ( the V4-wrap fallback ).  The essential
BC ( `mg_cc_bc = 1` )  improves interior + simple shell by ~100x over
fre;  letting fine blocks span the full φ period ( self-wrap )  keeps
them ≤ 2 interface axes and on the exact path.

Physical boundary conditions:

* The hydrodynamic boundary is `[boundary] kinds` as usual (
  `fre / out / per / ...`,  six entries = lo/hi per axis ).
* The DIFFUSION boundary is a separate knob,  `[nonideal] mg_cc_bc`:
  **default -1 = auto** ( Cartesian -> 0, curvilinear -> 1 );  `0` fre ( legacy one-sided ),  `1` essential
  ( boundary-plane E_t = 0 — the "edge-current-zero" wall ),  `2`
  pseudo-vacuum.  bc ≠ 0 requires `mg_cc_fmatch = 2` + `mg_cc_ct = 1`
  ( the defaults )  — a run-time guard enforces the combination.
* The explicit chain mirrors it via `[nonideal] bc_edge` ( same three
  values )  when you run STS / explicit references.
* For disk applications:  `mg_cc_bc = 1` is the recommended consistent
  wall;  it is what the T1 / st25 analytic benchmarks validate.
  **EXCEPT at a FREE / OUTFLOW wall** (`kinds` = `fre` or `out`):  there
  `bc = 1` freezes the boundary-normal face while the ideal CT moves it,
  so every implicit step injects `A dB / V` of divergence (~1.2e-2 per
  cycle on the sp32 test).  At such walls use `mg_cc_bc = 0` **with
  `[mesh] pole_rebin_bf = 0`** (the face-field polar rebin destabilizes
  `bc = 0` at r_out;  the state-only rebin is fine) — measured 20x less
  leakage and `max|div B|` back at the ideal level.  See
  `~/scratch/ppd_tst/handover/implicit-bc-natural-option2-2026-10-18.md`.
  **UNDER RE-VERIFICATION ( 2026-10-21 )**:  the 1.2e-2/cycle figure was
  measured BEFORE the thin-axis fix ( `ct_sfc_thin`,  the 2.5D identity-
  operator bug ) and before the `out`-wall advective release landed.  In
  every configuration re-measured since,  `bc = 1` now beats `bc = 0` by
  1–2 orders  —  spherical 2.5D and 3D `sph21` ( 8.78e-3 -> 5.39e-5 ),
  the stretched PPD-like 2.5D `pZ9b` ( 1.66e3 -> 9.29e2 ),  and the
  same-level SMR gates `sl3` / `sb1` ( `ddiv_sh` 0.051 -> 2.6e-4 ).
  The new default is therefore `-1 auto`  ( curvilinear -> 1 ).  The
  sp32 long-run comparison has NOT been redone yet,  so treat this
  paragraph's fre/out caveat as unconfirmed until it is.
  ( Also new:  `mg_cc_bc = 1` + `mg_cc_bclose = 0` core-dumps with a
  floating-point exception on `sph21_25d` —  do not pair them. )

---

## 5.  Parameter reference

> **完整清单（单一来源）**：[`nonideal-knobs.md`](nonideal-knobs.md) —— 67 个键的全表，含
> 默认值、在 223 个 par 里被显式设置的次数、对应成员名、以及在非注释代码里的使用计数
> （据此判定：**0 个死键**）。**本节只描述推荐集**；未列出的键属"内部 / 实验"，
> 其非默认分支多数**从未被执行验证**（表中 `set/223 = 0`）。

Mirrors the method paper appendix;  defaults in parentheses.  Sections
matter — a key appended in the wrong section of the .par is silently
ignored ( the parser binds `section|key` ).

### `[nonideal]` ( solver 2 and the module )

| key | default | meaning |
|---|---|---|
| `mg_cc` | 0 | enable solver 2 ( exclusive with `mg_diffusion`,  `mode`,  `sts` ) |
| `mg_cc_eta` | 0 | uniform total diffusivity |
| `mg_cc_eta_o` / `mg_cc_eta_a` | = `mg_cc_eta` / 0 | uniform Ohmic / ambipolar |
| `mg_cc_eta_o_z` / `mg_cc_eta_a_z` | 0 | linear z-profiles ( Cartesian only,  SPD-guarded ) |
| `mg_cc_bhat` | 0 | freeze b̂ from the field ( the anisotropic direction ) |
| `mg_cc_eta_var` | 0 | the provider path — all `mg_cc_eta*` must be 0 then |
| `prof_eta` | 0 | pgen switch:  use `prof_eta_t` as provider ( STS/explicit reference sees the same η ) |
| `prof_eta_o` / `prof_eta_a` | 0 | provider amplitudes |
| `prof_alpha_o` / `prof_alpha_a` | 0 | density power-law indices |
| `prof_rho_ref` | 1 | density reference |
| `mg_cc_lvl` | −1 | coarsening limit ( −1 = full;  0 = single level / pure relaxation — the transfer-isolation switch ) |
| `mg_cc_ct` | 1 | the CT-adjoint remainder ( exact divB ) — keep on |
| `mg_cc_fmatch` | 2 | slave-face rule ( 1 was removed and throws ) |
| `mg_cc_bc` | **−1 auto** | diffusion BC:  −1 auto ( Cartesian 0, curvilinear 1 ) / 0 fre / 1 essential / 2 pseudo-vacuum.  The auto rule is resolved in `guard_rail_once`;  set 0 explicitly to force the natural closure.  MEASURED ( 2026 - 10 - 22,  PPD n_phi = 1,  stag = 0 ):  bc = 0 keeps max | D.B | about 48x lower than bc = 1 at t = 2e-3,  but only 1.6x lower by t = 1e-2 -- the divergence accumulates under both closures ( dead - ends D30 ) |
| `mg_cc_rem` / `mg_cc_rem_eps` | 2 / 1e-6 | remainder passes / source roundoff filter |
| `mg_cc_npass` | 0 | outer-pass cap ( 0 = floor **96** on the CT path ( `implicit_cc.h` ) ) |
| `mg_cc_rtol` / `mg_cc_rstag` | **1e-6 / 0.99** | Richardson convergence ( geometric-tail estimate )  / stagnation exits |
| `mg_cc_romega` | 1 | Richardson residual damping ( adaptively halved + revert on stall ) |
| `mg_cc_time` | 0 | 0 BE / 1 CN / 2 BDF2 ( see below ) |
| `mg_cc_exact` | 0 | FEEC single-operator path ( experimental;  + var throws ) |
| `mg_cc_fpoint` | ( auto ) | 1 = point GS everywhere;  CN keeps the hybrid zebra default |
| `mg_cc_line` | 0 | zebra line axis ( needs `mg_cc_fpoint = 0` ):  0 radial ( coarse levels only ) / 1 theta ( all levels ) / 2 alternating ( all levels;  the sharp-interface recommendation ) |
| `mg_cc_rharm` | 0 | 1 = harmonic diagonal R restriction across sharp jumps ( no gain measured on the 1e8-contrast testbed ) |
| `eta_dump_file` | ( empty ) | write the FROZEN coefficient field once (`r_arr` + `e_arr`, padded, `KRETA001` header) — the replay source. Single block only |
| `eta_replay_file` | ( empty ) | **replay** that field instead of evaluating the provider each step. With `[prob] chem_on = 0` + `rad_on = 0` the run becomes **bit-reproducible** (the chemistry's atomics, trap M8, are out of the loop) — the only deterministic testbed that still has the PPD geometry, the PPD field and a realistic η |
| `eta_cap_dt` | **0 ( off )** | the provider's TOTAL-η parabolic cap:  the final `eo / ea` ( wedge blend included ) are limited to `0.25·dx_min² / eta_cap_dt`.  Set it to the run's `dt` so the per-cell parabolic number stays ≤ 0.25.  MEASURED ( 2026-10-22,  PPD `n_φ = 1`,  `[wedge]` on ) — without it the inner V-cycle diverges by ×6.4/iteration on one step ( `defect` 5.8e-4 → 1.5e-1 ), the rollback then fakes convergence and a `max\|D.B\| = 3e4` field is written back;  with it the solve injects a constant ~2.5 and the retained divergence reaches t = 1e-2 at **1.44 instead of 6.6e4**.  ⚠️ it weakens the PHYSICAL η on fine cells — a numerical stabilisation knob, not a physics improvement;  `STATUS.md` §3bis |
| `mg_cc_cgrad` | 1 | 1 = the base operator's ∇η first-order cross term ( `couple_grad`;  fixes the Richardson contraction at a sharp η edge with strong current;  0 = legacy base ) |
| `mg_cc_aniso_base` | 1 | 0 = scalar preconditioner with anisotropic exact row ( diagnostic hatch ) |
| `eta_o` / `eta_h` / `eta_a` / `hall_l` | 0 | the EXPLICIT chain's provider knobs — keep ALL zero when mg_cc runs ( an inert MHD-module provider;  nonzero + mg_cc = double diffusion outside var mode ) |
| `mode` | 0 | 0 coupled ( explicit ),  1 split sweep,  2 GS-Laplacian — with mg_cc keep 0 ( `diffuse` is then a no-op ) |
| `sts` / `sts_nu` | 0 / 0.01 | super-time-stepping ( explicit split only;  do not combine with mg_cc ) |
| `bc_edge` | 0 | the explicit chain's edge BC ( 0/1/2 as `mg_cc_bc` ) |
| `mg_cc_ene` | 1 | implicit-path Poynting-energy feedback:  1 = on ( flux form,  energy-conserving — the production default ),  0 = off ( the dissipated magnetic energy leaves the system;  the efficiency-study switch ) |

### `[multigrid]` ( shared substrate )

`n_iter` ( V-cycle cap,  300 ),  `n_smooth` ( 4 — sweeps per level ),
`dtol` ( 1e-6 ),  `iter_first` ( 50 ),  `print_info` ( 0 ),
`print_iter` ( 0 — the per-V-cycle defect probe ),  `stag` ( 1.5 —
stop when defect ≥ stag × previous ),  `iter_guard` ( 8 — the
monotonicity guard:  stop the V-cycle loop and roll the iterate back
to the pre-solve state once the defect exceeds  `iter_guard` × its
best value in the same solve;  <= 0 disables.  It catches a V-cycle
whose iteration matrix carries an amplifying mode — the defect then
RISES from one cycle to the next and multiplies the error until the
state is destroyed.  It only sees the *inner* loop,  so it cannot
help when the driver runs one V-cycle per pass ),  `phy_bnd_type`
( ignored by solver 2 — the fixed fold is forced on physical-boundary
grids ).

### `[cycle]`,  `[prob]` ( harness )

`n_cycle_lim = 1`,  `final_output = 1`,  `except_ferr = 1` ( test
conventions ).  In `mhd_ni_tst` additionally:  `test_mode` ( the seed
family;  22 = T1 annulus eigenmode,  24 = power-law ρ ring,  25 =
spherical eigenmode ),  `geometry` ( 0/1/2 ),  `mhd_off`,  `bamp`,
`eps`, `k_wave`, `ew_lam` / `ew_c` ( T1 eigenmode pars ),  `rho_pow`.

Time scheme choice ( 2026-10-07 convention ):  **CN ( `mg_cc_time = 1` )
is the reference** for new benchmarks / validation — second order at the
same per-cycle cost.  BE stays the regression-gate default ( the frozen
baselines are BE ).  BDF2 is the fallback for rough / stiff data ( CN
is not L-stable:  expect a ringing defect floor ~1e-4 at η ≥ 100 ).

---

## 6.  Validation status & the regression gate

* **Gate**:  `~/scratch/non_ideal_tst/runs/mhd_ni/batt_cleanup.sh` —
  26 tests / 932 metric lines,  single rank,  CUDA release build,
  minutes of runtime.  Current baseline `baselines/main_v14.txt`
  ( 2026-10-10,  recorded on the current binary — the gate is
  bit-sensitive:  re-`record` after any rebuild,  compare only
  same-binary runs ).
  ```sh
  cd ~/scratch/non_ideal_tst/runs/mhd_ni
  CUDA_VISIBLE_DEVICES=1 bash batt_cleanup.sh check           # vs main_v14 (default REF)
  CUDA_VISIBLE_DEVICES=1 bash batt_cleanup.sh record main_v15 # re-freeze
  ```
* **Analytic references**:  T1 ( cyl annulus Bessel eigenmode,  one-step
  decay vs the BE factor within 0.25 % at 32 cells,  gate t1_16/t1_32 );
  st25 ( spherical eigenmode,  −0.25 % @16 / −0.13 % @32 ).
* **divB discriminators**:  tm20 ( cyl )  / tm21 ( sph )  seeds — the
  CT-adjoint fixed point preserves divergence to roundoff on all
  geometry × SMR combinations ( the matrix of § 4 ).
* **Known residual floors** ( not failures ):  CN ringing ~1e-4 at
  η ≥ 100;  cyl SMR corner/seam columns 2–3e-3 with fre BCs ( B2 );
  spherical SMR clean at 1e-6.
* **Performance**:  see the method paper ( § performance and the
  implicit-vs-STS / explicit sweeps at η = 1..1000,  release build,
  GPU1-exclusive ).  Do not re-measure on GPU0 or with DEBUG=1.

---

## 7.  Debug workflow

NaN / illegal-memory / abort triage ( the documented route ):

1. Set `except_ferr = 1` in `[cycle]` of the failing .par.
2. Rebuild the SAME pgen on the debug farm ( separate checkout so the
   release binary survives ):
   ```sh
   cd ~/apps/kratos_mhd_cuda_dbg
   make clean && make USRDIR=usr_ext/std_tst/mhd_ni_tst \
        ARCH=HIPCPU DEBUG=1 MPI=0 -j8
   ```
   ( always `make clean` on an ARCH switch ).
3. Run under GDB with the two traps set BEFORE running:
   ```
   gdb --args bin/kratos bad.par
   (gdb) catch throw
   (gdb) break abort
   (gdb) run
   ```
   `catch throw` catches the input-validation `std::runtime_error`s at
   the throwing frame;  `break abort` catches device-side faults.
   The DEBUG build's bounds check prints `3d` / `i: [...]` on
   out-of-range array accesses — any nonzero count is a ghost-slot bug.
4. After diagnosis,  `make clean && make ARCH=CUDA DEBUG=0 USRDIR=...`
   for any longer / higher-resolution run.  DEBUG is for
   troubleshooting only;  GPU-side DEBUG=1 is NOT slower ( no `-G` ),
   but it changes the binary — timing and bit-comparisons need the
   release build regardless.

Environment specifics that bite:

* **GPU**:  `export CUDA_VISIBLE_DEVICES=1` for every run / gate /
  timing ( GPU0 hosts other programs;  timing additionally needs GPU1
  exclusive ).
* **Cross-line build breakage**:  a second work line ( chemistry )
  syncs WIP headers into this tree via Seafile;  a half-synced header
  can break the CUDA build ( e.g. a device-side `throw`,  which nvcc
  rejects ).  Do NOT edit their files:  touch the stale objects and
  rebuild —
  `find obj/usr/extension obj/modules/chemistry -name "*.o" -exec touch {} +`
  — or wait for their commit.  Only `src/modules/dynamics/mhd` is this
  line's territory.
* **Ghost zones**:  dump readers must mind them — `bf` has n_gh = 2
  and its ghost slots keep INITIAL values ( a whole-array max lies );
  `mg_field` / `mg_src` have n_gh = 1 with stale slave slots after the
  MG step ( slice real cells only );  `r_arr` / `e_arr` are NOT in the
  dumps.  Interface ghost slots of `bf` carry only the
  interface-normal component ( tangential slots may be NaN ) — mask
  ( `nanmax` )  or avoid.
* **Par-file section trap**:  keys appended after the last section
  header land in that section and silently no-op;  `grep -c` with zero
  matches exits 1 and breaks `&&` chains.

Probes ( all under `~/scratch/non_ideal_tst/runs/mhd_ni/` ):
`ni_d4/divb.py` ( cylindrical CT divergence,  used by the gate ),
`ni_sph21/divb_sph.py` ( spherical,  CLI on a dump ),
`ni_d4/run_bc/bcmp.py` ( shell / interior split ),  `t1.py` ( T1
analytic comparison ),  `vareta/cmp_b*.py` ( face-field comparators );
generic dump reading via `visual/binary_io.py` /
`visual/hydro_data.py`.

---

## 8.  Known limitations ( must-read before production runs )

1. **Hall implicit** — absent ( § 1.1 ).  Disk-wind science must plan
   the Hall treatment separately.
2. **Energy feedback** — implemented  ( § 3.5,  2026-10-11 ):
   the diffusion-Poynting-flux pass  ( flux form,  NOT a `J^2` source —
   that double-counts ),  validated to 96 % closure on the annulus ring;
   `mg_cc_ene = 0`  provides the with/without-heating efficiency switch.
3. **MPI untested** ( single-rank development only ).
4. Cylindrical SMR fre-BC corner columns drift 2–3e-3 ( B2 state );
   mitigations:  essential BC ( `mg_cc_bc = 1` ),  full-period fine
   blocks,  or avoiding 3-axis-interface blocks.
5. `mg_cc_exact` ( FEEC )  + variable η not audited ( throws );
   z-profiles Cartesian-only;  `mg_cc_fmatch = 1` removed ( throws ).
6. The Picard freeze is first-order in time for strongly-varying η
   ( chemistry agents:  verify η variation within dt is small,  or
   sub-cycle the freeze ).
7. Removed knobs ( silently ignored in old .pars ):  `mg_cc_gtrack`,
   `divbc`,  `bmask`,  `corner`,  `bcons`,  `lintrk`,  `bpin`,  `zgc`,
   `pecfill`,  `pvfill`,  `divp`,  `curvi_smr`.

---

## 9.  Pointers

| What | Where |
|---|---|
| Algorithm / derivation / validation document | `docs/mhd/nonideal/implicit-nonideal-mg-method.md` |
| Method paper ( LaTeX + figures ) | `~/scratch/non_ideal_tst/paper/implicit_ni_mg.tex` ( Overleaf repo in `paper/overleaf/` ) |
| Chronicle ( how / why,  stages A–N,  V,  W,  P,  B ) | `~/scratch/non_ideal_tst/status-smr-curvilinear-2026-10-05.md` |
| Literature surveys | `~/scratch/non_ideal_tst/reference/` ( boundary BC,  div-free MG,  face-centred MG,  … ) |
| Solver source | `src/modules/dynamics/mhd/nonideal/` :  `implicit_cc.h` ( the mg_cc driver + kernels ),  `implicit_mg.h` ( base operator / transfers ),  `nonideal.h` ( explicit chain,  providers,  EOS wrapper ),  `implicit.h` ( solver-2 GS-Laplacian ancestor ),  `mg_ni_base.h` ( the MG overload base ) |
| MG substrate ( pristine,  do not modify for ni ) | `src/modules/multigrid/` — all ni-specific changes live in `nonideal/` as overloads |
| Reference pgen | `usr_ext/std_tst/mhd_ni_tst/usr.cpp` |
| Regression gate + probes + all .par | `~/scratch/non_ideal_tst/runs/mhd_ni/` |
| Debug farm | `~/apps/kratos_mhd_cuda_dbg` ( HIPCPU DEBUG=1 ) |
| Interface-leak history ( Stages D–K ) | `docs/history/curvilinear-smr-interface-leak.md`（**已移出阅读路径**，`git log --diff-filter=D` 可找回） |
| Framework user guide ( modules,  pgen composition ) | `docs/README-USER.md` |

## 散度投影（Step B，`[nonideal] mg_cc_proj*` / `proj*`）

每步在写回 `bf`、`thin_axis_sync_ni` **之后**跑一次：解泊松 `dg·λ − Σλ = S`（`S` = 逐面通量失衡，
`dg` = 该格**可修正面**的个数，`mg_cc_face_ok ≡ ct_sfc > 0`），再按 `ΔB_a = (λ_j − λ_{j−e})/A_a` 修正面场。
理论见 `emf-unknown-route-plan.md` §3quater，实现见 `implicit_mg.h`，实测见 `STATUS.md` §3bis。

| 参数 | 默认 | 含义 |
|---|---|---|
| `mg_cc_proj` | 0 | 1 = 打开投影（**必须**配 `mg_cc_projchk = 1` 才看得到 `MGCCPROJ`/`MGCCDIV`） |
| `mg_cc_proj_iter` | 200 | RB-SOR 的 sweep 数（**生产用 2000**：2.5D/3D 都到 `resid ~1e−18`） |
| `mg_cc_proj_om` | 0 | 松弛因子 ≤0 = 自动（`2/(1+sin(π/n_max))`，上限 1.95） |
| `mg_cc_projchk` | 0 | 1 = 打印 `MGCCETA` / `MGCCPROJ resid` / `MGCCDIV` / `MGCCDIVARG` |
| `proj_cell` | — | `i0 i1 i2`：打印该格六面的 `ΔB`/`ΔF`（`MGCCPROJCELL`） |
| `proj_mg` | **0** | 1 = 用共享多重网格 V-cycle 代替 SOR。**实测：正确但没有性能优势，且有单精度地板**（`STATUS.md` §3bis / `proj-multigrid-plan.md` §9）。**不要在生产上开** |
| `proj_mg_v` | 20 | `proj_mg = 1` 时的 V-cycle 预算 |
| `proj_mg_dtol` | 1e−9 | `proj_mg = 1` 时的相对 defect 退出阈值 |
| `proj_mg_sc` | 0 | `proj_mg = 1` 时的粗层标度指数（`sc = 2^(−q·L)`）；实测 q=1 比 q=0 好 ~6× |

**判读**：`MGCCDIV max|D.B| = <入场> in -> <解前> solved -> <投影后> proj`。生产上"投影后"应是 **~2e−9**
（2.5D 与 3D 相同）。**2.5D 的验收基线**（`tdt.par`，cycle 0）：`resid ≈ 1e−18`（浮点地板）、
`max|lam| ≈ 5.227e−4`、`proj ≈ 2e−9`、`in = 0.495513`、`max|S| = 0.00242049`。
⚠️ **逐位基线只在掩码修复那一步成立**（`resid = 9.75782e−19`、`max|lam| = 0.000522682`、`proj = 1.64029e−9`）；
2026-10-23 的 (A) 修复（`mg_cc_hydro_k` 的 v 槽位写速度）**故意改动了默认路径**，2.5D 随之移到
`resid = 8.67362e−19`、`max|lam| = 0.000522666`、`proj = 1.99805e−9`，而 `in` / `max|S|` / `dv_pre` **不变**。
