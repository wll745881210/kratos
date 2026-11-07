# Tensor non-ideal MHD

The MHD module can add a diffusive sub-step for the magnetic field: Ohmic,
ambipolar and Hall diffusion.  The ideal integrator is unchanged; this module
wraps it and supplies either the non-ideal electromotive force, or an implicit
solve for the field update.

## What it solves

The induction equation gains

    dB/dt = -curl( eta_O J + eta_H (J x b) + eta_A (J - (J.b) b) ),   J = curl B,

where `B` is the magnetic field, `b = B/|B|`, `J` is the current density, and
`eta_O`, `eta_A` and `eta_H` are the Ohmic, ambipolar and Hall diffusivities.
The first and third terms define the symmetric tensor
`eta = eta_O I + eta_A (I - b b)`; the Hall term is antisymmetric.  The module can
deposit the magnetic energy lost to diffusion as heat (`mg_cc_ene`, `mg_cc_ene_heat`).
## Enabling it

The module is a template wrapper around an MHD integrator and a diffusivity
provider.  Enroll it in place of the plain integrator, then call `diffuse()`
once per cycle, after the ideal update:

```cpp
#include "../../src/modules/dynamics/mhd/nonideal/explicit_sts/nonideal.h"

using itg_t = dynamics::mhd::integrator::rk2_t<>;     // sph_t<> for spherical
using eta_t = mhd_ee::ni::const_eta_t;
using ni_t  = mhd_ee::ni::ni_t< itg_t, eta_t >;
using eos_t = mhd_ee::ni::ni_eos_t< eta_t >;
mod.enroll< dynamics::mhd::proxy_t, ni_t, eos_t >();
static_cast< ni_t & >( * mod.p_itg ).diffuse( mesh, mod );   // in the cycle
```

`ni_eos_t` wraps the adiabatic equation of state and adds the parabolic
time-step limit.  `const_eta_t` reads `eta_o`, `eta_h` and `eta_a`;
`hall_len_eta_t` adds `hall_l`; `prof_eta_t` reads the `prof_*` keys and is
Ohmic + ambipolar only.  With every diffusivity zero no kernel is launched.  A minimal explicit block:

```ini
[nonideal]
eta_o = 1.0e-3
mode  = 0
```
## Routes

The module has an explicit family and an implicit family; `[nonideal] mode`
selects among them.

| route | selected by | update |
|---|---|---|
| coupled explicit | `mode = 0` (default) | one explicit CT update per step; the diffusion EMF is added to the ideal EMF before the edge exchange |
| Godunov split | `mode = 1`, or `sts = 1` | a separate diffusion sweep after the ideal step |
| face-centred implicit | `mode = 2` | one backward-Euler solve of the face-centred Laplacian per step, by red-black Gauss-Seidel |
| edge-centred implicit | `ee_enable = 1` and `ee_solve = 1` | one implicit solve for the edge 1-form per step, by preconditioned conjugate gradients, then a curl write-back |

With `sts = 1` the module forces `mode = 1`, refreshes the primitives once and
covers the diffusion with damped Chebyshev sub-steps, which relaxes the
parabolic limit.  `ee_enable = 1` with `ee_solve = 0` runs the setup and the
diagnostic only and leaves the state untouched.  `ee_enable` and `mode = 2` are
two different implicit legs and are refused together, as are `ee_enable` and
`sts = 1`.
## Algorithm sketch (implicit route)

1. Build the edge 1-form `a`: the circulation of the vector potential, i.e. the
   time-integrated electric field along each primal edge.  Polar edges of zero
   length are removed from the lattice.
2. Solve `(M1 + dt K) a = M1 a^n`, with `M1 = int phi_e . phi_e'` and
   `K = int (eta curl phi_e) . (curl phi_e')`, by PCG preconditioned with a
   matrix-free multigrid V-cycle: 12x12 per-cell blocks, 2:1 coarsening in `r`
   and `theta`, Galerkin coarse operators, block-Jacobi smoothing, warm start.
3. Write `dB = C a` onto the faces, with `C` the discrete curl.  The divergence
   of a discrete curl is zero, so `D dB = 0` by construction.  The increment
   adds no divergence and leaves the divergence control of the ideal update
   unchanged.
4. The explicit route keeps the same property through the CT update: its edge
   current is the discrete adjoint of the curl, and its EMF enters the CT update.
## Parameters

Every `[nonideal]` key; the default is the value in the code.

| key | type | default | meaning |
|---|---|---|---|
| `eta_o`, `eta_a` | real | 0, 0 | Ohmic and ambipolar diffusivities |
| `eta_h` | real | 0 | constant Hall diffusivity |
| `hall_l` | real | 0 | ion-skin length; `eta_H = eta_h + hall_l sqrt(B.B)/sqrt(rho)` (`hall_len_eta_t`) |
| `prof_eta_o`, `prof_eta_a` | real | 0, 0 | Ohmic and ambipolar amplitudes of the profile provider |
| `prof_rho_ref` | real | 1 | reference density of the profile |
| `prof_alpha_o`, `prof_alpha_a` | real | 0, 0 | Ohmic and ambipolar density exponents |
| `mode` | int | 0 | 0 coupled explicit, 1 split, 2 face-centred implicit (see Routes) |
| `sts` | int | 0 | 1 = super-time-stepping (forces `mode = 1`) |
| `sts_nu` | real | 0.01 | damping of the RKC sub-steps |
| `mg_cc`, `mg_cc_eta_var` | int | 0, 0 | must stay zero; a positive or nonzero value throws |
| `mg_cc_ene` | int | 2 | energy deposit: 0 off, 2 measured form |
| `mg_cc_ene_heat` | int | 1 | 1 adds `dt (eta_O J^2 + eta_A J_perp^2)`; 0 keeps the field change only |
| `bc_edge` | int | 0 | 0 free one-sided, 1 essential `E_t = 0`, 2 pseudo-vacuum |
| `hall_chyp` | real | 0.05 | fourth-order hyper-resistivity weight; applies only with a Hall diffusivity |
| `hall_upw` | real | 0 | upwind Hall bias; the bias is capped at 0.45 |
| `ee_enable` | int | 0 | 1 = edge-centred implicit leg |
| `ee_solve` | int | 0 | 0 = setup and diagnostic only, 1 = solve and write back |
| `ee_subcell` | int | 0 | 1 = sub-cell `eta` gradient |
| `ee_coarsen` | int | 3 | bit mask: 1 = r, 2 = theta, 4 = phi |
| `ee_nlevel` | int | 8 | maximum number of multigrid levels |
| `ee_n_smooth` | int | 2 | smoother sweeps per level |
| `ee_iter` | int | 100 | PCG iteration cap |
| `ee_tol` | real | 1e-5 | PCG tolerance; 1e-12 at `PRECISION=2` |
| `ee_coarse_iter`, `ee_coarse_tol` | int, real | 2, 1e-10 | coarsest-level iteration cap and tolerance |
| `ee_writeback` | int | 0 | 0 strong curl increment, 1 representation increment |
| `ee_dt` | real | 0 | 0 = cycle `dt`; a positive value forces this `dt` |
| `ee_a_bridge` | int | 0 | 1 = rebuild the edge state from `B` at start |
| `ee_verb` | int | 0 | 1 = print one line per solve |
| `[profile] eta_out_kill` | int | 0 | zero the profile in the last N cells of `x0`: 1 both, 2 Ohmic, 3 ambipolar |

Neighbouring keys: the implicit route needs `[mesh]` to give a spherical block
with uniform `r` and `theta` spacing.  `[boundary] kinds` marks `out` and `per`
faces; a periodic axis has no physical boundary.  `[dynamics] cfl` sets the
ideal step and the non-ideal cap applies on top; `[cycle] dt_init` is the step
used for the first step when `ee_dt = 0`.
## Supported and not supported

| supported | not supported |
|---|---|
| explicit Ohmic, ambipolar and Hall in Cartesian, cylindrical and spherical geometry, including SMR blocks | the implicit route outside a uniform spherical (`r`, `theta`) block |
| super-time-stepping in three dimensions | super-time-stepping on a two-dimensional or thin-axis block (throws) |
| the implicit route with Ohmic + ambipolar on a single block, including polar cells | more than one block in the implicit solve, or Hall there (`eta_h` and `hall_l` must be zero; throws) |
| 2:1 coarsening in `r` and `theta`; `phi` may be periodic or stretched | `mode = 2` together with `ee_enable` (throws) |
| `mg_cc_ene` 0 or 2 | `mg_cc > 0`, `mg_cc_eta_var != 0`, or `mg_cc_ene` 1 and 3 (throw) |
| `const_eta_t`, `hall_len_eta_t`, `prof_eta_t` | `chem_eta_t` (throws) |
