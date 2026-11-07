# Gray flux-limited diffusion

The module advances one gray radiation energy density `E`, coupled to matter through
emission and absorption, with an implicit multigrid solve.

## What it solves

With `D` the diffusion coefficient, `kappa` the gray opacity and `S` the source,

    dE/dt = div( D grad E ) + c kappa rho (a T_m^4 - E) + S,

and one backward-Euler step is

    (I - dt div D grad + sigma) E = E^n + dt S + sigma a T_m^4,
    sigma = dt c kappa rho,   D = c lambda / (3 kappa rho).
`lambda` is the flux limiter and `E` an energy density in code units.  The module
adds `sigma a T_m^4` only with `t_coup`; otherwise emission comes from the
host-filled per-cell source `s_ir`, included when `s_on = 1`.
## Closure and opacity

`lambda` is `1/3` unless `var_coef = 1`, which freezes `D` and `sigma` per cell
from `R = 3 D |grad E| / E`; `limiter = 0` uses `1/3`, `1` Levermore-Pomraning,
`2` Minerbo.  `closure = 1` freezes the M1 tensor `chi` and rebuilds `F` after
the solve; the operator stays scalar.
The uniform coefficients are `d_unif` (= `D`) and `sig_unif` (= `c kappa rho`).
`kappa`, `kappa1` and `kappa_var` add a linear profile in the first coordinate
(active with `var_coef = 1`): `D` scales as `1/kappa`, `sigma` as `kappa`.  For a
host-owned per-cell opacity set `freeze_byp = 1` and write `co_arr[0] = D dt`,
`co_arr[1] = sigma`; `fld_kap.h` defines the `kappa_IR` contract.
## Boundaries

`[boundary] kinds` decides which faces are physical; a periodic axis has none.
`robin_lo`/`robin_hi` (bit 0, 1, 2 per axis) with `f_inc` set a Robin ghost
`E_g = E_i + dx f_inc / D`; `dir_lo`/`dir_hi` with `e_dir` anchor `E`, in steady
runs only.
At a physical face radiation escapes at the Marshak rate `c E/4` when
`lte_elim = 1` and `steady = 1`.  A cell with `sigma > c dx/4` is optically thick
and falls back to zero gradient; the inner radial face always drains.  Otherwise
the face is closed and traps energy unless an inflow is set.
## Coupling

`t_coup > 0` adds `sigma dt a_rad T_matt^4` to the source; `s_on = 1` includes
the host-filled `s_ir[0] + s_ir[1]`, `src_emit` a uniform volumetric term.
`lte_elim = 1` drops the absorption diagonal where `sigma > elim_sig` (local
thermodynamic equilibrium); `newton = 1` puts `5 sigma` there instead.
With `var_coef = 1` and `limiter != 0` the solve is wrapped in a Picard loop:
refreeze from the new `E`, solve, at most `n_picard` rounds, until
`max|E - E_old| / max|E| < r_tol`.
## Enabling it

```cpp
#include "../../usr/radiation/fld/fld_driver.h"

struct prx_t : multigrid::proxy_default_t
               < rad::fld::bdt_fld_t< type::float_t > >
{
    static void set_module( mesh::mod_base_t & ) {}
};
using fld_t = rad::fld::fld_t< prx_t >;
mesh.enroll_module< fld_t >();   // read() enrolls the operator
```

```ini
[fld]
enabled  = 1
d_unif   = 3.0e-2
sig_unif = 1.0e-1
```

`read()` rejects a configuration with no absorption, no per-cell coefficients
and no Robin inflow: the operator would be singular.  The module steps with the cycle `dt`.
## Parameters

Grouped by role; every key with its default from the code.

| key | type | default | meaning |
|---|---|---|---|
| **Solve** | | | |
| `enabled` | int | 0 | 1 = run the module |
| `d_unif`, `sig_unif` | real, real | 0, 0 | uniform `D` (not scaled by `dt`); uniform `c kappa rho` |
| `var_coef` | int | 0 | 1 = freeze `D` and `sigma` per cell |
| `n_picard`, `r_tol`, `freeze_byp` | int, real, int | 1, 1e-4, 0 | Picard rounds, stop, host-owned freeze |
| **Opacity** | | | |
| `kappa`, `kappa1`, `kappa_var` | real, real, int | 0, 0, 0 | linear profile; `kappa <= 0` becomes 1 |
| `e_floor`, `rho` | real | 1e-24, 1 | limiter/profile floor; `rho` unused by the freeze |
| **Closure** | | | |
| `limiter`, `closure`, `c_light` | int, int, real | 0, 0, 1 | 0/1/2 limiter; scalar or M1; speed of light |
| `chi_mode`, `chi_a` | int, real[3] | 0, 1/3 1/3 1/3 | 0 = Levermore `chi`, 1 = fixed diagonal `chi_a` |
| **Boundaries** | | | |
| `robin_lo`, `robin_hi` | int | 0, 0 | Robin face mask (bit 0, 1, 2 per axis) |
| `dir_lo`, `dir_hi` | int | 0, 0 | Dirichlet anchor mask |
| `f_inc`, `e_dir` | real | 0, 0 | incoming Robin flux, anchor energy density |
| **Coupling** | | | |
| `t_coup`, `a_rad`, `t_matt` | int, real, real | 0, 1, 0 | add `sigma dt a_rad T_matt^4`; matter temperature |
| `s_on`, `src_emit` | int, real | 0, 0 | include `s_ir`; uniform volumetric source |
| `lte_elim`, `elim_sig`, `newton` | int, real, int | 0, 1, 0 | LTE mask; `5 sigma` diagonal |
| `steady`, `f_accum` | int, int | 0, 0 | equilibrium solve/Marshak; accumulate the M1 flux |
| **Output and multigrid** | | | |
| `[multigrid] mg_lvl` | int | -1 | deepest level; -1 = from the block size |
| `[multigrid] n_iter`, `n_smooth` | int | 1, 1 | V-cycles per step, smoother sweeps |
| `[multigrid] dtol`, `iter_first` | real, int | 5e-4 (5e-6 double), 20 | defect tolerance, cycles on the first call |
| `[multigrid] print_iter`, `stag` | bool, real | false, 1.5 | defect trace, stagnation factor |
| `[multigrid] phy_bnd_type`, `thread_limit` | string, int | "free", 32 | boundary policy, threads per block |
## Algorithm sketch

1. Take `dt` from the cycle; scale `D dt` and `sigma = sig_unif dt`.
2. Seed the right-hand side with `E^n` (zero if `steady = 1`), the sources and
   the boundary rows; a Robin ghost adds its flux.
3. Freeze `co_arr` at level 0 (uniform or per cell), plus `chi` if `closure = 1`.
4. Run V-cycles on `(I - dt div D grad + sigma)` with 2:1 coarsening of the
   coefficients and a smoother; outflow faces add `c E/4`, or zero gradient
   when the cell is thick.  Stop at `dtol`, at `n_iter`, or on `stag`.
5. Refreeze within `n_picard`; rebuild `F` if `closure = 1`; matter couples via the source.
## Limitations

- One gray group; no frequency dependence or spectral regrouping.
- No matter-temperature update: the module does not evolve `T_m`, so the heat it
  deposits must be fed back by the host through `s_ir`.
- `closure = 1` rebuilds the flux with M1 but leaves the operator scalar.
- The Marshak outflow needs `lte_elim = 1` and `steady = 1`; a coupled run uses
  the cycle step.
