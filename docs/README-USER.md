# Kratos user guide

Kratos is a multi-physics astrophysical simulation framework for heterogeneous computing devices:
compressible hydrodynamics and magnetohydrodynamics, non-equilibrium thermochemistry, radiation
transport, self-gravity and tracer particles on block-structured adaptive meshes. The code is C++17,
with optional CUDA, HIP or MUSA backends and optional MPI.

The framework is compositional. The trunk supplies algorithms; a problem generator (pgen) chooses
which of them a binary contains. A binary that enrolls only hydrodynamics is a hydrodynamics code;
one that enrolls only the radiation modules is a radiative-transfer code. The module set is fixed at
compile time. The contributor contract is `AGENTS.md` at the repository root.

## 1. Framework

### 1.1 Layers

| layer | type | responsibility |
|---|---|---|
| device | `device::base_t` | memory, streams, kernel launch; `device_t` is the backend alias |
| communication | `comm::base_t` | rank, reductions, ghost exchange; MPI or serial |
| mesh | `mesh::mesh_t` | blocks, refinement tree, geometry, module registry |
| grid | `meshgen::base_t` | `[mesh]` keys, tree, coordinates, load balancing |
| time loop | `cycle::base_t` | time step, CFL reduction, snapshots |
| snapshot I/O | `binary_io::base_t` | the binary snapshot format |
| physics | `mesh::mod_base_t` | per-block data and its updates |
| problem | `prob::run` | enrolls the layers above for one problem |

`mesh::mesh_t` holds geometry only and has no knowledge of which physics modules are enrolled. The
infrastructure objects are enrolled explicitly or filled in by default; the device is required, and
`meshgen::base_t` is the Cartesian generator with `cyl_t` and `sph_t` deriving from it.

### 1.2 Index conventions

The state of a block is one array per cell; the hydrodynamic entries are qualified with `idx::`:

```cpp
u[ idx::rho ]      // mass density
u[ idx::ene ]      // total energy density
u[ idx::mom + a ]  // momentum density along axis a = 0, 1, 2
```

The qualification marks internal hydrodynamic indices rather than arbitrary offsets; the MHD state
adds `idx::bcc` for the cell-centred field. Cells are indexed by a `type::idx_t` triple,
`utils::loop( geo.n_ceff )` visits the cells of a block, and `geo.x_cc( i )` returns a cell centre.

## 2. Modules

A physics module derives from `mesh::mod_base_t` and overrides the parts of the life cycle it uses:

| method | when it runs | what to do there |
|---|---|---|
| `read( args )` | once, before any module initializes | read parameter keys |
| `init( args, mesh )` | once, after every `read` | allocate, build device state |
| `init_cond( d )` | for each new block | fill the initial state of one block |
| `step( mesh )` | once per time step | advance this module by the cycle's `dt` |
| `sync_streams( mesh )` | after `step` | drain this module's streams |
| `finalize( mesh )` | at exit | free device memory |

`init_cond` runs on every block that appears, not only at `t_0`: the mesh calls it during `init` and
again whenever refinement or a load-balance change creates a block. Read parameter keys in `read`,
not in `step`: every module's `read` completes before any module's `init` starts.

### 2.1 Compile-time composition

A module's numerical choices are template parameters, not runtime flags. For hydrodynamics,
`dynamics::hydro_t::enroll` fixes six policies: proxy, integrator, equation of state, reconstruction,
Riemann solver and boundary communication. The defaults are second-order Runge-Kutta, ideal-gas
equation of state, piecewise-linear reconstruction, an HLLC Riemann solver and hydro boundary
exchange.

A pgen that wants a different integrator derives from `dynamics::hydro_t` and passes its own policy
to `enroll`, for example `enroll< dynamics::hydro::proxy_t, itg_T >`. The CRTP base in
`dynamics::hydro::integrator` calls the derived object's device functions (`reconstruct`, `riemann`,
`add_flux`, `flux_to_du`), so the policy is resolved at compile time. The equation of state follows
the same pattern: derive from `dynamics::eos::interface_t< derived_T >` and supply the device
functions `c2p`, `dt` and `n_hvar`.

### 2.2 Block data and proxies

Each block owns two copies of its state, one on the host and one on the device.
`mesh::block::block_t` is the geometry: ids, owning rank, logical region, coordinates and
neighbours. `mesh::block::dual_t` pairs the host and device payloads of one block with the stream and
event used to move between them: `d.h()` is the host copy and `d.d()` the device copy. The payload
type `mesh::block::base_data_t` is module-defined; a module derives its data type and its proxy
through `enroll_bdk` and `enroll_dat`. The proxy adapts a generic block to that data:
`proxy_t::ref( h )` casts the host payload, `proxy_t::d( bd )` builds the device view passed to
kernels, and `set_module( mod )` lets a derived proxy publish the module. The default `ref` and `d`
in `src/mesh/block/block_proxy.h` are identity casts.

### 2.3 Cross-module sharing

Modules often need another module's data: chemistry needs the hydro state, and a radiation module
may need the particle pool. The extension proxy base `extension::prx_base_t` holds static weak
pointers to the enrolled hydro, multigrid and particle modules, and a block-level accessor such as
`d_hyd( reg )` returns the other module's device data for a region. Each module publishes itself in
`set_module` with `p_hyd = weak_from_this()`, so no module owns another. Sharing is by pointer: a
shared buffer is stored once per block, and a module that only reads it copies nothing. A module
that derives a composite view (for example the chemistry hydro proxy, which adds radiation fields to
the hydro block data) shallow-copies the base struct and the pointers at the start of the step.

### 2.4 Parasite modules

A module that needs no block data of its own can reuse another module's block map. `parasite( q )`
adopts the block allocation of `q`, sets `need_update = false`, and records `q` as a weak pointer.
Its `update` then does nothing, and its `finalize` leaves the shared allocation to the host module,
so a correction or a diagnostic runs on the same decomposition without a second allocation.

### 2.5 Execution order

The phases run in a fixed order: every module's `read` in enrollment order; then comm, device, cycle
and the grid generator, then module `init`s in ascending `i_init`; each step runs a grid update and
then every module's `step` in ascending `i_step`; finalize runs in enrollment order.
`enroll_module< T >()` gives a module the next free order, so modules run in enrollment order;
`enroll_module< T >( i_init, i_step )` places one explicitly, and two modules must not share a slot.

## 3. Problem generators

A problem generator is a directory that defines `prob::run`. `src/main.cpp` throws if no parameter
file is given and then calls `prob::run`; the default in `src/user/probgen.cpp` is a weak symbol that
prints `Default run function is empty!`, so a binary is usable only when the pgen supplies a strong
override. The shipped generators live in the separate `kratos_usrproj` repository; the example below
is complete and self-contained. Its include paths are written for a generator directory three levels
below the repository root (`usr_ext/std_tst/<pgen>/`), so `../../../src/...`; a generator one level
higher, in `usr_ext/<pgen>/`, uses `../../src/...`.

The generator derives from a module and fills in its hooks. This one starts from a uniform state:

```cpp
#include "../../../src/device/general.h"
#include "../../../src/comm/mpi/mpi.h"
#include "../../../src/mesh/mesh_enroll.h"
#include "../../../src/modules/dynamics/hydro/hydro.h"
#include "../../../src/types.h"
#include "../../../src/user/probgen.h"
#include "../../../src/utilities/mapping/loop.h"
namespace prob
{
class hydro_t : public dynamics::hydro_t
{
    type::float_t rho, pre, vel, gam;

    void read( const input & args ) override
    {
        rho = args.get< type::float_t >( "prob", "rho", 1.e0 );
        pre = args.get< type::float_t >( "prob", "pre", 1.e0 );
        vel = args.get< type::float_t >( "prob", "vel", 0.e0 );
        gam = args.get< type::float_t >( "dynamics", "gamma", 1.667 );
        dynamics::hydro_t::read( args );
        return enroll<>(  );
    }

    void init_cond( mesh::block::dual_t & d ) override
    {
        using prx_t = dynamics::hydro::proxy_t;
        const auto & b_h = prx_t::ref( d.h(  ) );
        for( auto i : utils::loop( d.b_w.get(  ).geo.n_ceff ) )
        {
            auto * u = b_h.hyd.u.at( i );
            u[ idx::rho ] = rho;
            u[ idx::ene ] = pre / ( gam - 1 ) + rho * vel * vel / 2;
            u[ idx::mom ] = rho * vel;
            u[ idx::mom + 1 ] = u[ idx::mom + 2 ] = 0;
        }
    }
};

void run( int argc, char * argv[] )
{
    mesh::mesh_t mesh;
    mesh.enroll_device< device_t >(  );
#ifdef __MPI__
    mesh.enroll_comm< comm::mpi_t >(  );
    mesh.enroll_binary_io< binary_io::mpi_t >(  );
#endif
    mesh.enroll_module< hydro_t >(  );
    input args;
    args.set_comm( mesh.p_com );
    cmd( argc, argv, args );
    mesh.init( args );
    return mesh.evolve(  );
}
}                       // namespace prob
```

`read` sets defaults, reads the pgen's `[prob]` keys and the `[dynamics] gamma` it needs for the
initial state, then calls the base `read` to enroll the default schemes; other shared keys are read
during initialization by the module that owns them. `init_cond` runs on the host for every new
block, with `prx_t::ref( d.h() )` giving typed access to the host state and the entries of `u` the
conserved variables. `run` enrolls the device, MPI communication and I/O when
the build has MPI, and the module; `cmd` reads the command-line files, `mesh.init` creates the
blocks, and `mesh.evolve` runs the cycle.

### 3.1 Replacing a component

| goal | base class | override |
|---|---|---|
| new physics | `mesh::mod_base_t` | block data, `read`, `init_cond`, `step` |
| different hydro scheme | `dynamics::hydro_t` | the policies passed to `enroll` |
| different equation of state | `dynamics::eos::interface_t< derived_T >` | device `c2p`, `dt`, `n_hvar` |
| different grid geometry | `meshgen::base_t` | `geo_type`, `location`, `surface`, `volume` |
| different time loop | `cycle::base_t` | `step`, `evolve` |
| different snapshot format | `binary_io::base_t` | `open`, `save`, `load` |
| new reaction rate form | `chemistry::reaction::std_t< derived_T >` | `rate_coef` |

## 4. Build and run

GNU Make is the only build system; the binary is `bin/kratos` and only one pgen is compiled per
binary.

| variable | default | options |
|---|---|---|
| `USRDIR` | `usr` | directory of the pgen to compile |
| `ARCH` | `CUDA` | `CUDA`, `HIP`, `HIPCPU`, `MUSA` |
| `DEBUG` | `0` | `1` adds `-g -O0`, `__GPU_DEBUG__`/`__CPU_DEBUG__` and `debug_macros.h` |
| `MPI` | `0` | `1` links MPI and defines `__MPI__` |
| `PRECISION` | `1` | `0` all float, `1` mixed, `2` all double |

```bash
make USRDIR=<pgen_dir> ARCH=CUDA MPI=1 PRECISION=2 -j12
bin/kratos run.par
```

Sources are discovered automatically from `src/`, `$(USRDIR)` and `$(EXTDIR)`; there are no source
lists. `make clean` removes `obj/` and `bin/`, and `make remake` runs `clean` then `all`. Always
`make clean` when switching `ARCH`. A binary without a pgen prints `Default run function is empty!`.

The command line accepts one or more parameter files; later files override earlier ones. A snapshot
is written as `<prefix_output>_<5-digit index>.bin`, and `visual/hydro_data.py` reads it back with
`get_dd()` and `get_last_dd()`.

## 5. Parameter files

Parameter files are INI-like: a `[section]` header, then `key = value` lines, with `#` starting a
comment. A `[mesh]` section might hold `n_cell_global = 512 2 1`, and `[dynamics]` `gamma = 1.4`.

Internally each entry is stored as `section|key`, so programmatic access is
`args.get< T >( "section", "key" )` or `args( "section", "key", value )`. A key read with a default
never throws; a key read without one throws if it is missing. Array values are whitespace-separated
lists. Values are not type-checked until they are read, and an unread key is silently ignored.

| section | configures | representative keys |
|---|---|---|
| `[mesh]` | extent, block size, geometry, refinement | `x_min`, `x_max`, `n_cell_global`, `n_cell_block` |
| `[cycle]` | time interval, step control, output cadence | `t_0`, `t_lim`, `dt_init`, `dt_output`, `prefix_output`, `except_ferr` |
| `[device]` | device selection and streams | `idx_device`, `max_streams`, `seed_rng` |
| `[boundary]` | boundary conditions, six entries | `kinds`, `ghost_init`, `omega_star` |
| `[dynamics]` | hydro and MHD scheme and floors | `gamma`, `cfl`, `rho_floor`, `tmp_floor`, `reconstruction` |
| `[chemistry]` | species, network, ODE solver | `species`, `reaction_file`, `solver`, `rtol`, `atol`, `reac_on_const` |
| `[radiation]` | ray-tracing radiation | `n_ray_cell`, `n_scat`, `injection`, `dir`, `flux_cgs` |
| `[multigrid]` | Poisson solver for self-gravity | `mg_lvl`, `n_iter`, `dtol`, `n_smooth` |
| `[particle]` | tracer and CIC particles | `n_par`, `f_par`, `step_max`, `n_th` |
| `[fld]` | grey flux-limited diffusion | see `docs/radiation/fld.md` |
| `[nonideal]` | non-ideal MHD terms | see `docs/mhd/nonideal.md` |
| `[init]`, `[profile]` | background state and disk/atmosphere profiles | `rho0`, `pre0`, `b0`; `n0`, `r0`, `T0` |
| `[prob]` | the pgen's private keys | defined by the pgen |

The disk and atmosphere templates also read `[suppress]`, `[guard]`, `[dust]` and `[sat]`.

The `[boundary] kinds` list has six entries in the order `x- x+ y- y+ z- z+`; faces outside the
problem dimension are still listed and ignored. A periodic direction is marked with `per`. The
accepted kinds are:

| kind | meaning |
|---|---|
| `ref` | reflecting wall; flips the normal momentum |
| `out` | outflow: zero-gradient state, inward normal velocity clamped |
| `fre` | free: zero-gradient, inflow allowed |
| `fixd` | fixed ghost state taken from `[init]` |
| `per` | periodic; pairs the two faces |
| `pol` | spherical theta through-pole continuation; theta sides only |
| `axs` | cylindrical r through-axis continuation; radial sides only |

## 6. Chemistry

A reaction network is a `[chemistry]` section that lists the species and points at a reaction
database. Species names are parsed for charge suffixes (`+`, `-`), parentheses, `*` markers and
element counts; `e-` is the electron. A reaction whose species is not in the list is silently
dropped, so a misspelt species shows up as a missing reaction rather than an error.

### 6.1 Abundance representations

| symbol | definition | units | where |
|---|---|---|---|
| `n_i` | number density | cm^-3 | database physics |
| `y_tilde_i` | `n_i m_A / rho_0` | dimensionless | what the hydro solver advects |
| `y_i` | `n_i m_A / rho` | dimensionless | what the rate solver consumes |

`rho_0` is the code-unit density in CGS, set by the pgen, and `rho` is the local mass density; the
solver converts by dividing by `rho_tilde = rho / rho_0`. The dimensionless form is numerical:
number densities in an atmosphere or a disk span many decades, and float arithmetic loses precision
away from unity.

The mass-action law in number densities is `dn_i/dt = xi k_X prod_j n_I`; multiplying by `m_A/rho`
gives `dy_i/dt = xi rho^(R-1) lambda_X prod_j y_I` for `R` reactants. Kratos applies the `rho^(R-1)`
factor inside `rate()`, so the database file must not include it.

### 6.2 Converting a database prefactor

The INI prefactor `A_tilde` absorbs the CGS-to-code conversion. For `R` reactants,

```
A_tilde = A * m_A^-(R-1) = A * N_A^(R-1)
```

with `m_A = 1.66054e-24 g`, the atomic mass unit, and `N_A = 1/m_A = 6.022141e23`. Only the prefactor
changes; the temperature exponent `b` and the activation energy `C` are copied unchanged, because
`k = A_tilde (T/300)^b exp(-C/T)`.

| `R` | order | conversion | example |
|---|---|---|---|
| 1 | unimolecular, dissociation | `A_tilde = A` | `CH4 -> CH3 + H`, `A` in s^-1 |
| 2 | bimolecular | `A_tilde = A N_A` | `OH + H -> H2O`, `A` in cm^3 s^-1 |
| 3 | termolecular | `A_tilde = A N_A^2` | `H + H + M -> H2 + M`, `A` in cm^6 s^-1 |

**The unimolecular pitfall.** A dissociation rate gets no `N_A` factor. Multiplying it by `N_A` inflates
the rate by about 6e23, which makes the network extremely stiff and collapses the solver step size.

### 6.3 Rate types and the Troe form

A reaction block is `[reaction_000001]` with `reactants`, `products`, `rate_type`, `parameters`,
`energy` and optional `fac_prod`; its entries are whitespace-separated lists.

`rate_type` accepts `ARR` (Arrhenius, the default), `TROE` (pressure-dependent) and `INTP` (same
Arrhenius form); `parameters` is `A b C` for Arrhenius. `energy` is the energy change in erg per
particle; with four coefficients it is `e0 + k_B T (e1 + e2 ln(T)^e3)` in the same units, encoding a
heat-capacity contribution. Listing the same product twice merges the duplicates and sums their
coefficients; reactant multiplicities are kept, and an explicit `fac_prod` list may give non-unit
product coefficients.

Troe falloff uses eleven parameters, in this order:

```
A0  b0  C0   Ainf  binf  Cinf   alpha  T1  T2  T3   mu
```

with the limits and the broadening factor

```
k0   = A0   (T/300)^b0   exp(-C0/T)
kinf = Ainf (T/300)^binf exp(-Cinf/T)
Pr   = k0 n_tot / kinf,      n_tot = rho / (mu m_p)
lg_Fc = log10( exp(-T2/T) + alpha exp(-T/T1) + (1-alpha) exp(-T/T3) )
C     = -0.4 - 0.67 lg_Fc,      N = 0.75 - 1.27 lg_Fc
lg_F  = lg_Fc / ( 1 + (lg_Pr + C)^2 / (N - D (lg_Pr + C))^2 ),   D = 0.14
k     = kinf * Pr/(1+Pr) * 10^lg_F
```

`A0` is the termolecular limiting coefficient and takes the `N_A^2` conversion; `Ainf` is the
unimolecular limiting coefficient and takes none. `mu` is the mean molecular weight of the collider
in units of the proton mass `m_p`. When `Ainf` is absent or zero the rate falls back to the
low-pressure form `k0 n_tot`.

### 6.4 Memory placement

Reaction data lives in a per-list container that is either in device constant memory or in global
memory. The constant pool is small (tens of kilobytes) and gives the fastest reads, so it is the
default. When a large network or a large cooling table exceeds the budget, set
`reac_on_const = false` in `[chemistry]` to move the reaction sub-lists to global memory; the
cooling, photo-reaction and dust reaction lists share one policy, so the switch applies to all.

## 7. Debugging

Set `except_ferr = 1` in `[cycle]` to trap floating-point errors and device faults at the source:
the mesh enables the divide-by-zero, invalid and overflow traps before any module initializes.

For a long or large run, build with `ARCH=CUDA` and run in production mode. When a run produces a
NaN, an illegal memory access or another unexpected abort, do not switch to `HIPCPU` to debug it.
Keep `ARCH=CUDA`, rebuild the same pgen with `DEBUG=1` after `make clean`, and run under GDB with
`catch throw` and `break abort`: this stops at the throwing frame for C++ exceptions and at a hard
abort for device-side failures. A release build reports the stale return value of the failing CUDA
call, while `DEBUG=1` synchronizes and re-reads the error, so the message belongs to the failing
call. `ARCH=HIPCPU` with `DEBUG=1` instruments only the host side.

| symptom | likely cause |
|---|---|
| `Default run function is empty!` | no pgen defines `prob::run` |
| `device unspecified` | the pgen did not call `enroll_device` |
| `Undef bnd <kind>` | misspelt `[boundary] kinds` entry |
| `Cannot identify ODE solver` | unknown `[chemistry] solver` value |
| reaction missing from the network | a species is not in `[chemistry] species` |
| step size collapses | a unimolecular rate was multiplied by `N_A` |
| constant memory oversize | set `reac_on_const = false` |
