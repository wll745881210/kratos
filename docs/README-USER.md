# README-USER.md — Kratos User Guide

> Covers the framework architecture, module design and interactions, and step-by-step guidance
> for composing problem generators of varying complexity, using four real projects as reference.

---

## 1. Overall Framework and Infrastructures

### 1.1 What Kratos Is

Kratos is a multi-physics astrophysical simulation framework designed for heterogeneous
computing devices (GPUs). It is built directly on GPU programming models (CUDA/HIP) without
intermediate portability layers, achieving zero-overhead abstraction while targeting multiple
backends. The framework is self-contained — no third-party libraries beyond MPI and the C++17
standard library.

**Published method paper:** Wang (2025), *ApJS*, 277:63.

### 1.2 Architecture at a Glance

```
┌─────────────────────────────────────────────────┐
│                  prob::run()                     │  ← You write this
│  Creates mesh, enrolls modules, reads params    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│               mesh::mesh_t                       │  ← Central orchestrator
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ p_dev    │ │ p_com    │ │ p_cyc            │ │  ← Infrastructure
│  │ (GPU)    │ │ (MPI)    │ │ (timestepping)   │ │
│  ├──────────┤ ├──────────┤ ├──────────────────┤ │
│  │ p_mgn    │ │ p_bio    │ │ p_tre (oct-tree) │ │  ← Mesh & I/O
│  │ (grid)   │ │ (snapshots)│                  │ │
│  ├──────────┤ └──────────┘ └──────────────────┘ │
│  │ mods[]   │                                    │  ← Physics container
│  │ (hydro, chemistry, multigrid, particle)      │
│  └──────────┘                                    │
└─────────────────────────────────────────────────┘
```

The mesh knows **only** about geometry — it has no awareness of which physics modules are
enrolled. Modules extract geometry from the mesh unidirectionally. This separation is the
foundation of Kratos's flexibility (Wang 2025a, §2.4).

### 1.3 Infrastructure Layers

| Layer | Class | Your Role |
|-------|-------|-----------|
| **Device** | `device::base_t` → `device_t` (CUDA/HIP/MUSA) | Enroll via `mesh.enroll_device<device_t>()` |
| **Communication** | `comm::base_t` → `comm::mpi_t` | Enroll via `mesh.enroll_comm<comm::mpi_t>()` (MPI builds only) |
| **Snapshot I/O** | `binary_io::base_t` → `binary_io::mpi_t` | Enroll via `mesh.enroll_binary_io<binary_io::mpi_t>()` |
| **Time cycle** | `cycle::base_t` | Enroll via `mesh.enroll_cycle<MyCycle>()`; override for custom output |
| **Mesh generator** | `meshgen::base_t` → `cartesian_t`, `spherical_t`, `cylindrical_t` | Enroll via `mesh.enroll_mgen<MyMgen>()` |
| **Physics modules** | `mod_base_t` → your subclasses | Enroll via `mesh.enroll_module<MyModule>(init_order, step_order)` |

### 1.4 Configuration File Format

INI-like format. Sections are `[section]` headers, followed by `key = value` lines:

```ini
[mesh]
x_min         = -1.0
x_max         =  1.0
n_cell_global =  128

[dynamics]
gamma         = 1.6667

[cycle]
dt_output     = 0.01
t_lim         = 1.0

[prob]
rho0          = 1.0
prob|pre0         1.0
```

Internally, the `input` class stores entries as `section|key` → value pairs, so
programmatic access uses `args.get<T>("section", "key")`. Lines starting with `#` are
comments.

Usage: `bin/kratos my_problem.par [override.par ...]`

Later files override earlier ones. In MPI mode, only root reads; all parameters are broadcast.

### 1.5 Index Conventions

Hydrodynamic variable indices are always qualified with `idx::` to distinguish them
from arbitrary array offsets:

```cpp
u[ idx::rho ]       // mass density
u[ idx::ene ]       // total energy density
u[ idx::mom + a ]   // momentum density (a = 0,1,2)
```

This convention is used throughout Kratos and is expected in all problem generators.

---

## 2. Design Ideas: Modules and Interactions

### 2.1 The Module Lifecycle

Every physics module inherits from `mesh::mod_base_t` and follows this contract:

```
read()    →  Called first. Parse your parameters from the input file.
              Register block data factories.
              Call your enroll<...>() method.

init()    →  Called after all reads complete. Allocate memory.
              Configure sub-modules.
              Access geometry from the now-built mesh.

step()    →  Called every cycle. Implement your physics.
              Iterate local blocks, launch kernels, exchange boundaries.

finalize() →  Called at teardown. Free resources, print diagnostics.
```

### 2.2 Enrollment: The Compile-Time Composition Pattern

Modules do not use runtime virtual dispatch for physics sub-components. Instead, they use
template-based enrollment with CRTP (Curiously Recursive Template Pattern). The rationale
is documented in Wang (2025a, §2.4): virtual function tables on GPUs cause _"significant
performance degradation by ~10² times"_ due to global memory latency.

Each module exposes an `enroll<>()` template that wires sub-modules together:

```cpp
// Hydro module example:
template< class prx_T, class itg_T, class eos_T,
          class rec_T, class rie_T, class com_T >
void enroll();
```

At compile time, the template parameters select concrete implementations (e.g., RK2 integrator,
adiabatic EOS, PLM reconstructor, HLLC Riemann solver), and a lambda-based integration
pipeline is assembled: `cons2prim → act_bnd → calc_flx → act_flx → intg_flx`.

### 2.3 Block Data and Proxies

Each module stores per-block physics data in a `block_data_t` inheriting from
`mesh::block::base_data_t`. Access follows a three-layer model:

```
block_t                → Metadata: id, rank, region, geometry
base_data_t (h/d)      → Physics fields: dat_3d_t<T> arrays
dual_t                 → Wraps block_t + host copy + device copy + stream/event
```

A **proxy struct** provides typed access:

```cpp
struct my_proxy_t : mesh::block::proxy_t< my_block_data_t > {
    using bdt_t = my_block_data_t;
    static my_block_data_t & ref( base_data_t & bd ) {
        return dynamic_cast< my_block_data_t & >( bd );
    }
};
```

### 2.4 Cross-Module Data Sharing

When Module B needs data from Module A (e.g., chemistry reads hydro density), Kratos uses
a **shallow-copy proxy** pattern (Wang 2025a, §2.4, Figure 2):

1. Module A stores a `weak_ptr` to itself in a static proxy member (`prx_base_t::p_hyd`)
2. Module B's proxy struct includes both modules' block data types
3. Before each GPU kernel launch, the proxy constructs a temporary `bdt_T` that
   **shallow-copies pointers** from Module A's data (no GPU copies)
4. The kernel sees a unified data structure

```cpp
// Simplified proxy for hydro + multigrid coupling:
struct prx_hyd_t : prx_base_t, mesh::block::proxy_t< hydro::block_data_t > {
    struct bdt_hyd_t : hydro::block_data_t {
        multigrid::data::base_t mgd;  // shallow-copied from MG module
    };
    static auto d( const bdt_T & bdt ) {
        return bdt_hyd_t( bdt );  // constructor shallow-copies mgd
    }
};
```

### 2.5 Parasite Pattern

Modules that operate on another module's data use the **parasite pattern**:

```cpp
auto p = mesh.enroll_module< hydro_t >( 0, 0 );
auto q = mesh.enroll_module< chem_t >( 1, 2 );
q->parasite( p );  // chemistry depends on hydro data
```

The parasite stores a `weak_ptr` to the host. It accesses host data via the proxy's static
accessors, which `dynamic_cast` to the concrete type through the weak pointer.

### 2.6 Module Execution Order

`mesh.enroll_module<T>(init_order, step_order)` controls the initialization and per-cycle
execution sequence. Common ordering:

| Order | Module | Rationale |
|-------|--------|-----------|
| 0 | Hydro | Primary physics |
| 1 | Multigrid | Must run after hydro (needs density for source), before hydro step applies gravity |
| 2 | Chemistry | After hydro (parasite, updates species) |
| 3 | Particles/Radiation | After chemistry (photon sources depend on species) |

---

## 3. Composing a Problem Generator

### 3.1 Minimal Example: 92-line Shock Tube

The simplest possible problem is `usr_ext/std_tst/sod/usr.cpp` (92 lines). It:

1. Subclasses `dynamics::hydro_t`
2. Overrides `read()` to parse `[prob]` parameters and call `enroll<>()`
3. Overrides `init_cond()` to set the left/right Riemann states
4. Defines `prob::run()` to enroll device + comm + module, init, evolve

This is the reference for all hydro-only problems. See it at `usr_ext/std_tst/sod/usr.cpp`.

### 3.2 Example A: `usr_ext/kh_gas` — Intermediate Complexity

**Physics:** Kelvin-Helmholtz instability with self-gravity and radiative cooling.

**What it demonstrates:**
- Custom integrator (`stp_t`) with gravity and cooling source terms
- Cross-module proxy: hydro accesses multigrid potential
- Custom multigrid module with conditional enable/disable
- Unit system for CGS<->code conversion

#### Structure

```
usr_ext/kh_gas/
  usr.cpp          # All classes in one file (514 lines)
```

#### Key design decisions

**Custom integrator (`stp_t<>`):** The CRTP-derived integrator overrides `add_flux()` to inject
gravitational acceleration and cooling energy loss into the hydro update. The gravity potential
comes from the multigrid module via the cross-module proxy (`prx_hyd_t`).

```cpp
// In add_flux(): compute gravity from MG potential
auto d_fld = b_d.mgd.fld( i );     // MG potential from proxy
...
acc[ a ] = -d_fld / d_x;            // central difference gradient
u[ idx::mom + a ] += acc[ a ] * dt_rho;  // momentum update

// Apply cooling
self.cool_e( eg, rho, rho_div, b_d, f_eos );
```

**Proxy (`prx_hyd_t`):** Extends both `prx_base_t` (cross-module weak_ptr) and
`mesh::block::proxy_t<hydro::block_data_t>`. Defines a composite `bdt_hyd_t` that
shallow-copies the `mgd` (multigrid data) member alongside hydro fields.

**Multigrid (`mg_t`):** Conditionally enabled via `grav_conv` parameter. If gravity
is disabled (`grav_conv <= 0`), `step()` is a no-op — the simulation runs pure hydro.

**Cooling (`cool_t`):** Implements a backward-Euler solver for optically-thin radiative
cooling with escape probability correction. Cooling data is loaded from a file; if loading
fails, cooling silently disables.

**How `prob::run()` wires it:**

```cpp
mesh.enroll_device< device_t >();
mesh.enroll_comm  < comm::mpi_t >();
mesh.enroll_binary_io< binary_io::mpi_t >();
mesh.enroll_module< hydro_t >();    // enrolled first
mesh.enroll_module< mg_t >();       // enrolled second; cross-linked via proxy
```

#### What to learn from this example

1. **Custom integrator via CRTP:** inherit from `rk2_t<derived_T>`, override `add_flux()`.
   The device-side kernel recovers your concrete type via `get_self()`.

2. **Cross-module data:** define a composite proxy struct. The `d()` method constructs a
   temporary block data that shallow-copies fields from multiple modules. The kernel sees
   a unified view.

3. **Unit conversion:** use `phys::unit_t` to convert between code units and CGS.
   The integrator reads parameters in physical units and converts.

4. **Optional physics:** if a parameter is zero/negative, skip the computation in `step()`.

---

### 3.3 Example B: `usr_ext/bin_bh` — Compact with Custom Physics

**Physics:** Binary star system with sink particles and stellar outflows, orbiting in a
shearing box. Self-gravity from the binary is computed analytically (not via multigrid).

**What it demonstrates:**
- Custom cycle with per-timestep hooks and statistics output
- Analytic gravity source (point masses on circular orbits)
- Sink particle treatment (mass removal near stars)
- Stellar outflow boundary condition
- Shortest `prob::run()` in the codebase (38 lines)

#### Structure

```
usr_ext/bin_bh/
  bin_bh_test.par     # Parameter file
  usr.cpp              # Minimal orchestrator (38 lines)
  hydro/
    cycle.h            # Custom cycle with gravity stats
    hydro.h            # Custom hydro module
    intg.h             # Integrator with analytic gravity
    grav_src.h         # star_t and bin_star_t classes
    boundary.h         # Inflow boundary condition
```

#### Key design decisions

**Separation of concerns in subdirectories:** The 38-line `usr.cpp` only enrolls modules.
All physics logic is in `hydro/` subdirectories — a clean pattern for complex projects.

**Custom cycle (`cycle_t`):** The cycle exposes `std::function` hooks (`f_pre`, `f_acc`,
`f_out`) that the integrator wires up during `init()`. This is a callback-based approach:
the cycle owns the output logic, the integrator provides the data.

```cpp
// In stp_t::init():
cyc.f_pre = [ & ] ( const bool & stat_grav ) { this->stat_grav = stat_grav; };
cyc.f_acc = [ & ] ( ) {
    mod.p_dev->cp( p_grav_h, p_grav, 6 );    // D→H copy
    mod.p_mesh->p_com->reduce_all( p_grav_h, comm::sum, 6 );  // MPI reduce
    // Print or store gravitational acceleration
};
```

**Analytic gravity in the integrator:** `bin_star_t` contains two `star_t` objects, each
with position updated via `φ = ω·t`. The `f_grav()` method computes `1/r²` acceleration
with a Plummer softening length. Particles inside `r_sink` have their momentum and energy
halved (sink treatment). Outflow stars set the cell state to a prescribed wind profile.

**Custom boundary condition (`inflow_t`):** Inherits from
`dynamics::hydro::boundary::phys_base_t<prx_T, inflow_t<prx_T>>` (CRTP). The `hydro()`
device method sets ρ, p, v in ghost cells.

```cpp
template< class d1d_T, class bdt_T >
__forceinline__ __device__ void hydro
( d1d_T & d, int i_3d[], bdt_T & b_d ) const {
    for( int i = -d.n_spa/2; i < 0; ++i ) {
        d( i, idx::rho ) = rho;
        d( i, idx::ene ) = pre;
        for( int a = 0; a < 3; ++a )
            d( i, idx::mom + a ) = vel[a];
    }
}
```

**How `prob::run()` wires it:**

```cpp
mesh.enroll_device< device_t >();
mesh.enroll_comm  < comm::mpi_t >();
mesh.enroll_binary_io< binary_io::mpi_t >();
mesh.enroll_cycle     < friction::cycle_t >();    // custom cycle
mesh.enroll_module    < friction::hydro_t >();    // custom hydro (enrolls the rest)
```

#### What to learn from this example

1. **Custom cycle with callback hooks:** expose `std::function` members, wire them during
   `init()`. Keeps cycle logic separated from integrator logic.

2. **Analytic gravity sources:** compute gravity directly on the GPU in `add_flux()`.
   No multigrid enrollment needed — simpler and faster for point masses.

3. **Sink particles:** modify cell state inside `add_flux()` when a cell center lies within
   `r_sink_2`. The integrator has full access to `u[]`.

4. **Modular file layout:** physics classes in `hydro/`, thin orchestrator in `usr.cpp`.
   This pattern scales to dozens of files cleanly.

5. **`#warning` for TODO markers:** `#warning "TODO: Full 3D selection of the omega
   direction"` — the code compiles but the comment flags incomplete work.

---

### 3.4 Example C: `usr_ext/planet_atm_sph` — Complex Multi-Module

**Physics:** Exoplanetary atmosphere in spherical-polar coordinates with:
- 3D hydrodynamics with planetary + stellar gravity and rotation
- Non-equilibrium thermochemistry with radiative cooling
- Ray-tracing ionizing radiation from the host star
- Spherical boundary conditions (inflow wind, outflow, polar)

**What it demonstrates:**
- Four cooperating modules (hydro, chemistry, radiation, custom cycle)
- Parasite pattern linking modules together
- Custom EOS with non-ideal heat capacity
- Spherical mesh generator, dedicated spherical integrator
- Pole rebinning (coarsening cells near θ=0,π)
- Full extension stack: `usr/extension/radiation/`, `usr/extension/hydro/`, `usr/extension/chem_therm/`

#### Structure

```
usr_ext/planet_atm_sph/
  usr.cpp            # Orchestrator: enrolls 4 modules + cycle + spherical meshgen (63 lines)
  cycle.h            # Custom cycle: hydro_off mode, error aggregation
  hydro.h            # Hydro with chemical species, spherical initial conditions
  intg.h             # Spherical integrator with stellar+planetary gravity, Coriolis, damping
  eos.h              # EOS with non-ideal cv(T), spherical dt calculation, pole rebinning
  chemistry.h        # Chemistry module: conditionally active after t_micro_start
  radiation.h        # Radiation module with pole rebinning, photon generation
  boundary.h         # Four custom boundary conditions (sph_out, sph_ref, sph_pol, inflow)
  pole_rebin.h       # Pole rebinning utility (coarsens cells near poles)
```

#### Key design decisions

**Module cross-linking in `prob::run()`:**

```cpp
auto p = mesh.enroll_module< hydro_t >( 0, 0 );      // hydro first
auto r = mesh.enroll_module< radiation_t >( 2, 1 );    // radiation second
auto q = mesh.enroll_module< chem_t >( 1, 2 );         // chemistry third

q->parasite( p );       // chemistry reads hydro data
r->p_che   = q;         // radiation reads chemistry data
r->p_hyd   = p;         // radiation reads hydro data
q->p_rad   = r;         // chemistry reads radiation data
```

This creates a **fully connected graph**: hydro ↔ chemistry ↔ radiation. The `p_che`,
`p_hyd`, `p_rad` members are raw `shared_ptr`s while `q_mod` (the parasite's host link)
is a `weak_ptr`.

**Custom EOS (`eos_t`):** Uses `chemistry::hydro::cv_poly_list_t` for non-ideal
heat capacity (polynomial in ln T). The `dt()` method computes the CFL timestep in
spherical coordinates, accounting for the pole rebinning factor.

```cpp
auto dx = b_d.geo.dx_f( a, i )
    * ( a >= 1 ? x[0] : 1 )          // radius scaling for angular directions
    * ( a == 2 ? sin(x[1]) : 1 )     // sin(θ) for φ direction
    * ( a == 2 ? rebin.num_rebin(b_d.geo, i) : 1 );  // pole rebinning
```

**Spherical integrator (`stp_t`):** Extends `dynamics::hydro::integrator::sph_t<>` with:

1. **Planetary gravity:** `-GM_p / r²` radial acceleration
2. **Stellar gravity in the rotating frame:** point-mass acceleration from a
   companion star on a fixed circular orbit, plus centrifugal and Coriolis terms
3. **Velocity damping:** all velocity components multiplied by `f_damp < 1` before
   `t_micro_start` (allows hydrostatic relaxation before turning on microphysics)

The velocity transformation between spherical `(r,θ,φ)` and Cartesian `(x,y,z)` is
done on-device for every cell.

**Conditional microphysics:** Both `chem_t` and `radiation_t` check `t > t_start`:

```cpp
// In chem_t::step():
if( mesh.p_cyc->t > t_start )
    extension::chem_t::step( mesh );
// else: no-op. Pure hydro until microphysics activates.
```

This allows the simulation to first reach hydrostatic equilibrium before turning on
chemistry and radiation.

**Custom cycle (`cycle_t`):** Supports `hydro_off` mode (chemistry-only testing) by
skipping the CFL-limited dt and using a fixed timestep. Also implements cross-rank
error aggregation: if any rank throws an exception during `step()`, the MPI reduce
propagates the error and the simulation stops cleanly.

**Custom boundary conditions (4 types):**
- `sph_out_t`: outflow with power-law extrapolation for density/temperature
- `sph_ref_t`: reflecting boundary in spherical coordinates
- `sph_pol_t`: special treatment for polar axis (θ=0,π)
- `inflow_t`: stellar wind injection with species-dependent composition

**Pole rebinning:** The `rebin_t` utility merges fine cells near θ=0 and θ=π to avoid
extremely small azimuthal cell sizes that would force a tiny CFL timestep. Applied to
both hydro fields and radiation fields.

#### What to learn from this example

1. **Multi-module coupling:** use `parasite()` for weak ownership, raw `shared_ptr`
   members for strong ownership when modules truly require each other.

2. **Extension stack:** build on `usr/extension/hydro/`, `usr/extension/chem_therm/`,
   `usr/extension/radiation/` — they provide the proxy infrastructure and base
   implementations that your problem customizes.

3. **Spherical geometry:** requires a spherical meshgen (`mesh::meshgen::sph_t`),
   a spherical integrator (`dynamics::hydro::integrator::sph_t`), and careful
   handling of geometric source terms.

4. **Conditional physics:** use `t_micro_start` to delay expensive microphysics
   until the hydro has relaxed. Check `t > t_start` in `step()`.

5. **Custom EOS:** inherit from `chemistry::hydro::eos_t<cv_model, derived_t>` (CRTP).
   The non-ideal heat capacity is handled by the `cv_poly_list_t` model; the EOS
   override handles spherical dt calculation and pole rebinning.

6. **Pole rebinning:** a registry pattern where utilities apply to multiple modules
   via a `get_data_fld()` template — the rebinner queries the block data for a
   field reference, coarsens, and updates.

---

### 3.5 Example D: `usr_ext/cosmic` — Custom Mesh and Scaling

**Physics:** Cosmological simulation with:
- Comoving coordinates with scale factor evolution
- Dark matter particles (CIC deposition)
- Self-gravity (multigrid Poisson solver)
- Hydrodynamics with cosmic expansion source terms
- Custom binary I/O format

**What it demonstrates:**
- Subclassing `mesh_t` itself (`cosmic::mesh_t`)
- Custom binary I/O class
- Cosmological scaling factors via a global singleton
- Three-module enrollment with non-standard ordering
- Custom mesh initialization

#### Structure

```
usr_ext/cosmic/
  usr.cpp              # Orchestrator: enrolls custom mesh, 3 modules, custom cycle
  mesh.h / mesh.cpp    # cosmic::mesh_t extends mesh::mesh_t
  binary_io.h          # cosmic::binary_io_t extends binary_io::base_t
  indices.h            # Extended field indices
  cosmology/
    scaling.h          # Global singleton for scale factor a(t), Hubble, etc.
    cycle.h            # Custom cycle for cosmic time
  hydro/               # Hydro with comoving source terms
  multigrid/           # Multigrid Poisson in comoving coordinates
  particle/            # Dark matter particles
  io/                  # Additional I/O utilities
  types/               # Cosmology-specific types
```

#### Key design decisions

**Custom mesh (`cosmic::mesh_t`):** Extends `mesh::mesh_t` — the only example in the
codebase that subclasses the central orchestrator. This provides hooks for cosmological
initialization that need to happen at the mesh level.

**Global scaling singleton (`cosmic::scaling`):** A global interface accessed via
`cosmic::scaling::interface().read(args)`. Provides `a(t)`, `da/dt`, `H(t)`, and
conversion between comoving and physical coordinates. This is read before `mesh.init()`
so that all modules can query scale factors during their own `init()`.

**Non-standard module ordering:**

```cpp
mesh.enroll_module< hydro_t >( 1, 0 );     // Hydro first in step order
mesh.enroll_module< cic_t   >( 2, 1 );     // Particles second
mesh.enroll_module< mg_t    >( 0, 2 );     // Gravity third in step order
```

Note `mg_t` has `step_order=2` but `init_order=0` — it initializes first (to set up the
multigrid hierarchy) but steps last (after particles have deposited density, which must
appear as source for gravity at the *next* step).

**Custom binary I/O:** `cosmic::binary_io_t` extends `binary_io::base_t` — stores
cosmological metadata alongside snapshot data.

#### What to learn from this example

1. **Custom mesh subclass:** when initialization requires non-standard control flow,
   subclass `mesh_t`. This is rare — most problems use the default `mesh_t`.

2. **Global singleton for universal parameters:** `cosmic::scaling` provides `a(t)`,
   `H(t)`, etc., to all modules without explicit parameter passing. Modules call
   `cosmic::scaling::interface().get_a(t)` on-device via `__constant__` memory.

3. **Module ordering matters:** `init_order` and `step_order` can differ. MG
   initializes first but steps last, because particle density is needed for the
   gravity source.

4. **Custom I/O:** extend `binary_io::base_t` when snapshots need extra metadata
   (e.g., scale factor, redshift).

---

## 4. Module Customization Reference

### 4.1 Which class to override for what

| You want to... | Inherit from | Override |
|---|---|---|
| Add a new physics module | `mesh::mod_base_t` | `read()`, `init()`, `step()`, `finalize()`, `init_cond()` |
| Customize the EOS | `dynamics::eos::base_t` (CRTP) | `c2p_kernel()` |
| Change integrator source terms | `rk2_t<>` or `sph_t<>` (CRTP) | `add_flux()` |
| New Riemann solver | `dynamics::riemann::base_t` (CRTP) | flux computation |
| New reconstructor | `dynamics::reconstruct::base_t` (CRTP) | face state reconstruction |
| Custom boundary condition | `dynamics::hydro::boundary::phys_base_t` (CRTP) | `hydro()` GPU kernel |
| Change timestep/output behavior | `cycle::base_t` | `init()`, `step()`, `redc_dt_start/finish()` |
| Non-Cartesian geometry | `meshgen::base_t` → `spherical_t`, `cylindrical_t` | `set_coord_axes()`, `volume()`, `surface()` |
| Cross-module data access | `extension::prx_base_t` + your proxy | `d()`, `set_module()` |

### 4.2 Complexity tiers

| Tier | Scope | Example | Lines of code |
|------|-------|---------|---------------|
| **Simple** | 1 module, no cross-module | `std_tst/sod` | ~90 |
| **Intermediate** | 2 modules, 1 cross-module | `kh_gas` | ~510 |
| **Compact advanced** | 1 module but complex physics | `bin_bh` | ~380 total, 38 in usr.cpp |
| **Full multi-module** | 4+ modules, full cross-linking | `planet_atm_sph` | ~1,000+ across 9 files |
| **Framework-level** | Custom mesh, I/O, global state | `cosmic` | ~500+ across 10+ files |

---

## 5. Build and Run

The Makefile uses the `USRDIR` variable to select which problem generator (pgen) to
compile. It defaults to `usr/` (which contains only the weak-symbol default stub),
so you **must** specify the pgen directory:

```bash
# Build with a specific pgen
make USRDIR=usr_ext/std_tst/sod

# Full build with custom options
make USRDIR=usr_ext/bin_bh DEBUG=0 ARCH=CUDA MPI=1 PRECISION=1

# Build with the new pgen cloned from bin_bh
make USRDIR=usr_ext/bin_bh_tst

# Run
bin/kratos usr_ext/bin_bh/bin_bh_test.par [override1.par ...]
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `USRDIR` | `usr` | Problem generator directory (contains the `prob::run()` override) |
| `ARCH` | `CUDA` | GPU backend: `CUDA`, `HIP`, `HIPCPU`, `MUSA` |
| `DEBUG` | `0` | Debug mode (`1` = `-g -O0`, force-includes `debug_macros.h`) |
| `MPI` | `0` | MPI support (`1` = links `-lmpi`) |
| `PRECISION` | `1` | Floating-point precision (`0`=all float, `1`=mixed, `2`=all double) |

Sources are auto-discovered from the `USRDIR` path recursively. Only one pgen can be
compiled at a time — the binary always produces `bin/kratos`. See `AGENTS.md` for
further build details.

---

## 6. Config Reference

### 6.1 Runtime Parameters (New: June 2026)

| Parameter | Section | Default | Added | Purpose |
|---|---|---|---|---|
| `reac_on_const` | `[chemistry]` | `true` | 2026-06 | Place reaction lists in constant vs global device memory |
| `pole_rebin_threadlim` | `[mesh]` | `32` | 2026-06 | Max GPU threads per block for pole rebinning kernel |

### 6.2 Flexible Memory Placement (`reac_on_const`)

The `const_list_t` template (`src/utilities/types/var_list.h`) now supports dual memory
placement via the `on_const` flag:

- **Constant memory** (`on_const=true`, default): uses `f_malloc_const` + `f_cc` copy.
  GPU constant memory offers low-latency cached reads but is limited to ~64 KB total
  (across all modules). Suitable for small to medium reaction databases.

- **Regular device memory** (`on_const=false`): uses `f_malloc` + `f_cp` copy. Unlimited
  capacity at the cost of global memory latency. Required when reaction databases
  (gas-phase + dust + photon + cooling sub-lists) exceed the constant budget.

The flag is propagated from the `[chemistry]` `reac_on_const` parameter through the
reaction-list hierarchy: `reaction::list::std_t` → `chemistry::reaction::list_t`
(usr/extension) → each sub-list (`mol_cool`, `atom_cool`, `phot_reac`, `dust_reac`).
All sub-lists share the same memory policy — you cannot mix constant and global within
one simulation.

A `free()` method on `const_list_t` and `reaction::list::std_t` cleans up device memory
when `on_const=false` (constant memory does not need explicit freeing).

### 6.3 Pole Rebin Threading (`pole_rebin_threadlim`)

The pole rebinning kernel (`src/mesh/geometry/pole_rebin.h`) now uses a 2D block grid:

- **Grid:** `n_bl_x × n_bl_y` blocks, where `n_bl_x = ceil(n_cell_x / n_th)` —
  multiple blocks share the θ-direction work.
- **Threads per block:** `min(pole_rebin_threadlim, n_cell_x)` — capped for GPU occupancy.
- **Default:** 32 threads/block. Adjust downwards if pole rebinning occupies too many
  GPU resources, or upwards (up to ~256) for very wide radial meshes.

Previously the kernel used a 1D grid with one block per θ-cell, which could under-utilize
the GPU for large polar cell counts.

### 6.4 Product Species Deduplication

Reaction product arrays (`i_chem` / `f_chem`) are now deduplicated during `setup_pars()`
(`src/modules/chemistry/reaction/reaction.cpp`). After stoichiometric reduction, duplicate
product entries (same species appearing multiple times) are merged with summed `f_chem`
coefficients. Reactant multiplicities are preserved because kinetic rate coefficients
depend on them. This eliminates redundant computation in the chemistry ODE solver.

---

## 7. Writing Reaction Databases

### 7.1 Rate Coefficient Conversion (CGS → INI)

**Reference:** `docs/AGENTS.md` §Rate Coefficient Conversion, `docs/Chemistry.tm` §1.1

Kratos uses three abundance representations: CGS number density `n_i` (databases),
reduced number density `ỹ_i = n_i·m_A/ρ₀` (hydro advection), and molar abundance
`y_i = n_i·m_A/ρ` (chemistry solver). The hydro solver advects `ỹ_i`; the chemistry
solver converts to `y_i` by dividing by `ρ̃ = ρ/ρ₀`.

Reaction rate coefficients in the INI file must be pre-converted from number-density
CGS units to Kratos molar-abundance units. The conversion depends on the number of
reactants R:

```
Ã = A · N_A^(R−1)   (CGS)
```

Where N_A ≈ 6.022e23 in CGS.

| R | Type | Conversion | Database A units → INI Ã |
|:---:|---|---|---|
| 1 | Unimolecular | Ã = A | s⁻¹ → s⁻¹ |
| 2 | Bimolecular | Ã = A·N_A | cm³/s → cm³/s · N_A |
| 3 | Termolecular | Ã = A·N_A² | cm⁶/s → cm⁶/s · N_A² |

Kratos multiplies the rate expression by ρ^(R−1) automatically in `rate()`.

**Critical rule:** Unimolecular (R=1) reactions require **no** N_A factor. Multiplying a
dissociation prefactor by N_A inflates the rate by ~6e23, causing solver instability.

### 7.2 Supported Rate Types

| `rate_type` | R | Params | Notes |
|---|---|---|---|
| `ARR` / `std` / `stand` | any | `A b C` | Modified Arrhenius: k = A·(T/300)^b·exp(−C/T) |
| `intp` / `inter` | any | `T₁ k₁ T₂ k₂ ...` | Interpolation: k = interp(T, pairs) |
| `TROE` | any | `A₀ b₀ C₀ A∞ b∞ C∞ α T₁ T₂ T₃ μ̄` | Pressure-dependent Troe falloff (11 params) |

### 7.3 Troe Falloff Details

For pressure-dependent reactions, Kratos computes k(T, n_tot) using the Troe formalism.
The 11 parameters are interpreted as:

- Low-pressure limit: k₀ = A₀·(T/300)^b₀·exp(−C₀/T)
- High-pressure limit: k∞ = A∞·(T/300)^b∞·exp(−C∞/T)
- Reduced pressure: P_r = k₀·n_tot / k∞
- Broadening: F_cent = (1−α)exp(−T/T₃) + α·exp(−T/T₁) + exp(−T₂/T)
- Effective rate: k = k∞ · (P_r/(1+P_r)) · F

where n_tot = ρ/(μ̄·m_A) and m_A = 1.67262e-24 g. At the pure low-pressure limit
(A∞ ≈ 0), k_eff = k₀·n_tot (linear density scaling). At the pure high-pressure limit
(k₀ ≈ 0), k_eff = k∞ (density-independent). α ∈ [0,1]; α=0 with large T_i gives the
Lindemann limit (F=1).

## 8. Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `Default run function is empty!` | `USRDIR` not set or points to a directory without a `prob::run()` override. Pass `USRDIR=usr_ext/your_pgen` to `make`. |
| Segfault in `act_bnd` | `p_bdk` is null — you forgot `enroll_comm<>()` or `enroll_bdk<>()` in your module |
| Parameter not found | Typo in section name or key. Check `section\|key` spelling. Use `get(key, default)` for optional params |
| MPI deadlock | Two modules using different MPI tags for the same direction. Check communication boundary setup |
| `q_che unbound` | Chemistry module enrolled but `parasite()` never called — or hydro module was destroyed |
| NaN in conserved variables | Check `d_floor` and `t_floor` in EOS. Increase floors, decrease CFL, or enable `except_ferr` |
| build error: `usr_ext/ not found` | `usr_ext/` is a git submodule. Run `git submodule update --init` |
