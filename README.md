# Kratos

Kratos is a multi-physics astrophysical simulation framework for heterogeneous computing devices.
It solves compressible hydrodynamics and magnetohydrodynamics on block-structured adaptive meshes,
coupled to non-equilibrium thermochemistry, radiation transport and self-gravity.  The code is
C++17, runs on CUDA, HIP or MUSA devices, and writes kernels directly against those programming
models behind a thin launch and memory abstraction.  MPI is optional.

The physics modules:

- compressible hydrodynamics and ideal MHD;
- non-ideal MHD: Ohmic, ambipolar and Hall diffusion;
- non-equilibrium thermochemistry, cooling and heating, and reaction networks;
- radiation: Monte Carlo ray tracing, and gray flux-limited diffusion with an M1 closure;
- self-gravity through a multigrid Poisson solver;
- particles: cloud-in-cell deposition, tracers and Monte Carlo photon packets.

## Quick start

GNU Make is the only build system.  `USRDIR` selects the problem generator to compile; the binary is
always `bin/kratos` in the directory you build in.

Build outside the source tree: a build writes `obj/` and `bin/` next to the `Makefile`, and a run
writes its snapshots into the working directory, so neither belongs inside a git worktree.  Symlink
the sources into a scratch directory and build there:

```bash
REPO=/path/to/kratos
mkdir -p ~/scratch/kratos-build && cd ~/scratch/kratos-build
for d in Makefile src usr usr_example data usr_ext; do
    [ -e "$REPO/$d" ] && ln -sfn "$REPO/$d" .
done
make USRDIR=usr_example/bow_shock ARCH=CUDA DEBUG=0 MPI=0 PRECISION=2 -j8
```

| variable | values | meaning |
|---|---|---|
| `USRDIR` | a problem-generator directory | Selects the problem generator compiled in: the strong `prob::run` override.  Required for a runnable binary. |
| `ARCH` | `CUDA` (default), `HIP`, `HIPCPU`, `MUSA` | Device backend.  `HIPCPU` is a CPU implementation of the HIP runtime, intended for debugging. |
| `DEBUG` | `0` (default), `1` | Debug build; the flags and the debug macros depend on the backend (see `Makefile`). |
| `MPI` | `0` (default), `1` | Adds `-D__MPI__` and links MPI. |
| `PRECISION` | `1` (default), `0`, `2` | `0` = all single, `1` = single/double mixed, `2` = all double.  See `src/types.h`. |

Run the binary from a directory of its own, with one parameter file and optional override files
(later files win); the output lands in that directory:

```bash
mkdir -p run && cd run
../bin/kratos problem.par [override.par ...]
```

A minimal parameter file (INI-like: `[section]` headers, `key = value` lines, `#` comments):

```ini
[mesh]
x_min         =   0   -1   -1
x_max         =   1    1    1
n_cell_global = 512   2    1

[boundary]
kinds = fre fre fre fre fre fre

[cycle]
t_lim     = 2
dt_output = 1e0

[dynamics]
gamma = 1.4
```

Inside the code each entry is addressed as `section|key`, so a value is read with
`args.get<T>("mesh", "x_min")`.

## Repository layout

| path | contents |
|---|---|
| `src/` | Core library: mesh and blocks, device backends, communication, I/O, physics modules. |
| `usr/` | Trunk library code: `extension/` (reusable physics, built into every binary), `intermediate/` (header-only building blocks) and `radiation/` (header-only radiation modules). |
| `visual/` | Python post-processing: snapshot readers and plotting scripts. |
| `data/` | Data files: chemistry networks and cooling tables. |
| `docs/` | Documentation: user guide, style guide, module pages (`mhd/`, `radiation/`) and method papers. |
| `usr_example/` | A complete problem generator in the trunk (`bow_shock/`), with its own module and parameter file. |
| `usr_ext/` | Nested repository `kratos_usrproj`: the private problem generators and verification programs.  Not tracked here. |
| `docs/dev/` | Nested repository `kratos_devdocs`: the development record.  Not tracked here. |

The trunk's documentation is the user-facing part.  The project's development record lives in the
nested `docs/dev/` repository, and the trunk does not depend on it.

## The plugin model

`src/main.cpp` calls `prob::run(argc, argv)`.  The trunk provides only a weak default, which prints
an error and returns.  A problem generator supplies the strong override: it builds a `mesh::mesh_t`,
enrols the infrastructure handles and the physics modules it needs, reads the parameter files and
calls `mesh.evolve()`.  `mesh::mesh_t` holds the block and geometry structure, so one binary can
carry any subset of the physics.

## Modules

| module | adds | where |
|---|---|---|
| hydrodynamics | compressible flow, Riemann solvers, equation of state, boundary kinds | `src/modules/dynamics/hydro/` |
| ideal MHD | magnetic field, HLLD solver, constrained transport | `src/modules/dynamics/mhd/` |
| non-ideal MHD | Ohmic, ambipolar and Hall diffusion, explicit or implicit | `src/modules/dynamics/mhd/nonideal/` |
| thermochemistry | reaction networks, cooling and heating, nuclear reactions | `src/modules/chemistry/`, `usr/extension/chem_therm/` |
| self-gravity | multigrid Poisson solver | `src/modules/multigrid/` |
| particles | cloud-in-cell deposition, tracers, photon packets | `src/modules/particle/` |
| radiation | Monte Carlo ray tracing, gray flux-limited diffusion and M1 closure | `usr/extension/radiation/`, `usr/radiation/fld/` |

## Documentation map

| document | covers |
|---|---|
| `README.md` | Overview, build and run, repository layout. |
| `AGENTS.md` | Contributor contract: build rules, code layout, device dispatch, style pointer, documentation discipline. |
| `docs/README-USER.md` | User guide: architecture, module composition, parameter files, troubleshooting. |
| `docs/STYLE_GUIDE.md` | C++ typesetting and style rules. |
| `docs/mhd/nonideal.md` | The non-ideal MHD module: methods, parameters and usage. |
| `docs/radiation/fld.md` | The gray flux-limited diffusion module: method, parameters and usage. |
| `docs/methods/method_0.pdf` | The method paper. |

## Citation and license

If you use Kratos in published work, cite the method paper: Wang, L. (2025), *The Astrophysical
Journal Supplement Series*, 277, 63 (`docs/methods/method_0.pdf`).

The repository carries no LICENSE file.  No license terms are granted here; contact the authors
about reuse and redistribution.
