# AGENTS.md

The contributor contract for Kratos: how to build, where code goes, and the conventions a change
must follow.  `README.md` covers what the project is and how to run it.

## Build and run

```bash
make USRDIR=<pgen-dir> ARCH=CUDA DEBUG=0 MPI=0 PRECISION=1
bin/kratos problem.par [override.par ...]
make clean     # remove bin/ and obj/
make remake    # clean, then build
```

GNU Make is the only build system and every flag lives in `Makefile`.  Sources are discovered with
`find` under `src/`, `$(USRDIR)` and `$(EXTDIR)` (`usr/extension/` by default), so there is no
source list to edit.  The variable meanings are in `README.md`.

## The one-pgen rule

`USRDIR` names the problem generator compiled into the binary, and only one is compiled at a time.
The binary is always `bin/kratos`.  The default `usr` supplies no `prob::run` override, so the
resulting program prints `Default run function is empty!` and exits; always pass `USRDIR`.

## Where code lives

| path | role |
|---|---|
| `src/` | Core library, built for every problem generator. |
| `src/mesh/` | Blocks, mesh, cycle, boundaries, module base (`mesh::mesh_t`). |
| `src/device/` | Device backends and the kernel launch API. |
| `src/io/` | The `input` parameter class and binary snapshot I/O. |
| `src/modules/` | Physics: `dynamics/` (hydro, MHD, non-ideal MHD), `chemistry/`, `multigrid/`, `particle/`. |
| `usr/extension/` | Reusable physics and proxy code, built into every binary. |
| `usr/intermediate/` | Header-only building blocks shared by problem generators. |
| `usr/radiation/` | Header-only radiation modules (gray flux-limited diffusion, M1 closure). |
| `visual/` | Python post-processing. |
| `data/` | Data files. |
| `docs/` | User guide, style guide, module pages, method papers. |

Do not edit `src/` to set up one problem.  Problem-specific classes belong to their problem
generator; only generic, reusable code belongs in the trunk.

## Parameter files

INI-like: `[section]` headers, then `key = value` lines; `#` starts a comment.  The `input` class
(`src/io/args/input.h`) stores each entry as `section|key`, and code reads it with
`args.get<T>( "section", "key", default )`.  A key without a default is a required key.  Extra files
on the command line are read after the first and override it.  Read your keys in `read()`, not in
`step()`.

## Device dispatch convention (no runtime type resolution on the device)

Resolve a policy type on the host, once, and instantiate one kernel per type.  A `__global__` kernel
takes its policy objects as parameters, each a compile-time template argument; the launch function
lives in a CRTP base that passes the derived object into the kernel template; the runtime choice, if
any, is a host-side virtual or factory.  A policy object is copied by value into the kernel, so keep
it trivially copyable: scalars only, and every pointer and buffer in a POD view struct.

Good: the kernel takes its policies, and the host passes the derived object in:

```cpp
template< class dat_T, class bdt_T, class rec_T, class rie_T,
          class eos_T, class itg_T >
__global__ void flux_kernel( ..., const rec_T f_rec, const rie_T f_rie,
                             const eos_T f_eos, const itg_T f_itg );

// in the CRTP base, with get_self() the derived algorithm:
dev.launch( flux_kernel< dat_t >, ..., f_rec, f_rie, f_eos, f_itg );
```

Bad: the kernel branches on a runtime tag, so every branch is compiled and no policy inlines.

```cpp
__global__ void flux_kernel( ..., const int riemann_type )
{
    if     ( riemann_type == 0 ) hllc( ... );
    else if( riemann_type == 1 ) hlld( ... );
}
```

The canonical files to copy from are `src/modules/dynamics/prototypes/integrator.h` for the kernel
and `src/modules/dynamics/hydro/integrator/integrator.h` for the `calc_flx` launch.

## Code style

`docs/STYLE_GUIDE.md` is authoritative: column limit, parenthesis and angle-bracket spacing,
control-flow braces, `__device__` conventions, naming, include order, and where a function body
lives.  Read it before editing and follow it exactly.

## Documentation discipline

- A new verified fact goes into the one document that owns the subject: `README.md` for the
  overview, `docs/README-USER.md` for use, a module page for a module's physics,
  `docs/STYLE_GUIDE.md` for style, this file for the contributor contract.
- The trunk carries no development record: no dates, no records of failed attempts, no measurement
  campaigns, no changelogs, no handover notes, no local paths.  That material belongs to the nested
  development-record repository.
- Comments state what the code does.  A document points to code by file and symbol instead of
  restating the implementation.
- A wrong statement is corrected where it lives, or deleted.  Do not append a correction and leave
  the superseded claim standing.

## Testing

There is no test runner and no test target in `Makefile`.  Verification programs live with the
problem generators in the `kratos_usrproj` repository and build like any problem generator, by
passing their directory as `USRDIR`.  Run the programs covering a module before claiming that a
change to it works.

## Nested repositories

`usr_ext/` and `docs/dev/` are separate git repositories with their own remotes and history.  The
parent `.gitignore` excludes both and the parent repository has no submodule entry for them:
`git status` in the trunk does not show changes inside them, and a worktree of the trunk does not
contain them.  Commit there separately; do not add them as submodules or copy their content in.

## Gotchas

- `src/utilities/debug_macros.h` is force-included with `-include` by the CUDA and HIPCPU builds;
  do not add a redundant `#include` and do not rely on it in the other builds.
- Snapshots use the suffix `.bin`, not `.dat`.
- `float_min` in `src/types.h` is a small sentinel (`1e-24f`, `1e-28f` at `PRECISION=2`), not
  `std::numeric_limits<float>::min()`.
- `input` is the only core type without the `_t` suffix.
- `mod_base_t::shared_from_this()` is deleted; use `weak_from_this()`, so that no module holds a
  strong reference to itself.
- Objects are not portable between backends: run `make clean` after switching `ARCH`.
- The debug `HIP` build compiles with `-std=c++20`; every other configuration uses C++17.
