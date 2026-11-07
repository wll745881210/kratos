# Kratos C++ Typesetting Guide

This document is the authoritative style reference for all files under
`usr_ext/line_rt/`.  Every new file must follow it; existing files must
be brought into compliance before any unrelated edit touches them.

---

## 1. Column limit

**60 characters** per line (excluding trailing whitespace).

This is the single hardest constraint.  When a line would exceed 60
columns, break it using the rules in § 2–4.  The column count includes
indentation.

```cpp
//  OK — 59 columns
bdt.rad.mfp_i_sca_0( i ) = fabsf( ini.f_mfp_i_sca_0( x ) );

//  VIOLATION — 68 columns
bdt.rad.excitation_flux( i ) = fabsf( ini.f_excitation_flux( x ) );
```

If an expression truly cannot fit, break it into multiple statements or
extract a local variable.

---

## 2. Parentheses — spacing and line-breaking

### 2.1 Spaces inside parentheses

Always put a **space** after `(` and before `)`:

```cpp
//  ✓
if( a > 0 ) 
{ 
    ... 
}
for( int k = 0; k < n_u; ++ k )
    f( x );
g(  );
h(  );
return dev.malloc< char >( n * sizeof( float ) );

//  ✗
if (a > 0) { ... }
g();
h( );
for (int k = 0; k < n_u; ++k)
    f(x);
```

### 2.2 Empty parentheses

Use **two spaces** between the parentheses:

```cpp
//  ✓
device::rand_dev(  )
d.d(  )
order(  )
n_fld(  )
prefix(  )
set_nearest(  )
sync_all_streams(  )
empty(  )
null(  )
is_root(  )
prx_t(  )
```

### 2.3 Line-breaking — break BEFORE `(`

When a function call or constructor spans multiple lines, the opening
parenthesis goes on the **next** line, indented by one level (or aligned
with the first argument):

```cpp
//  ✓
rad.mfp_i_sca_0.init
    ( f_n, geo.n_ceff, n_gh, 1 );

return particle::proxy_base_t::setup
    ( d, m_rank, mod );

const float2_t uk = float2_t
    ( -u_max + du * float_t( k ) );

d_cdf = ( float_t * )dev.malloc_device< char >
    ( n_total * sizeof( float_t ) );

dev.f_mset
    ( rad.flx.dat, 0,
      rad.flx.n_size, d.stream );

//  ✗  (paren on same line as function name)
rad.mfp_i_sca_0.init(
    f_n, geo.n_ceff, n_gh, 1 );

return particle::proxy_base_t::setup(
    d, m_rank, mod );
```

**Exception:** single-line calls that fit within 60 columns stay on one
line:

```cpp
rad.mfp_i_sca_0.free( f_f );
dev.free_device( d_cdf );
```

### 2.4 Spaces inside brackets

Always put a **space** after `[` and before `]`:

```cpp
//  ✓
par.x[ a ] = d_ph[ 3 + a ];
par.dir[ a ] = d_ph[ a ];
f_vel[ a ].read_bin
    ( bio, "vel_" + std::to_string( a ) + "_" );

//  ✗
par.x[a] = d_ph[3 + a];
```

### 2.5 Spaces inside angle brackets

Always put a **space** after `<` and before `>`:

```cpp
//  ✓
template< class par_T >
args.get< int >( "particle", "n_scat", 0 )
interp_t< float, 3 >
std::map< int, int >
dynamic_cast< const bdt_t & >( bdt )
malloc_device< char >

//  ✗
template<class par_T>
args.get<int>( "particle", "n_scat", 0 )
interp_t<float, 3>
```

### 2.6 Function pointer / array decay

The `*` binds to the variable name, with a space on each side:

```cpp
//  ✓
float_t * row_cdf = h_cdf + j * n_u;
const auto * v_cc = par.v_cc(  );
auto * d_ph = ( float * )( data_h )

//  ✗
float_t* row_cdf = h_cdf + j * n_u;
```

---

## 3. Operators and control flow

### 3.1 Unary operators

Space before the operand:

```cpp
++ a
-- k
```

### 3.2 Binary operators

Space on both sides:

```cpp
n_total * sizeof( float_t )
( lo + hi ) >> 1
a_voigt > float_t( 1e-6 )
```

### 3.3 for / while / if

Space inside the outer parentheses (§ 2.1).  The loop body goes on the
next line at the normal indent:

```cpp
for( int k = 0; k < n_u; ++ k )
{
    cum += row_cdf[ k ];
    row_cdf[ k ] = cum;
}
```

`if` / `else` follow the same rule, and two cases are **mandatory**:

- the brace after `if` / `else` goes on **its own line** — never `if( c ) {`;
- `else` **never shares a line with the preceding closing brace** —
  both `} else {` and `} else` are wrong.

`else if` may stay on one line, but its opening brace still goes on its own
line.

```cpp
// GOOD
if( n_u > 0 )
{
    // body
}
else if( n_u < 0 )
{
    // body
}
else
{
    // body
}
```

```cpp
// BAD  ( brace on the if/else line, and else on the closing-brace line )
if( n_u > 0 ) {
    // body
} else {
    // body
}
```

### 3.4 Braces always alone; no body on the control-flow line

**Two mandatory rules** (2026-10-28, user ruling — they extend § 3.3 from
`if`/`else` to *every* block):

1. **Every brace that opens a block stands on its own line** — functions,
   `class` / `struct` / `namespace`, `if` / `else` / `for` / `while` /
   `switch` / `do`, and lambdas whose body does not fit on one line.
   Nothing follows the opening brace on that line, and nothing precedes
   the closing brace on its line.
2. **A control-flow body never sits on the same line as its header.**
   ONE statement is no exception, and neither is a body that would fit in
   60 columns.  The statement goes on the next line, indented one level.

```cpp
// GOOD
if( cfl > fac_max )
{
    cfl = fac_max;
}
for( int a = 0; a < 3; ++ a )
{
    x[ a ] = w[ a ];
}
while( n_iter < n_iter_max )
{
    ++ n_iter;
}
```

```cpp
// BAD  (body on the header line — forbidden, even for one statement)
if( cfl > fac_max ) cfl = fac_max;
for( int a = 0; a < 3; ++ a ) x[ a ] = w[ a ];
while( n_iter < n_iter_max ) ++ n_iter;
if( eta_p.on ) return;

// BAD  (opening brace on the header line)
void ni_t::read( const input & args ) {
    // ...
}
struct bdt_t {
    // ...
};
namespace edge_mg {
    // ...
}
```

**Lambdas** (2026-10-29 ruling) are exempt from the one-line ban: a lambda
definition is an *expression argument*, not a control-flow body, so it may
be written on one line with **no length threshold of its own**.  But a
lambda that needs more than one line follows rule 1 like any other block —
the opening brace goes on **its own line**, never `[ & ]( ... ) {`.

```cpp
// OK — inline
auto f = [ & ]( const f_t & a ) { return a * a; };
```

A multi-line lambda therefore looks like this (brace alone, body indented):

```cpp
auto f_bry = [ & ]( const f_t & a )
{
    const f_t b = a * a;
    return b + f_t( 1 );
};
```

---

## 4. Declarations and alignment

### 4.1 Column-aligned variable declarations

Group related declarations and align by type and name.  Multiple
closely-related variables may share a single line:

```cpp
int            n_scat;
int            n_step;
int            ph_mode;
float_t        b_sca;
```

```cpp
int      n_xo, n_xp, n_g;
float_t  xo_max,  xp_max;
float_t  dxo,    dxp, dg;
```

```cpp
auto & tgt
    = dynamic_cast< block_data_t & >( tgt_ );
```

Align `using` declarations by inserting spaces after `::` so the
names line up:

```cpp
using type::  idx_t;
using type::float_t;
using type::coord_t;
```

Align `super_t::` member declarations by inserting spaces before the
member name:

```cpp
using super_t::     x;
using super_t::     i;
using super_t::   dir;
using super_t::  ib_l;
```

### 4.2 Section separators

Two levels:

**File-level** (between major sections of a file): full-width 60 `/`:

```cpp
////////////////////////////////////////////////////////////
//  Field I/O
////////////////////////////////////////////////////////////
```

**Inline** (within a struct/class body, separating member groups): short
`//////////` with a label on each side:

```cpp
struct riia_table_t
{
    ////////// Data //////////
    float_t * dat;

    ////////// Functions //////////
    __device__ float_t lookup( ... ) const;
};
```

### 4.3 Blank lines

One blank line between functions.  No blank lines inside short
single-statement blocks.  One blank line after the opening brace of a
function body if the first statement is a declaration block.

---

## 5. `__device__` functions

### 5.1 `const` references for unmodified arguments

All non-modified arguments to `__device__` functions must be passed as
`const` references (or `const` values for scalars that fit in a
register):

```cpp
//  ✓
__device__ float_t sample_upar
( const float_t & xa ) const;

__device__ float_t _invcdf
( const float_t * cdf,
  const float_t & r ) const;

//  ✗
__device__ float_t sample_upar( float_t xa ) const;
```

This is important for GPU register pressure — passing by value forces a
register copy even when the compiler could avoid it.

### 5.2 `__forceinline__`

**Small accessors** (rule of thumb: body ≤ 15 lines, no loop over a
stencil, no `switch`) pair `__device__` with `__forceinline__`:

```cpp
__device__ __forceinline__
float_t _invcdf
( const float_t * cdf,
  const float_t & r ) const
{
    ...
}
```

**Large device functions** take plain `__device__` — do **not** add
`__forceinline__`:

```cpp
__device__ void assemble_cell
( const geo_sph_t & g, const idx_t & i,
  f2_t ( & m1 )[ 12 ][ 12 ] );
```

Why (measured, `docs/mhd/nonideal/compile-cost-analysis.md` § 3):
`__forceinline__` copies the whole callee into every caller's device code.
A 22-line kernel became **40,868 lines of PTX (1858×)** that way, and
`nonideal` alone reached 10.85 MB / 400,063 lines of PTX.  Inlining a big
function is therefore a *compile-time* decision with no runtime upside —
this tree builds with `-rdc=true`, so a genuine `__device__` out-of-line
call is available and cheap.

Rule of thumb for the boundary: if the function is big enough that you
would hesitate to paste it three times, leave it out of line.

---

## 6. Function bodies

### 6.1 `return;` and `};`

Explicit `return;` at the end of a `void` function body is permitted
and used consistently:

```cpp
__host__ void build( device::base_t & dev, float_t a )
{
    _build_usampler( dev, a );
    _build_riia_gpu( dev );
    return;
};
```

A semicolon after the closing brace of a function definition (`};`) is
permitted and used consistently.  This applies to both `__host__` and
`__device__` functions.

### 6.2 Inline member initializers

Prefer constructor initialization over inline (`= value`) initializers.
All member defaults should live in the constructor body (or
initializer list), not on the declaration line:

```cpp
//  ✓
struct intg_t
{
    bool  imaging;
    int   n_chan;
    __host__ intg_t(  )
    {
        imaging = false;
        n_chan  = 0;
        return;
    };
};

//  ✗
struct intg_t
{
    bool imaging  = false;
    int  n_chan   = 0;
};
```

Exception: simple `public:` data members in classes that lack a custom
constructor (e.g. `rad_img_t`) may use inline initializers.

For structs without a custom constructor that are initialized by a
dedicated setup function (e.g. `geo_t::setup`), set every member in
that setup function instead.  Device-side objects in particular do
NOT need default initialization: they are always populated by a
host-side copy (`cp_shlw` / `cp_deep` / proxy construction), so a
default value is dead weight — and can mask a missed copy by
silently making uninitialized data look valid.

### 6.3 Where a function body lives

A header holds interfaces and data.  A body lives in the header only
when the compiler forces it there.

**1. Device-visible functions stay in the header.**  `__device__`,
`__host__ __device__` and `__global__` functions must be visible to
every translation unit that launches or calls them, so their bodies
stay where they are declared (`__global__` additionally lives only in
`kernels/`).  This is a requirement, not a preference.

**2. A `__host__`-only function is declared in the header and defined
in the matching `.cpp`.**  "Matching" is literal: `edge_mg_cplx.h` →
`edge_mg_cplx.cpp`, `explicit_sts/nonideal.h` → `nonideal.cpp`; one
`.cpp` per header, and a header that has no host non-template body
gets none.  There is **no size exemption**: a 3-line `resource()` or
`ok()` goes to the `.cpp` like a 300-line driver.  The declaration
keeps the `__host__` prefix, the definition drops `inline`, default
arguments appear **only** in the declaration, and the `.cpp` includes
its own header first, then wraps the definitions in the header's
namespace in declaration order.

**3. A `__host__` template cannot leave the header** — every TU that
instantiates it needs the body.  Length then decides the layout:

| body length | where                                                     |
| ----------- | --------------------------------------------------------- |
| ≤ 15 lines  | at the declaration, inside the class/struct                |
| 16–120      | below the class/struct, in the same header (`inline`)      |
| > 120       | in a separate `*_impl.h`, included at the end of the header |

The 15-line threshold is the same one § 5.2 uses for `__forceinline__`
and for the same reason: below it the body is shorter than its own
signature plus the reader's trip to another file.

**Why this is a rule and not taste.**  The `edge_mg` line carried
1,277 lines of host bodies inside class bodies plus 928 lines of
namespace-scope host definitions inside headers — 19% of its 11,750
header lines (audit: `docs/mhd/nonideal/
host-code-and-member-placement-audit.md`).  A header's cost is paid by
every TU that includes it, and a body in a header is compiled once per
TU while providing nothing a declaration does not: the pgen-facing
`edge_mg_iface.h` advertised a 166-line interface but dragged a
3,088-line include closure behind it.  The rest of the tree already
works this way (278 `class::method` definitions across the 48 `.cpp`
under `src/`).

Exceptions: none.  If a host body is really needed in a header — it is
not known how to today — it belongs in the template bucket (§ 6.3.3),
not in a host-only one.

---

## 7. Naming

| Element             | Convention              | Example                          |
| ------------------- | ----------------------- | -------------------------------- |
| Classes / structs   | `snake_t`               | `block_data_t`, `gen_t`, `rad_t` |
| Template parameters | `snake_T`               | `pol_T`, `map_T`, `bdt_T`        |
| Member variables    | `snake`                 | `n_scat`, `b_sca`, `data_h`      |
| Local variables     | `snake`                 | `row_cdf`, `inv_sum`, `a_eff`    |
| Functions / methods | `snake`                 | `try_read_photon`, `sample_upar` |
| Constants           | `snake`                 | `n_total`, `a_eff`               |
| Private helpers     | `_snake`                | `_invcdf`, `_build_usampler`     |
| Namespace           | `prob`                  | `namespace prob { ... }`         |
| File names          | `snake.h` / `snake.cpp` | `photon.h`, `usr.cpp`            |
| Macros / enums      | `snake` or `UPPER`      | `read, gen_read`                 |

---

## 8. Include order

```cpp
#include <system_headers>

#include "../../src/device/general.h"
// ... other src/ headers ...
#include "../../usr/extension/algo/interp.h"

#include "line_rt headers in dependency order"
```

---

## 9. Header guards

Use `#pragma once` — no include guards.

---

## 10. Namespace style

```cpp
namespace prob
{
// ...
}
```

Opening brace on its own line.  Contents indented by 4 spaces.  Closing
brace with no trailing whitespace.

---

## 11. Comment style

- `//` for single-line comments (with a space after `//`).
- `/* ... */` for multi-line comments inside function bodies (rare).
- Full-width `////...` comment lines for file-level section separators
  (§ 4.2); short `//////////` for inline section separators.
- Comment above the line it explains, at the same indent level.
- Inline comments after code on the same line are permitted when they
  are short (typically a single phrase annotating a declaration):

```cpp
bool use_const_mem;  //  true -> const mem (ph_mode 2/3)
float_t * dat;       //  R_IIA table grid
int n_vu = 5000;      //  1-D log Voigt table size
```

- Keep comments as short as possible.  If a variable's comment is long
  and unavoidable, move the comment to the NEXT line, indented to align
  with the variable's own indent;  if several related variables need
  long comments, merge them into ONE comment block placed above the
  group, aligned with the variables' indent:

```cpp
    //  The scan window and the line-grid parameters of the
    //  Voigt lookup:  n_vu  the table size,  a_voigt  the
    //  damping parameter,  ...
    int      n_vu;
    float_t  a_voigt;
    float_t  da;
```

- No multi-line derivation / design narrative in code.  A comment
  block longer than ~4 lines that explains *why a design is the way
  it is* belongs in `docs/` (a method document);  the code keeps ONE
  short comment with a pointer to it:

```cpp
//  The boundary rows close through the given face flux
//  ( conservation proof:  docs/methods/fld-method.md § 3 ).
```

  The code near the comment must stay readable on its own;  the
  document carries the math, the alternatives tried, and the
  history.

---

## 13. GPU kernels and device functions

### 13.1 `__global__` kernels stay thin

A `__global__` kernel's body should be as simple as possible:  the
thread-index dispatch and the call.  The implementation lives in the
member functions (or function templates) of the objects passed in.
Reference:  `src/modules/dynamics/hydro/boundary/comm_kernels.h`
(`ker_to_buf` only dispatches to `bnd.to_same / to_finer / to_coarser`)
and `src/modules/dynamics/hydro/integrator/integrator.h` (`intg_ker`
is a one-line forward to `itg.add_flux`).

```cpp
//  ✓
template< class bdt_T >
__global__ void fill_k( const bdt_T b_d )
{
    type::idx_t i;
    if( ! dispatch( i, b_d.geo.n_ceff ) )
        return;
    fill_cell( b_d, i );
    return;
}

//  ✗  (the whole computation inside the kernel body)
```

### 13.2 `__device__` argument passing and returns

Pass arguments by const reference (or const value for scalars that fit
in a register — see § 5.1).  A complex result is written through a
reference parameter at the head of the argument list;  the function
then returns `void`:

```cpp
//  ✓
__device__ __forceinline__ void weights
( const bdt_T & b_d, const idx_T & i,
  f_T ( & w )[ 3 ][ 2 ] );

//  ✗  (returning a big aggregate by value)
```

### 13.4 Policy types are resolved on the host,  never on the device

A `__global__` kernel takes its policy objects **as parameters** and every policy is a
compile-time template argument;  the launch lives in a CRTP base that passes the derived
object into the kernel template;  the runtime choice is a host-side virtual or a host-side
factory.  So the kernel body never branches on a type tag, and "two backends" means two
derived types plus one common base -- not one type with an `if` inside.  The policy object
is copied **by value** into the kernel, hence it must stay trivially copyable:  scalars
only, with every pointer and buffer in a POD view struct that is itself trivially copyable.
The full convention,  with the canonical files to copy from,  is in
`docs/AGENTS.md` § "Device dispatch convention (no runtime type resolution on the device)".

### 13.3 Deep indentation

If a block's indentation grows too deep, extract the deeply indented
part into a separate member function (arguments by reference or const
reference).  Whether it is `__forceinline__` follows § 5.2: small helper
yes, large body no.

---

## 14. Static data on device-reachable classes

No `static` (including `inline static`) data members on classes whose
objects are passed to or touched by device code:  a static is a
process-global with no device copy, and reading it on device is either
a silent host readback or a link error.  Configuration belongs in the
driver's members (set in `read()`);  per-instance state belongs in
ordinary members (initialized in the constructor, § 6.2).


---

## 12. Complete example

```cpp
#pragma once

#include "block_data.h"
#include "../../src/device/general.h"

namespace prob
{

////////////////////////////////////////////////////////////
//  Photon generator (reads from binary file)
////////////////////////////////////////////////////////////

struct gen_t : particle::generate::base_t< gen_t >
{
    ////////// Types //////////
    using super_t = particle::generate::base_t< gen_t >;

    ////////// Data //////////
    int      ncol_ph;
    void *   data_h;
    size_t   s_data;
    int      n_ph_read;
    int      ph_mode;
    int      n_scat;
    int      n_step;
    float_t  b_sca;

    ////////// Functions //////////
    __host__ gen_t(  ) : data_h( nullptr ) {  };

    __host__ virtual void init
    ( const input & args,
      particle::base_t & mod ) override
    {
        super_t::init( args, mod );
        n_scat = args.get< int >
            ( "particle", "n_scat", 0 );
        n_step = args.get< int >
            ( "particle", "n_step", 1000 );
        ph_mode = args.get< int >
            ( "line_rt", "ph_mode", 0 );
        b_sca = args.get< float_t >
            ( "line_rt", "b_sca", 30e5 );
        n_ph_read = args.get< int >
            ( "line_rt", "n_ph_read", -1 );
        try_read_photon( args );
        mode = gen_read;
        return;
    };

    template< class par_T > __host__
    void load_par
    ( par_T & par, const size_t & i_par )
    {
        auto * d_ph = ( float * )( data_h )
            + i_par * ncol_ph;
        for( int a = 0; a < 3; ++ a )
        {
            par.x  [ a ] = d_ph[     a ];
            par.dir[ a ] = d_ph[ 3 + a ];
        }
        par.proper = d_ph[ 6 ];
        return;
    };
};

}
```
