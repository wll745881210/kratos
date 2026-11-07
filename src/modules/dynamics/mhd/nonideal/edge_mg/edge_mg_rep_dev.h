#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "edge_mg_cplx.h"
#include "edge_mg_data.h"
#include "edge_mg_dev_vec.h"

//  the __global__ layer of this file lives in the kernels/ subdirectory here.
#include "kernels/ni_ie_kernels.h"

////////////////////////////////////////////////////////////
//  P5 / K6 -- the DEVICE "representation out" of one
//  implicit edge step:
//
//      rep = M2^-1 M12^T a
//
//  The host path (`_assemble_m2` + `_represent` in
//  edge_mg.cpp) assembles the face mass M2 as a CSR matrix
//  every step, reads the edge state back and solves with a
//  host CG. This header replaces the whole chain with three
//  kernels and a device CG:
//
//    (1) rhs    rep = M12^T a     edge -> face,  atomicAdd,
//                                 degenerate faces get 0  ( M56 )
//    (2) matvec y = M2 x          6x6 block gather per cell,
//                                 skip degenerate rows AND
//                                 columns, atomicAdd scatter,
//                                 then an identity row for the
//                                 dropped faces ( mask kernel )
//    (3) pre    z = M2^-1 r       6x6 per-cell block Cholesky
//                                 ( the SAME skip rule ), one
//                                 block-Jacobi sweep: accumulate
//                                 with atomicAdd and divide by
//                                 the per-face cell count
//
//  Why M2 is never assembled: it is a 6x6 block per cell, so
//  the operator is one gather + one scatter per cell. The
//  assembled CSR of the host path is ( for this operator ) a
//  copy of the same numbers with a row index -- the face
//  mass is always read in place.
//
//  THE TWO RULES THIS FILE MUST NOT BREAK:
//
//  * M56 -- a dropped face DOF carries an identity row, so
//    its right-hand side MUST be zero, or the row reads
//    `a = 0 + rhs`. `rep_rhs_op_t` skips the degenerate rows;
//    nothing else may drive them.
//  * A4.31 -- the scatter ACCUMULATES ( atomicAdd ) and is
//    normalised afterwards ( `rep_scale_op_t` ). A plain
//    store would make the preconditioner depend on the thread
//    order.
//
//  The 6x6 blocks are the per-cell ones -- a face is shared
//  by up to two cells, so ( exactly as for the 12x12 block
//  Jacobi of `blk_solve_t` ) the block-Jacobi solve of the
//  assembled operator is "sum of the per-cell solves / number
//  of contributions", and the count is a topology invariant.

namespace mhd_ee::ni::edge_mg
{
using type::float_t;
using type::float2_t;
using type::idx_t;


////////////////////////////////////////////////////////////
//  Setup once per ( block, topology ):  the degenerate-face
//  mask, the per-face cell count, and the CG workspace. Every
//  one of those is a topology invariant -- nothing changes
//  with dt, eta or the state.
struct rep_dev_t
{
    ////////// Data //////////
    int         n_face { 0 };
    int         n_edge { 0 };
    int         n_cell { 0 };
    //  topology invariants ( device )
    int *        p_deg { nullptr };   //  n_face : 1 = degenerate ( dropped )
    int *        p_cnt { nullptr };   //  n_face : # of cells with a live row
    //  workspace ( device )
    float_t *    p_rhs { nullptr };   //  n_face : M12^T a, degenerate = 0
    float_t *      p_z { nullptr };   //  n_face : the preconditioned vector
    float_t *      p_r { nullptr };   //  n_face : the residual
    float_t *      p_p { nullptr };   //  n_face : the direction
    float_t *     p_ap { nullptr };   //  n_face : M2 times the direction
    float_t *   p_part { nullptr };   //  n_blk  : the dot partials
    //  the CG's scalars in ONE 8-float allocation:  rz / rn /
    //  al / be / rr ( also pAp ) / b2s / ok / res.  Same layout
    //  as `mf_t::p_sc`,  so the host reads ONE 8-float copy per
    //  iteration and computes nothing.
    float_t *     p_sc { nullptr };
    int          n_blk { 32 };        //  FIXED, not a knob ( determinism )
    int           n_th { 256 };       //  the launch thread limit
};

////////////////////////////////////////////////////////////
//  (1) rhs   r += M12^T a    ( edges -> faces )
//
//  Same gathering as `ref_rhs_t` / `rhs_m12_t`,  but the
//  output is a SCALAR array:  the host path wraps `a` into
//  float2 and keeps the real part,  so this kernel reads the
//  real part of the float2 array the caller hands it ( the
//  convention of `_represent`: `a2[ e ] = float2( a[ e ] )`
//  ).  Degenerate faces are SKIPPED -- that is M56: the
//  dropped DOF owns an identity row,  so its right-hand side
//  has to be zero,  not "whatever the source scattered into
//  it".
struct rep_rhs_op_t
{
    cplx_t              c;
    const cell_blk_t *  p_blk;
    const float2_t   *    p_a;
    const int        *  p_deg;
    float_t          *    p_r;

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ], gf[ 6 ];
        c.cell_edges( i, ge );
        c.cell_faces( i, gf );
        for( int f = 0; f < 6; ++ f )
        {
            if( p_deg[ gf[ f ] ] != 0 )
            {
                continue;
            }
            float_t t( 0. );
            for( int e = 0; e < 12; ++ e )
            {
                t += p_blk[ ic ].m12[ e * 6 + f ]
                   * float_t( p_a[ ge[ e ] ] );
            }
            atomicAdd( & p_r[ gf[ f ] ], t );
        }
        return;
    };
};

////////////////////////////////////////////////////////////
//  (2a) y += M2 x     ( the face mass,  6x6 block a cell )
//
//  Rows AND columns of a degenerate face are skipped,  so the
//  dropped face is decoupled in BOTH directions:  no live row
//  reads it and its own row receives nothing.  The identity
//  row is added by the mask kernel below,  in ONE place,  so
//  the operator has exactly the host `_assemble_m2` action.
struct rep_mv_op_t
{
    cplx_t              c;
    const cell_blk_t *  p_blk;
    const int        *  p_deg;
    const float_t    *    p_x;
    float_t          *    p_y;

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int gf[ 6 ];
        c.cell_faces( i, gf );
        for( int f = 0; f < 6; ++ f )
        {
            if( p_deg[ gf[ f ] ] != 0 )
            {
                continue;
            }
            float_t s( 0. );
            for( int f2 = 0; f2 < 6; ++ f2 )
            {
                if( p_deg[ gf[ f2 ] ] != 0 )
                {
                    continue;
                }
                s += p_blk[ ic ].m2[ f * 6 + f2 ] * p_x[ gf[ f2 ] ];
            }
            //  ACCUMULATE:  a face belongs to up to two cells
            atomicAdd( & p_y[ gf[ f ] ], s );
        }
        return;
    };
};

////////////////////////////////////////////////////////////
//  (2b) the identity row of every dropped face
//
//    y[ f ] = x[ f ]     for a degenerate f
//
//  This is the `( f, f, 1 )` triplet of `_assemble_m2`. It is
//  a separate kernel because it touches the OPERATOR,  not a
//  cell,  and because a mask applied after the accumulation
//  is exactly equivalent to the host identity row ( no cell
//  scatters into a degenerate row ).
struct rep_mask_op_t
{
    const int     *  p_deg;
    const float_t *    p_x;
    float_t       *    p_y;

    __device__ void apply( const int & f ) const
    {
        if( p_deg[ f ] != 0 )
        {
            p_y[ f ] = p_x[ f ];
        }
        return;
    };
};

////////////////////////////////////////////////////////////
//  (3a) the 6x6 block-Jacobi preconditioner
//
//  Per cell:  take the m x m sub-block of `m2` over the LIVE
//  local faces ( m = 6 unless the cell touches the polar
//  ring,  where it is 5 -- the SAME skip rule `_assemble_m2`
//  uses ),  Cholesky,  solve for the gathered residual,  and
//  accumulate.  m is read at run time from the mask;  the
//  storage is the fixed 6x6 of `m2`,  which is what makes the
//  kernel branch-free in its arrays.
struct rep_blk_op_t
{
    cplx_t              c;
    const cell_blk_t *  p_blk;
    const int        *  p_deg;
    const float_t    *    p_r;
    float_t          *    p_z;

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int gf[ 6 ], lf[ 6 ];
        c.cell_faces( i, gf );
        int m( 0 );
        for( int f = 0; f < 6; ++ f )
        {
            if( p_deg[ gf[ f ] ] == 0 )
            {
                lf[ m ] = f;
                ++ m;
            }
        }
        if( m == 0 )
        {
            return;
        }
        //  the sub-block,  row-major in a 6-wide buffer so the
        //  Cholesky loop bounds stay static
        float_t L[ 36 ];
        for( int a = 0; a < m; ++ a )
        {
            for( int b = 0; b < m; ++ b )
            {
                L[ a * 6 + b ] = p_blk[ ic ].m2[ lf[ a ] * 6 + lf[ b ] ];
            }
        }
        //  in-place Cholesky,  lower triangle ( SPD:  the
        //  block is a mass matrix ).  A non-positive pivot is
        //  a bug,  and it surfaces as a NaN in the solution,
        //  which the gate treats as a FAIL.
        for( int a = 0; a < m; ++ a )
        {
            for( int b = 0; b <= a; ++ b )
            {
                float_t s = L[ a * 6 + b ];
                for( int k = 0; k < b; ++ k )
                {
                    s -= L[ a * 6 + k ] * L[ b * 6 + k ];
                }
                L[ a * 6 + b ] = ( a == b )
                               ? sqrt( s ) : s / L[ b * 6 + b ];
            }
        }
        float_t v[ 6 ], y[ 6 ];
        for( int a = 0; a < m; ++ a )
        {
            v[ a ] = p_r[ gf[ lf[ a ] ] ];
        }
        for( int a = 0; a < m; ++ a )
        {
            float_t s = v[ a ];
            for( int k = 0; k < a; ++ k )
            {
                s -= L[ a * 6 + k ] * y[ k ];
            }
            y[ a ] = s / L[ a * 6 + a ];
        }
        for( int a = m - 1; a >= 0; -- a )
        {
            float_t s = y[ a ];
            for( int k = a + 1; k < m; ++ k )
            {
                s -= L[ k * 6 + a ] * y[ k ];
            }
            y[ a ] = s / L[ a * 6 + a ];
        }
        for( int a = 0; a < m; ++ a )
        {
            //  ACCUMULATE -- see A4.31 in the file comment
            atomicAdd( & p_z[ gf[ lf[ a ] ] ], y[ a ] );
        }
        return;
    };
};

////////////////////////////////////////////////////////////
//  (3b) the normalisation of the block-Jacobi sweep
//
//    z[ f ] = acc[ f ] / cnt[ f ]      ( cnt > 0 )
//    z[ f ] = r[ f ]                   ( cnt = 0: a dropped
//                                        face,  whose exact
//                                        inverse is the
//                                        identity )
//
//  cnt is `Sigma_cells 1` over the LIVE local rows,  i.e. the
//  number of per-cell blocks that wrote the face. It cannot be
//  zero for a live face and it is zero for every degenerate
//  one,  so the two cases are exactly "assembled" and
//  "identity row".
struct rep_scale_op_t
{
    const int     *  p_cnt;
    const float_t *    p_r;
    float_t       *    p_z;

    __device__ void apply( const int & f ) const
    {
        const int q = p_cnt[ f ];
        p_z[ f ] = ( q > 0 ) ? p_z[ f ] / float_t( q ) : p_r[ f ];
        return;
    };
};

////////////////////////////////////////////////////////////
//  The thin kernel:  index arithmetic only ( the house
//  pattern of `blk_assemble_k` / `real_k` )


////////////////////////////////////////////////////////////
//  Host entries

__host__ void rep_dev_free
( device::base_t & dev, rep_dev_t & st );

//  Setup ONCE per topology. `rep_dev_ensure` ( below ) calls
//  it lazily, so a caller may hand over a default-constructed
//  state.
__host__ void rep_dev_setup
( device::base_t & dev, const cplx_t & c, rep_dev_t & st );

//  The lazy variant:  a state from another topology ( or a
//  fresh default-constructed one ) is rebuilt. This is what
//  lets the driver call the entry point once per step without
//  a setup call of its own.
__host__ void rep_dev_ensure
( device::base_t & dev, const cplx_t & c, rep_dev_t & st,
  const int & n_th_lim );

//  (1) rhs = M12^T a,  degenerate faces zero ( M56 )
__host__ void rep_dev_rhs
( device::base_t & dev, const cplx_t & c, rep_dev_t & st,
  const cell_blk_t * p_blk, const float2_t * p_a );

//  (2) y = M2 x,  with the identity row of every dropped face
__host__ void rep_dev_matvec
( device::base_t & dev, const cplx_t & c, const rep_dev_t & st,
  const cell_blk_t * p_blk, const float_t * p_x,
  float_t * p_y );

//  (3) z = M^-1 r,  z is the state's own scratch and stays
//  valid until the next call
__host__ const float_t * rep_dev_pre
( device::base_t & dev, const cplx_t & c, const rep_dev_t & st,
  const cell_blk_t * p_blk, const float_t * p_r );

////////////////////////////////////////////////////////////
//  (4) the device CG.
//
//  Every scalar of the iteration -- the b2 guard,  alpha
//  ( with a curvature-breakdown flag ),  beta,  the residual
//  -- is computed ON THE DEVICE by one-thread kernels,  and
//  every vector update reads its coefficient from the device.
//  The host's part of an iteration is ONE 8-float copy ( `res`
//  for the loop test,  `ok` for the breakdown ) plus the
//  launches:  it DISPATCHES and does not compute ( H1,  as
//  STATUS.md ( 90 ) states it ).  `mf_t::pcg` in
//  `edge_mg_mf.h` has the same shape on the same slot layout.
//
//  The cold path ( `p_x0 == nullptr` ) starts from x = 0,  r =
//  b,  which is bit-identical to what `pcg_dev_t` did -- the
//  gate numbers did not move when this replaced it.  The warm
//  path copies `p_x0`,  forms A x into `p_r` and flips it into
//  b - A x,  which saves 2-3 iterations:  the face state of a
//  step is a good guess for the next one.
struct rep_resid_op_t
{
    const float_t * p_rhs;
    float_t       *   p_r;

    __device__ void apply( const int & iface ) const
    {
        p_r[ iface ] = p_rhs[ iface ] - p_r[ iface ];
        return;
    };
};

struct rep_solve_stat_t
{
    int        it { 0 };
    float_t   res { 0. };
};

__host__ rep_solve_stat_t rep_dev_solve
( device::base_t & dev, const cplx_t & c, const rep_dev_t & st,
  const cell_blk_t * p_blk, float_t * p_rep,
  const float_t * p_x0, const float_t & tol, const int & it_max );

////////////////////////////////////////////////////////////
//  THE ENTRY POINT the implicit step calls  ( K6 ):
//
//      rep = M2^-1 M12^T a
//
//  `p_a` is the device edge array in the FLOAT2 convention of
//  `_represent` (`a2[e] = float2(a[e])`;  only the real part
//  is read),  `p_rep` is `c.n_face` floats on the device,  and
//  `st` is set up lazily on the first call. `it` / `res` are
//  the CG statistics ( the RECURSIVE residual,  as
//  `pcg_dev_t` reports it ).
__host__ void ni_ie_represent_dev
( device::base_t & dev, const cplx_t & c,
  const cell_blk_t * p_blk, const float2_t * p_a, rep_dev_t & st,
  float_t * p_rep, const float_t & tol, const int & it_max,
  int & it, float_t & res, const int & n_th_lim,
  const float_t * p_x0 = nullptr );

////////////////////////////////////////////////////////////
//  The RAW variant of the bridge:  `p_a` is indexed by
//  GLOBAL EDGE id and holds one float_t each.
//
//  The float2 entry point mirrors the host `_represent`,
//  which materialises `a2[e] = float2(a[e])`.  A device step
//  has no such array and no reason to make one:  `mf_t::expand`
//  already writes the raw per-edge values (`p_xraw0`),  and the
//  bridge only ever reads the real part.  So this variant takes
//  that array directly -- same rhs,  same operator,  same solve.

struct rep_rhs_raw_op_t
{
    cplx_t              c;
    const cell_blk_t *  p_blk;
    const float_t *       p_a;
    const int        *  p_deg;
    float_t          *    p_r;

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ], gf[ 6 ];
        c.cell_edges( i, ge );
        c.cell_faces( i, gf );
        for( int f = 0; f < 6; ++ f )
        {
            if( p_deg[ gf[ f ] ] != 0 )
            {
                continue;
            }
            float_t t( 0. );
            for( int e = 0; e < 12; ++ e )
            {
                t += p_blk[ ic ].m12[ e * 6 + f ]
                   * p_a[ ge[ e ] ];
            }
            atomicAdd( & p_r[ gf[ f ] ], t );
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};

__host__ void rep_dev_rhs_raw
( device::base_t & dev, const cplx_t & c, rep_dev_t & st,
  const cell_blk_t * p_blk, const float_t * p_a );

__host__ void ni_ie_represent_dev_raw
( device::base_t & dev, const cplx_t & c,
  const cell_blk_t * p_blk, const float_t * p_a, rep_dev_t & st,
  float_t * p_rep, const float_t & tol, const int & it_max,
  int & it, float_t & res, const int & n_th_lim,
  const float_t * p_x0 = nullptr );

};  // namespace mhd_ee::ni::edge_mg
