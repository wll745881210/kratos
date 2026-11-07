#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../../../../../../src/types.h"
#include "../../../../../../src/device/device.h"

#include "edge_mg_cplx.h"
#include "edge_mg_csr.h"
#include "edge_mg_data.h"
#include "edge_mg_dev_vec.h"
#include "edge_mg_device.h"
#include "edge_mg_hier.h"
#include "edge_mg_solve.h"

//  the __global__ layer of this file lives in the kernels/ subdirectory here.
#include "kernels/ni_mg_kernels.h"

////////////////////////////////////////////////////////////
//  The device-resident,  MATRIX-FREE multigrid.
//
//  The operator is
//
//      A = M1 + dt K ,
//
//  stored as one 12x12 block per cell ( level 0 ) and
//  applied by gather / block-multiply / scatter,  so no
//  assembled CSR,  no `nc x 1600 B` block array on the bus
//  and no global matrix of any kind takes part in a step.
//
//  The coarse operators are the Galerkin images
//  A_l = P^T A_{l-1} P.  `build_P` ( edge_mg_hier.h ) maps
//  the twelve edges of a fine cell ONLY onto the twelve
//  edges of its own parent coarse cell,  and that is what
//  makes this hierarchy cheap to hold on the device:
//
//      A_l = sum_{ C in level l } B_C ,
//      B_C = sum_{ c in C } W_c^T A_{l-1,c} W_c ,
//
//  with W_c the ( sparse, at most two entries per row ) 12
//  by 12 restriction of P to the cell's rows.  So EVERY
//  level is again a sum of per-cell 12x12 blocks,  the
//  matrix-free apply and the block-Jacobi smoother run
//  verbatim at every level,  and the only per-step work on
//  a coarse level is a deterministic block aggregation
//  kernel ( one thread per coarse cell,  fixed order,  no
//  atomics ).
//
//  What is NOT here:  the auxiliary ( HX node space,  face
//  space ) smoothers of the assembled path.  The smoother is
//  a per-cell block Jacobi,  which is the device-friendly
//  form of the same operator family.
//
//  Two invariants this file relies on:
//
//    (1) the aggregation plan is derived from the P CSR that
//        the host built,  not from a second reading of the
//        prolongation weights -- a re-derivation would repeat
//        any weight mistake silently;
//    (2) every coarse column of a fine cell's P rows must
//        belong to the twelve local edges of that cell's
//        parent -- otherwise the block structure is broken
//        and build() throws instead of answering wrongly.

namespace mhd_ee::ni::edge_mg
{

using type::   idx_t;
using type:: float_t;
using type::float2_t;
using mesh:: dat_3d_t;

////////////////////////////////////////////////////////////
//  How one coarse cell is assembled from its fine children.
//
//  Per ( child, local edge e ) there are at most four
//  prolongation entries:  `p_cidx[ ( child*12 + e )*4 + k ]`
//  is the coarse LOCAL edge of the parent ( 0..11,  or -1
//  for "no such entry" ) and `p_w` the matching weight.
//  Two entries happen per coarsened TRANSVERSE axis,  where
//  a fine edge sits between a coarse node and a coarse
//  half-node,  so two coarsened axes give four.
//
//  Rows and children are grouped by parent:  the children of
//  coarse cell C are
//  `p_child[ p_start[ C ] .. p_start[ C + 1 ] - 1 ]`.
struct mf_agg_t
{
    int             n_child { 0 };
    int *         p_start { nullptr };   // [ n_cell + 1 ]
    int *         p_child { nullptr };   // [ n_child ]
    int *          p_cidx { nullptr };   // [ n_child * 48 ]
    float_t *         p_w { nullptr };   // [ n_child * 48 ]
};

////////////////////////////////////////////////////////////
//  One level.
//
//  `c`,  `p_keep`,  `p_comp` and `p_cnt` are TOPOLOGY:  they
//  are built once.  `p_ablk` ( level > 0 ) is the level's
//  operator as 144 numbers a cell,  refilled every step from
//  the level below.  Level 0 has no `p_ablk`:  its operator
//  is the per-step cell block `p_blk0`,  which is already on
//  the device.
struct mf_lvl_t
{
    cplx_t                c;
    int             n_compact { 0 };
    int *             p_keep { nullptr };
    int *             p_comp { nullptr };
    int *              p_cnt { nullptr };
    float_t *         p_ablk { nullptr };   // level > 0,  144 a cell
    mf_agg_t             agg;                // level > 0
    dev_csr_t              P;                // level < last: fine x coarse
    float_t *            p_x { nullptr };
    float_t *            p_b { nullptr };
    float_t *            p_r { nullptr };
    float_t *            p_z { nullptr };
    float_t *            p_w { nullptr };   // the V-cycle iterate
    float_t *            p_d { nullptr };   // diag( A_l )
    float_t *          p_inv { nullptr };   // 1 / diag( A_l )
    float_t *          p_lam { nullptr };   // lam_max( D^-1 A_l ),  on the device
    float_t *           p_om { nullptr };   // its damping factor, on the device
    float_t            lam { 0. };          // the host copy:  diagnostics only
};

////////////////////////////////////////////////////////////
//  A per-cell block aggregation,  one thread per coarse cell.
//
//  `B_C = sum over the children c of  W_c^T A_c W_c`.  The
//  loop order is fixed by the cell / edge numbering,  not by
//  the thread scheduler:  no atomics,  a private accumulator,
//  one store.  Compared with the host Galerkin product,  a
//  difference can only be the floating-point summation order.
struct blk_agg_t
{
    cplx_t                  cc;
    const cell_blk_t *   p_blk { nullptr };   // children's blocks,  level 1
    const float_t *       p_ab { nullptr };   // children's dense blocks,  level > 1
    float_t                 dt { 0. };
    const int *      p_start { nullptr };
    const int *      p_child { nullptr };
    const int *       p_cidx { nullptr };
    const float_t *      p_w { nullptr };
    float_t *           p_out { nullptr };
    int               n_cell { 0 };

    __device__ __forceinline__ float_t a_of
    ( const int & ic,  const int & e,  const int & e2 ) const
    {
        const int q = e * 12 + e2;
        if( p_ab != nullptr )
        {
            return p_ab[ ic * 144 + q ];
        }
        const cell_blk_t & b = p_blk[ ic ];
        return float_t( b.m1[ q ] ) + dt * float_t( b.k[ q ] );
    };

    __device__ void apply( const int & ic ) const
    {
        float_t acc[ 144 ];
        for( int q = 0; q < 144; ++ q )
        {
            acc[ q ] = float_t( 0. );
        }
        const int s1 = p_start[ ic + 1 ];
        for( int s = p_start[ ic ]; s < s1; ++ s )
        {
            const int      c = p_child[ s ];
            const int *   ci = p_cidx + s * 48;
            const float_t * w = p_w + s * 48;
            for( int e = 0; e < 12; ++ e )
            {
                for( int e2 = 0; e2 < 12; ++ e2 )
                {
                    const float_t a = a_of( c, e, e2 );
                    if( a == float_t( 0. ) )
                    {
                        continue;
                    }
                    for( int k1 = 0; k1 < 4; ++ k1 )
                    {
                        const int e1 = ci[ e * 4 + k1 ];
                        if( e1 < 0 )
                        {
                            continue;
                        }
                        const float_t w1 = w[ e * 4 + k1 ];
                        for( int k2 = 0; k2 < 4; ++ k2 )
                        {
                            const int eo = ci[ e2 * 4 + k2 ];
                            if( eo < 0 )
                            {
                                continue;
                            }
                            acc[ e1 * 12 + eo ] += w1 * a * w[ e2 * 4 + k2 ];
                        }
                    }
                }
            }
        }
        float_t * o = p_out + ic * 144;
        for( int q = 0; q < 144; ++ q )
        {
            o[ q ] = acc[ q ];
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};

////////////////////////////////////////////////////////////
//  The DIAGONAL of a level,  accumulated from its cells.
//
//  A per-cell 12x12 block is NOT a smoother here:
//  lam_max( S A ) grows like h^-2,  while the diagonal Jacobi
//  has a mesh-independent lam_max( D^-1 A ) of about 3.8.  So
//  the smoothing is a DAMPED diagonal Jacobi and this kernel
//  is its only setup:  d[ comp[e] ] = sum over the cells that
//  hold e of A_c[ e ][ e ],  which is exactly the CSR
//  diagonal of the assembled operator.
struct blk_diag_t
{
    cplx_t                    c;
    const cell_blk_t *    p_blk { nullptr };
    const float_t *        p_ab { nullptr };
    float_t                  dt { 0. };
    const int *         p_keep { nullptr };
    const int *         p_comp { nullptr };
    float_t *              p_d { nullptr };

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ];
        c.cell_edges( i, ge );
        for( int e = 0; e < 12; ++ e )
        {
            const int g = ge[ e ];
            if( p_keep[ g ] == 0 || p_comp[ g ] < 0 )
            {
                continue;
            }
            const int q = e * 12 + e;
            float_t a;
            if( p_ab != nullptr )
            {
                a = p_ab[ ic * 144 + q ];
            }
            else
            {
                const cell_blk_t & b = p_blk[ ic ];
                a = float_t( b.m1[ q ] ) + dt * float_t( b.k[ q ] );
            }
            atomicAdd( & p_d[ p_comp[ g ] ], a );
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  z = omega * r / d   ( one damped Jacobi update ).
////////////////////////////////////////////////////////////
//  The scalars of an iteration,  on the device.
//
//  Finishing every dot on the host and computing alpha,  beta
//  and the residual there is a few hundred flops against
//  1e6--1e7 device flops -- noise for speed,  but it would
//  break the invariant:  the host DISPATCHES a step and never
//  computes in it.  The host therefore reads ( res, ok ) once
//  per iteration,  and `res` is a scalar a step is allowed to
//  read back.
//
//  Every op below runs in ONE thread (`real_one_k`).

//  b2 guard,  alpha,  beta,  res
//  (`cg_b2s_t`/`cg_alpha_t`/`cg_beta_t`/`cg_res_t`) live in
//  `edge_mg_dev_vec.h`,  shared with the face solve.

struct cg_om_t
{
    const float_t * p_lam;
    float_t       * p_om;

    //  omega = 2 / ( 1.35 lam ):  the highest mode is damped
    //  by | 1 - omega lam | = 0.48,  and the 35% margin keeps
    //  the sweep stable even if lam is under-estimated by a
    //  tenth ( a factor 1.78 lam diverges once lam is
    //  under-estimated by 12% ).
    __device__ void apply(  ) const
    {
        p_om[ 0 ] = ( * p_lam ) > float_t( 0. )
                  ? float_t( 2. ) / ( ( * p_lam ) * float_t( 1.35 ) )
                  : float_t( 0.5 );
        return;
    };
};

////////////////////////////////////////////////////////////
//  The power iteration that estimates lam_max( D^-1 A ),  on
//  the device.  An ALL-ONES start is ( nearly ) the smoothest
//  vector there is,  so its iterates never leave the bottom of
//  the spectrum and lam comes out near lam_min -- too large a
//  damping factor,  and the sweep diverges on a long axis.
//  Alternating has half of its energy in the highest mode.

struct lam_start_t
{
    float_t * p_v { nullptr };
    int         n { 0 };

    __device__ void apply( const int & i ) const
    {
        p_v[ i ] = ( i % 2 == 0 ) ? float_t( 1. ) : float_t( -1. );
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};

struct lam_div_t
{
    const float_t * p_num { nullptr };
    const float_t * p_den { nullptr };
    float_t *       p_lam { nullptr };

    __device__ void apply(  ) const
    {
        p_lam[ 0 ] = ( * p_den ) > float_t( 0. )
                   ? ( * p_num ) / ( * p_den ) : float_t( 0. );
        return;
    };
};

struct diag_upd_t
{
    const float_t *     p_inv { nullptr };
    const float_t *       p_r { nullptr };
    float_t *             p_z { nullptr };
    const float_t *      p_om { nullptr };   // on the device ( cg_om_t )
    int                     n { 0 };

    __device__ void apply( const int & i ) const
    {
        p_z[ i ] = ( * p_om ) * p_inv[ i ] * p_r[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  1 / d  ( zero on a dof with a non-positive diagonal ).
struct diag_inv_t
{
    const float_t *      p_d { nullptr };
    float_t *          p_inv { nullptr };
    int                    n { 0 };

    __device__ void apply( const int & i ) const
    {
        const float_t d = p_d[ i ];
        p_inv[ i ] = d > float_t( 0. ) ? float_t( 1. ) / d : float_t( 0. );
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};



////////////////////////////////////////////////////////////
//  The matrix-free apply on a DENSE-BLOCK level:  gather /
//  multiply / scatter,  with the twelve by twelve block read
//  from `p_ab` instead of the cell block.
struct mf_blk_t
{
    cplx_t                  c;
    const float_t *       p_ab { nullptr };
    const int *         p_keep { nullptr };
    const int *         p_comp { nullptr };
    const float_t *        p_x { nullptr };
    float_t *              p_y { nullptr };

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ];
        c.cell_edges( i, ge );
        const float_t * ab = p_ab + ic * 144;
        float_t xl[ 12 ];
        for( int e = 0; e < 12; ++ e )
        {
            const int g = ge[ e ];
            xl[ e ] = ( p_keep[ g ] != 0 )
                    ? p_x[ p_comp[ g ] ] : float_t( 0. );
        }
        for( int e = 0; e < 12; ++ e )
        {
            float_t acc( 0. );
            for( int e2 = 0; e2 < 12; ++ e2 )
            {
                acc += ab[ e * 12 + e2 ] * xl[ e2 ];
            }
            const int g = ge[ e ];
            if( p_keep[ g ] != 0 )
            {
                atomicAdd( & p_y[ p_comp[ g ] ], acc );
            }
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  The block-Jacobi smoother on a dense-block level:  a
//  twelve by twelve Cholesky solve,  same gather / scatter
//  rule,  the block read from `p_ab`.
struct blk_blk_t
{
    cplx_t                  c;
    const float_t *       p_ab { nullptr };
    const int *         p_keep { nullptr };
    const int *         p_comp { nullptr };
    const float_t *        p_r { nullptr };
    float_t *              p_x { nullptr };

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ];
        c.cell_edges( i, ge );
        const float_t * ab = p_ab + ic * 144;
        float_t L[ 12 ][ 12 ];
        for( int r = 0; r < 12; ++ r )
        {
            const int gr = ge[ r ];
            const bool live_r = p_keep[ gr ] != 0;
            for( int q = 0; q < 12; ++ q )
            {
                const bool live_q = p_keep[ ge[ q ] ] != 0;
                //  A coarse block is SINGULAR along the local edges the
                //  compaction dropped:  their prolongation column does
                //  not exist,  so the whole row and column of the
                //  aggregated block is zero.  Those dofs get no
                //  correction anyway ( the scatter below skips them ),
                //  so give them a unit row instead of a zero pivot --
                //  otherwise the Cholesky returns inf and the whole
                //  level's preconditioner turns into NaN ( the polar
                //  axis is where this shows ).
                L[ r ][ q ] = ( live_r && live_q ) ? ab[ r * 12 + q ]
                                                   : float_t( 0. );
            }
            L[ r ][ r ] = ( live_r ) ? ab[ r * 12 + r ] : float_t( 1. );
        }
        //  in-place Cholesky,  lower triangle
        for( int r = 0; r < 12; ++ r )
        {
            for( int q = 0; q <= r; ++ q )
            {
                float_t s = L[ r ][ q ];
                for( int k = 0; k < q; ++ k )
                {
                    s -= L[ r ][ k ] * L[ q ][ k ];
                }
                if( r == q )
                {
                    L[ r ][ q ] = ( s > float_t( 0. ) ) ? sqrt( s )
                                                       : float_t( 1. );
                }
                else
                {
                    L[ r ][ q ] = s / L[ q ][ q ];
                }
            }
        }
        float_t v[ 12 ], y[ 12 ];
        for( int e = 0; e < 12; ++ e )
        {
            const int g = ge[ e ];
            v[ e ] = ( p_keep[ g ] != 0 )
                   ? p_r[ p_comp[ g ] ] : float_t( 0. );
        }
        for( int e = 0; e < 12; ++ e )
        {
            float_t s = v[ e ];
            for( int k = 0; k < e; ++ k )
            {
                s -= L[ e ][ k ] * y[ k ];
            }
            y[ e ] = s / L[ e ][ e ];
        }
        for( int e = 11; e >= 0; -- e )
        {
            float_t s = y[ e ];
            for( int k = e + 1; k < 12; ++ k )
            {
                s -= L[ k ][ e ] * y[ k ];
            }
            y[ e ] = s / L[ e ][ e ];
        }
        for( int e = 0; e < 12; ++ e )
        {
            const int g = ge[ e ];
            if( p_keep[ g ] != 0 )
            {
                //  ACCUMULATE ( the cell blocks overlap ),  then
                //  divide by the number of contributions --
                //  `blk_scale_t` below.  A plain store would make
                //  the answer depend on the thread order.
                atomicAdd( & p_x[ p_comp[ g ] ], y[ e ] );
            }
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  The last step of a matrix-free residual:  the apply
//  ACCUMULATED  A x  into r,  so this turns it into
//  `r = b - A x` without a second pass over the blocks.
struct resid_fix_t
{
    const float_t *        p_b { nullptr };
    float_t *              p_r { nullptr };
    int                      n { 0 };

    __device__ void apply( const int & i ) const
    {
        p_r[ i ] = p_b[ i ] - p_r[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  The edge field -> compact vector gather ( level 0 ).
//
//  This is `reduce_rhs` without the host:  it reads the edge
//  1-form out of the block's `a` field directly,  so a step
//  never has to bring `v.a` back to the host to build `a_raw`
//  ( a dropped dof takes its right-hand side with it,  and
//  `p_comp < 0` is exactly that ).
struct gather_a_t
{
    cplx_t                   c;
    const int *          p_keep { nullptr };
    const int *          p_comp { nullptr };
    dat_3d_t< float_t >       a;
    float_t *               p_x { nullptr };

    __device__ void apply( const int & e ) const
    {
        const int k = p_comp[ e ];
        if( k < 0 )
        {
            return;
        }
        int a_( 0 );
        idx_t nd( 0, 0, 0 );
        c.edge_key( e, a_, nd );
        p_x[ k ] = ( p_keep[ e ] != 0 )
                 ? float_t( a.at( nd )[ a_ ] ) : float_t( 0. );
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  The compact vector -> edge field scatter ( level 0 ),  and
//  the uncompacted copy the representation bridge eats ( level
//  0's `x_raw` ).
struct expand_a_t
{
    cplx_t                   c;
    const int *          p_comp { nullptr };
    const float_t *         p_x { nullptr };
    dat_3d_t< float_t >       a;
    float_t *             p_raw { nullptr };

    __device__ void apply( const int & e ) const
    {
        const int k = p_comp[ e ];
        const float_t v = ( k >= 0 ) ? p_x[ k ] : float_t( 0. );
        if( p_raw != nullptr )
        {
            p_raw[ e ] = v;
        }
        int a_( 0 );
        idx_t nd( 0, 0, 0 );
        c.edge_key( e, a_, nd );
        a.at( nd )[ a_ ] = v;
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  The matrix-free hierarchy,  resident on the device.
//
//  One `build` ( from the host hierarchy the assembled path
//  already produces ),  then one `refill` a step,  then
//  `solve`.  Nothing here touches the bus except the O(1)
//  scalars the PCG needs and the vectors the step hands in
//  and out.
struct mf_t;

//  The apply / preconditioner pair the device PCG is
//  templated on:  an operator at a level,  and the V-cycle
//  as its preconditioner.
struct mf_op_t
{
    device::base_t *       dev { nullptr };
    mf_t *                   m { nullptr };
    int                      l { 0 };

    __host__ void apply( const float_t * p_x,
                         float_t * p_y ) const;
};

struct mf_pre_t
{
    device::base_t *       dev { nullptr };
    mf_t *                   m { nullptr };
    int                      l { 0 };
    bool                 cycle { true };

    __host__ const float_t * apply
    ( const float_t * p_r ) const;
};

struct mf_t
{
    device::base_t *          dev { nullptr };
    int                     n_lvl { 0 };
    int                  n_th_lim { 256 };
    int                     n_blk { 32 };   // FIXED ( determinism )
    int                  n_smooth { 2 };
    //  The COARSEST level's nested PCG:  cap and tolerance.
    //  That solve only conditions the V-cycle -- it never
    //  changes the outer answer -- so a tight tolerance buys
    //  nothing.  A cap of 1 conditions about as well as 2 or 10
    //  for a fraction of the step;  2 is the shipped default,
    //  the second iteration left as a margin for meshes whose
    //  coarse level is less benign.  Both are knobs under
    //  [nonideal] ( ee_coarse_iter / ee_coarse_tol ) because the
    //  right cap is a property of the mesh,  not of the method.
    int              coarse_iter { 2 };
    float_t          coarse_tol { float_t( 1e-10 ) };
    //  Diagnostics only ( printed when MF_COARSE_STATS is in
    //  the environment ):  the V-cycles of the last solve,  and
    //  the iterations the coarsest nested PCG spent in them.
    //  That solve is cold started per cycle,  so the second
    //  count is the price of the nested loop rather than of the
    //  outer iteration.
    long long              n_vcyc { 0 };
    long long          n_coarse_it { 0 };
    float_t                    dt { 0. };
    const cell_blk_t *     p_blk0 { nullptr };
    int                    n_edge0 { 0 };
    float_t *              p_part { nullptr };
    float_t *             p_xraw0 { nullptr };
    //  The iteration's scalars live on the device:  row 0 is
    //  the outer solve,  row 1 the coarse solve one level down
    //  ( the two nest,  so they cannot share a slot ).
    //  0 rz,  1 rn,  2 al,  3 be,  4 rr,  5 b2 ( then b2s ),
    //  6 ok,  7 res.
    float_t *         p_sc[ 2 ][ 8 ];
    //  PCG scratch.  Row 0 serves the outer solve,  row 1 the
    //  coarse solve one level down,  so a V-cycle inside the
    //  outer PCG cannot clobber its parent's vectors.
    float_t *         p_scr[ 2 ][ 3 ];
    std::vector< mf_lvl_t >      lv;

    mf_t(  )
    {
        for( int i = 0; i < 2; ++ i )
        {
            for( int j = 0; j < 3; ++ j )
            {
                p_scr[ i ][ j ] = nullptr;
            }
            for( int j = 0; j < 8; ++ j )
            {
                p_sc[ i ][ j ] = nullptr;
            }
        }
        return;
    };

    mf_t( const mf_t & ) = delete;
    mf_t & operator = ( const mf_t & ) = delete;

    ////  allocation helpers  ////

    __host__ int * _up_int
    ( const std::vector< int > & v ) const;

    __host__ float_t * _up_real
    ( const std::vector< float_t > & v ) const;

    __host__ float_t * _alloc
    ( const int & n ) const;

    ////  setup  ////

    //  Collect the topology of the assembled hierarchy the
    //  host just built,  then let the host hierarchy go:  the
    //  production path answers from the device state alone.
    __host__ void build( device::base_t & dev_,
                         const hier_t & h,
                         const int & n_th_lim_ );

    __host__ void free(  );

    ////  per step  ////

    //  Rebuild the coarse blocks from level 0's cell blocks
    //  ( `dt` is only needed at level 0 ).
    __host__ void refill( const cell_blk_t * p_blk,
                          const float_t & dt_ );

    __host__ void rhs_m1
    ( const float_t * p_a,  float_t * p_b ) const;

    __host__ void gather( const dat_3d_t< float_t > & a,
                          float_t * p_x ) const;

    __host__ void expand( const dat_3d_t< float_t > & a,
                          const float_t * p_x,
                          float_t * p_raw ) const;

    ////  the solver  ////

    __host__ void apply( const int & l,
                         const float_t * p_x,
                         float_t * p_y ) const;

    __host__ void apply_dt( const int & l,
                            const float_t * p_x,
                            float_t * p_y,
                            const float_t & dt_ ) const;

    __host__ void refill_diag(  );
    __host__ void lam_est(  );
    __host__ void smooth( const int & l,
                          const float_t * p_b,
                          float_t * p_x );
    __host__ void jacobi( const int & l,
                          const float_t * p_r,
                          float_t * p_z ) const;

    __host__ void resid( const int & l,
                         const float_t * p_x,
                         const float_t * p_b,
                         float_t * p_r ) const;

    //  THE one host read of a solve:  `res` ( slot 7 ) and the
    //  curvature flag ( slot 6 ) of the given nesting depth.
    __host__ float_t _read_res( const int & depth ) const;

    //  One V-cycle in place on the level's `p_x` / `p_b`
    //  ( the coarse solve included ).
    __host__ void vcycle( const int & l,  float_t * p_x );

    __host__ const float_t * cycle_apply( const int & l,
                                          const float_t * p_r );

    __host__ pcg_stat_t pcg( const int & l,
                             const bool & cycle,
                             float_t * p_x,
                             const float_t * p_b,
                             float_t * p_r,
                             float_t * p_p,
                             float_t * p_ap,
                             const float_t & tol,
                             const int & it_max,
                             const bool & warm,
                             const int & depth );

    __host__ pcg_stat_t solve( float_t * p_x,
                               const float_t * p_b,
                               const float_t & tol,
                               const int & it_max,
                               const bool & warm );

    ////  the level-block aggregation plan  ////

    __host__ void _plan( const int & l,
                         const std::vector< int > & keep_f,
                         const std::vector< int > & comp_f,
                         const cplx_t & cf,
                         const std::vector< int > & comp_c,
                         const csr_t & P );
};

////////////////////////////////////////////////////////////
//  Count,  per compact dof of a level,  how many cells touch
//  it:  the divisor of a block-Jacobi sweep.  A topology
//  invariant ( it does not see dt ),  so it is built once.
__host__ std::vector< int > mf_cnt
( const cplx_t & c,  const std::vector< int > & keep,
  const std::vector< int > & comp,  const int & n_compact );

};  // namespace edge_mg