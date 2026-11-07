#pragma once

#include <algorithm>
#include <tuple>

#include "edge_mg_assemble.h"
#include "edge_mg_csr.h"
#include "edge_mg_cplx.h"
#include "edge_mg_data.h"
#include "edge_mg_geo.h"

//  the __global__ layer of this file lives in the kernels/ subdirectory here.
#include "kernels/ni_mg_kernels.h"

namespace device
{
class base_t;
}

////////////////////////////////////////////////////////////
//  The device path of the edge-MG leg,  in the house style:
//
//  * a THIN __global__ function template -- it computes the
//  flat
//  index and calls the operation object's member, nothing
//  else;
//  * the operation TYPE owns the policy, the complex, eta
//  and the
//  output pointer, and its `apply( ic )` does the actual
//  work
//      ( the same assemble_cell the host path calls );
//  * `resource()` computes the launch geometry ( blocks,
//  threads,
//  shared bytes ), the way comm_t::resource_pack does, so
//  the
//      launcher never guesses.
//
//  The grid-stride loop lives in the kernel, not in
//  `apply`, so the
//  same object works with one thread ( the HIP-CPU case )
//  and with a
//  full grid.

namespace mhd_ee::ni::edge_mg
{
using type::float_t;
using type::idx_t;


////////////////////////////////////////////////////////////
//  Assemble every per-cell block

template< class geo_T >
struct blk_assemble_t
{
    geo_T        g;
    cplx_t       c;
    //  ONE tensor for the whole block. Used when p_eta is
    //  null, so every
    //  existing caller is BIT-IDENTICAL.
    float2_t   eta[ 6 ];
    //  THE PRODUCTION FORM: one tensor a cell ( 6 numbers
    //  ), optionally
    //  with its sub-cell gradient ( 18 ) and the expansion
    //  centre ( 3 ).
    //  When p_eta is given it WINS over eta[ 6 ]. This is
    //  the piece the
    //  implicit step needs and the gates did not: a
    //  spatially varying
    //  eta cannot be expressed by a block constant.
    const float2_t * p_eta  { nullptr };
    const float2_t * p_deta { nullptr };
    const float2_t * p_xc   { nullptr };
    cell_blk_t * p_blk;
    int         n_cell;
    //  diagnostics: thread 0 records gridDim * blockDim, so
    //  the host can
    //  tell "the emulator ran one thread" from "it ran a
    //  full grid" -- the
    //  bitwise check passes in BOTH cases and cannot
    //  separate them.
    int *       p_n_thr;

    //// the work of ONE cell
    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        if( p_eta == nullptr )
        {
            assemble_cell< geo_T >( g, c, i, eta, p_blk[ ic ] );
            return;
        }
        float2_t e[ 6 ];
        for( int q = 0; q < 6; ++ q )
        {
            e[ q ] = p_eta[ ic * 6 + q ];
        }
        if( p_deta != nullptr && p_xc != nullptr )
        {
            assemble_cell< geo_T >( g, c, i, e, p_blk[ ic ],
                                    p_deta + ic * 18, p_xc + ic * 3 );
            return;
        }
        assemble_cell< geo_T >( g, c, i, e, p_blk[ ic ] );
        return;
    };

    //// the launch geometry
    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const
    {
        const int n = n_cell > 0 ? n_cell : 1;
        const int t = std::min( n_th_lim, n );
        return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
    };
};

////////////////////////////////////////////////////////////
//  The thin kernel:  index arithmetic only


////////////////////////////////////////////////////////////
//  The RHS map   r = M12 b   ( edges <- faces )
//
//  This one differs from the assembly kernel in a way that
//  matters: its
//  writes ACCUMULATE, and several cells write the same
//  edge. On the
//  device that is a race unless it is an atomic add, and in
//  float_t
//  precision atomicAdd needs sm_60+ ( HIP-CPU emulates it
//  serially ).
//  The host version needs no such thing, which is exactly
//  why the two
//  must be compared with a TOLERANCE and not byte for byte:
//  the ORDER of
//  a floating-point sum is not fixed on a device.
struct rhs_m12_t
{
    cplx_t               c;
    const cell_blk_t * p_blk;
    const float2_t   *   p_b;
    float2_t         *   p_r;
    int              n_cell;

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ], gf[ 6 ];
        c.cell_edges( i, ge );
        c.cell_faces( i, gf );
        for( int e = 0; e < 12; ++ e )
        {
            float2_t t( 0 );
            for( int f = 0; f < 6; ++ f )
            {
                t += float2_t( p_blk[ ic ].m12[ e * 6 + f ] ) * p_b[ gf[ f ] ];
            }
            atomicAdd( & p_r[ ge[ e ] ], t );
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  A CSR operator on the device.
//
//  Uploaded once per ( level, dt ) and then used by
//  everything that needs
//  the operator: the 3x3 gather of the smoother, the
//  residual matvec of
//  the V-cycle, the HX projection. It REPLACES the per-site
//  upload of
//  nine doubles,  which is where the memory actually went.
struct dev_csr_t
{
    int           n { 0 };
    int        ncol { 0 };
    int *       ptr { nullptr };
    int *       idx { nullptr };
    float_t *    val { nullptr };
};

__host__ void upload_csr
( device::base_t & dev, const csr_t & A, dev_csr_t & d );

//  The mirror image. Every pointer is tested, so a
//  default-constructed
//  ( or already freed ) dev_csr_t is a no-op rather than a
//  bad free.
__host__ void free_csr
( device::base_t & dev, dev_csr_t & d );

////////////////////////////////////////////////////////
struct matvec_t
{
    dev_csr_t     S;
    const float_t * p_x;
    float_t       * p_y;

    __device__ void apply( const int & i ) const
    {
        float_t acc( 0. );
        for( int q = S.ptr[ i ]; q < S.ptr[ i + 1 ]; ++ q )
        {
            acc += S.val[ q ] * p_x[ S.idx[ q ] ];
        }
        p_y[ i ] = acc;
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};

////////////////////////////////////////////////////////////
//  The TRANSPOSE matvec y += A^T x on the device (
//  restriction ).
//
//  A transpose multiply SCATTERS, so several rows hit the
//  same column:
//  that is an atomic add, and the comparison with the host
//  has to be a
//  tolerance ( the summation order is not fixed ), unlike
//  matvec_t.
struct matvec_t_t
{
    dev_csr_t     S;
    const float_t * p_x;
    float_t       * p_y;

    __device__ void apply( const int & i ) const
    {
        const float_t xi = p_x[ i ];
        for( int q = S.ptr[ i ]; q < S.ptr[ i + 1 ]; ++ q )
        {
            atomicAdd( & p_y[ S.idx[ q ] ], S.val[ q ] * xi );
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  P5's K3:  the MATRIX-FREE operator,  applied cell by cell.
//
//      y = A x,      A = M1 + dt K
//
//  `_cells_dev` already leaves the 12x12 blocks on the
//  device ( cell_blk_t,  400 floats a cell ),  so the
//  operator can be applied as three local passes per cell --
//  gather the twelve edge values,  multiply the block,
//  scatter the twelve results -- with NO global matrix in
//  between.  That is what replaces `Sl[ l ]`
//  ( assemble_m1k_a + csr_t ) in P5,  and it is what makes
//  the `nc x 1600 B` read-back disappear.
//
//  The numbering is the one the assembled operator used:
//  `comp[ e ]` is the compact index of edge e,  or -1 when
//  `keep[ e ] == 0` ( a dropped dof,  kept out of the
//  system entirely -- M56:  the right-hand side is dropped
//  with it ).  A block entry that meets a dropped edge
//  contributes 0,  so the dropped row reads a = f + l only on
//  the assembled path;  here it is simply absent.
//
//  The scatter ACCUMULATES ( a global edge belongs to up to
//  four cells ),  so it is an atomic add.  That is also why
//  the gate compares this against the CSR matvec with a
//  TOLERANCE and not byte for byte:  the order of a
//  floating-point sum is not fixed on the device.

struct mf_edge_t
{
    cplx_t                c;
    const cell_blk_t * p_blk;
    const int     *    p_keep;
    const int     *    p_comp;
    const float_t *       p_x;
    float_t       *       p_y;
    float_t                 dt;

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ];
        c.cell_edges( i, ge );
        const cell_blk_t & b = p_blk[ ic ];
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
                acc += ( b.m1[ e * 12 + e2 ]
                       + dt * b.k[ e * 12 + e2 ] ) * xl[ e2 ];
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
//  P5's K4:  the PER-CELL block solve,  the smoother of a
//  matrix-free V-cycle.
//
//  One thread per cell:  factor the 12x12 block
//  `A_c = m1 + dt k` ( SPD:  M1 is a mass matrix,  K is a
//  weighted curl-curl ),  then apply `A_c^-1` to the twelve
//  values the cell gathers from the compact vector -- the
//  gather/scatter rule is K3's,  so this is one block-Jacobi
//  sweep with NO global matrix and NO global inverse.
//
//  It is a DIAGONAL-BLOCK preconditioner of the compacted
//  operator,  not the exact diagonal of it:  a live edge is
//  shared by up to four cells,  so its assembled diagonal is
//  a sum of contributions.  That is the standard block
//  Jacobi,  and it is what the host `dinv` path does in the
//  assembled world.
//
//  Cholesky ( not LU ) on purpose:  the block is SPD by
//  construction,  and it is the same factorisation the host
//  reference runs,  so the gate can compare numbers instead
//  of algorithms.

struct blk_solve_t
{
    cplx_t                c;
    const cell_blk_t * p_blk;
    const int     *    p_keep;
    const int     *    p_comp;
    const float_t *       p_r;
    float_t       *       p_x;
    float_t                 dt;

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ];
        c.cell_edges( i, ge );
        const cell_blk_t & b = p_blk[ ic ];
        float_t L[ 12 ][ 12 ];
        for( int r = 0; r < 12; ++ r )
        {
            for( int q = 0; q < 12; ++ q )
            {
                L[ r ][ q ] = b.m1[ r * 12 + q ]
                            + dt * b.k[ r * 12 + q ];
            }
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
                L[ r ][ q ] = ( r == q )
                            ? sqrt( s ) : s / L[ q ][ q ];
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
                //  ACCUMULATE:  the cell blocks OVERLAP ( a live
                //  edge belongs to up to four cells ),  so a
                //  block-Jacobi sweep of the compacted operator
                //  is the sum of the per-cell solves divided by
                //  the number of contributions -- `p_cnt` below.
                //  A plain store would make the answer depend on
                //  the thread order.
                atomicAdd( & p_x[ p_comp[ g ] ], y[ e ] );
            }
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


//  The second half of a block-Jacobi sweep:  divide the
//  accumulated per-cell solves by the number of cells that
//  contributed.  The counts are a TOPOLOGY invariant ( which
//  cells touch a live edge ),  so they are computed once in
//  setup and never change with dt.
struct blk_scale_t
{
    const int *     p_cnt;
    float_t   *     p_x;
    int                 n;

    __device__ void apply( const int & i ) const
    {
        const int c = p_cnt[ i ];
        p_x[ i ] = ( c > 0 ) ? p_x[ i ] / float_t( c )
                             : float_t( 0. );
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  The RESIDUAL   r = b - A x   in one pass.
//
//  The smoother needs the residual before every colour, and
//  computing it on
//  the host meant uploading b, uploading x and downloading
//  r -- three bus
//  crossings per colour. Here it is one kernel: same
//  association as the
//  host (`sum over the row, then b - sum`), so the
//  arithmetic is unchanged
//  and only the FMA contraction of the device compiler
//  differs.
struct resid_t
{
    dev_csr_t      S;
    const float_t * p_x;
    const float_t * p_b;
    float_t       * p_r;

    __device__ void apply( const int & i ) const
    {
        float_t acc( 0. );
        for( int q = S.ptr[ i ]; q < S.ptr[ i + 1 ]; ++ q )
        {
            acc += S.val[ q ] * p_x[ S.idx[ q ] ];
        }
        p_r[ i ] = p_b[ i ] - acc;
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};

////////////////////////////////////////////////////////////
//  The ACCUMULATING matvec   y += A x   ( prolongation ).
//
//  `x[ q ] += corr[ q ]` on the host needed the correction
//  back on the host.
//  Adding in place on the device drops that round trip; the
//  addition itself
//  is the same float_t-plus-float_t,  in the same order.
struct matvec_add_t
{
    dev_csr_t      S;
    const float_t * p_x;
    float_t       * p_y;

    __device__ void apply( const int & i ) const
    {
        float_t acc( 0. );
        for( int q = S.ptr[ i ]; q < S.ptr[ i + 1 ]; ++ q )
        {
            acc += S.val[ q ] * p_x[ S.idx[ q ] ];
        }
        p_y[ i ] += acc;
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


};  // namespace edge_mg
