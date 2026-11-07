#include "edge_mg_rep_dev.h"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

////////////////////////////////////////////////////////////
//  The host half of the device representation.
//
//  Three kernels and a device CG,  no host chain:  the mask
//  and the per-face counts are topology,  the 6x6 blocks are
//  read in place,  and the solve's scalars never leave the
//  device -- the host DISPATCHES and does not compute.
//  Nothing here is device code:  the `__global__` wrappers are
//  in kernels/ni_ie_kernels.h and the seven kernel-op structs
//  stay in the header.
////////////////////////////////////////////////////////////

namespace mhd_ee::ni::edge_mg
{

__host__ void rep_dev_free
( device::base_t & dev, rep_dev_t & st )
{
    if( st.p_deg != nullptr )
    {
        dev.f_free( st.p_deg );
    }
    if( st.p_cnt != nullptr )
    {
        dev.f_free( st.p_cnt );
    }
    if( st.p_rhs != nullptr )
    {
        dev.f_free( st.p_rhs );
    }
    if( st.p_z != nullptr )
    {
        dev.f_free( st.p_z );
    }
    if( st.p_r != nullptr )
    {
        dev.f_free( st.p_r );
    }
    if( st.p_p != nullptr )
    {
        dev.f_free( st.p_p );
    }
    if( st.p_ap != nullptr )
    {
        dev.f_free( st.p_ap );
    }
    if( st.p_part != nullptr )
    {
        dev.f_free( st.p_part );
    }
    if( st.p_sc != nullptr )
    {
        dev.f_free( st.p_sc );
    }
    st = rep_dev_t(  );
    return;
};

__host__ void rep_dev_setup
( device::base_t & dev, const cplx_t & c, rep_dev_t & st )
{
    rep_dev_free( dev, st );
    st.n_face = c.n_face;
    st.n_edge = c.n_edge;
    st.n_cell = c.n_cell;
    st.p_deg = ( int * ) dev.f_malloc
        ( sizeof( int ) * size_t( c.n_face ) );
    st.p_cnt = ( int * ) dev.f_malloc
        ( sizeof( int ) * size_t( c.n_face ) );
    st.p_rhs = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * size_t( c.n_face ) );
    st.p_z = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * size_t( c.n_face ) );
    st.p_r = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * size_t( c.n_face ) );
    st.p_p = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * size_t( c.n_face ) );
    st.p_ap = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * size_t( c.n_face ) );
    st.p_part = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * size_t( st.n_blk ) );
    st.p_sc = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * size_t( 8 ) );
    //  the mask and the counts are TOPOLOGY: computed once on
    //  the host,  uploaded once,  never rebuilt
    std::vector< int > deg( size_t( c.n_face ), 0 );
    std::vector< int > cnt( size_t( c.n_face ), 0 );
    for( int f = 0; f < c.n_face; ++ f )
    {
        deg[ size_t( f ) ] = c.degenerate_f( f ) ? 1 : 0;
    }
    for( int ic = 0; ic < c.n_cell; ++ ic )
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int gf[ 6 ];
        c.cell_faces( i, gf );
        for( int f = 0; f < 6; ++ f )
        {
            if( deg[ size_t( gf[ f ] ) ] == 0 )
            {
                ++ cnt[ size_t( gf[ f ] ) ];
            }
        }
    }
    dev.f_cp( st.p_deg, deg.data(  ),
              sizeof( int ) * size_t( c.n_face ) );
    dev.f_cp( st.p_cnt, cnt.data(  ),
              sizeof( int ) * size_t( c.n_face ) );
    dev.f_mset( st.p_rhs, 0,
                sizeof( float_t ) * size_t( c.n_face ), nullptr );
    dev.f_mset( st.p_z, 0,
                sizeof( float_t ) * size_t( c.n_face ), nullptr );
    dev.f_mset( st.p_r, 0,
                sizeof( float_t ) * size_t( c.n_face ), nullptr );
    dev.f_mset( st.p_p, 0,
                sizeof( float_t ) * size_t( c.n_face ), nullptr );
    dev.f_mset( st.p_ap, 0,
                sizeof( float_t ) * size_t( c.n_face ), nullptr );
    dev.f_mset( st.p_sc, 0,
                sizeof( float_t ) * size_t( 8 ), nullptr );
    dev.sync_all_streams(  );
    return;
};

__host__ void rep_dev_ensure
( device::base_t & dev, const cplx_t & c, rep_dev_t & st,
  const int & n_th_lim )
{
    if( st.n_face != c.n_face || st.n_edge != c.n_edge
     || st.n_cell != c.n_cell || st.p_deg == nullptr )
    {
        rep_dev_setup( dev, c, st );
    }
    if( n_th_lim > 0 )
    {
        st.n_th = n_th_lim;
    }
    return;
};

__host__ void rep_dev_rhs
( device::base_t & dev, const cplx_t & c, rep_dev_t & st,
  const cell_blk_t * p_blk, const float2_t * p_a )
{
    dev.f_mset( st.p_rhs, 0,
                sizeof( float_t ) * size_t( st.n_face ), nullptr );
    rep_rhs_op_t op;
    op.c     = c;
    op.p_blk = p_blk;
    op.p_a   = p_a;
    op.p_deg = st.p_deg;
    op.p_r   = st.p_rhs;
    const int n = c.n_cell > 0 ? c.n_cell : 1;
    const int t = std::min( st.n_th, n );
    dev.launch( rep_cell_k< rep_rhs_op_t >,
                dim3( ( n + t - 1 ) / t ), dim3( t ), 0, nullptr,
                op, c.n_cell );
    dev.sync_all_streams(  );
    return;
};

__host__ void rep_dev_matvec
( device::base_t & dev, const cplx_t & c, const rep_dev_t & st,
  const cell_blk_t * p_blk, const float_t * p_x,
  float_t * p_y )
{
    dev.f_mset( p_y, 0,
                sizeof( float_t ) * size_t( st.n_face ), nullptr );
    rep_mv_op_t op;
    op.c     = c;
    op.p_blk = p_blk;
    op.p_deg = st.p_deg;
    op.p_x   = p_x;
    op.p_y   = p_y;
    {
        const int n = c.n_cell > 0 ? c.n_cell : 1;
        const int t = std::min( st.n_th, n );
        dev.launch( rep_cell_k< rep_mv_op_t >,
                    dim3( ( n + t - 1 ) / t ), dim3( t ), 0, nullptr,
                    op, c.n_cell );
    }
    {
        rep_mask_op_t mk;
        mk.p_deg = st.p_deg;
        mk.p_x   = p_x;
        mk.p_y   = p_y;
        const int n = st.n_face > 0 ? st.n_face : 1;
        const int t = std::min( st.n_th, n );
        dev.launch( rep_cell_k< rep_mask_op_t >,
                    dim3( ( n + t - 1 ) / t ), dim3( t ), 0, nullptr,
                    mk, st.n_face );
    }
    dev.sync_all_streams(  );
    return;
};

__host__ const float_t * rep_dev_pre
( device::base_t & dev, const cplx_t & c, const rep_dev_t & st,
  const cell_blk_t * p_blk, const float_t * p_r )
{
    dev.f_mset( st.p_z, 0,
                sizeof( float_t ) * size_t( st.n_face ), nullptr );
    rep_blk_op_t op;
    op.c     = c;
    op.p_blk = p_blk;
    op.p_deg = st.p_deg;
    op.p_r   = p_r;
    op.p_z   = st.p_z;
    {
        const int n = c.n_cell > 0 ? c.n_cell : 1;
        const int t = std::min( st.n_th, n );
        dev.launch( rep_cell_k< rep_blk_op_t >,
                    dim3( ( n + t - 1 ) / t ), dim3( t ), 0, nullptr,
                    op, c.n_cell );
    }
    {
        rep_scale_op_t sc;
        sc.p_cnt = st.p_cnt;
        sc.p_r   = p_r;
        sc.p_z   = st.p_z;
        const int n = st.n_face > 0 ? st.n_face : 1;
        const int t = std::min( st.n_th, n );
        dev.launch( rep_cell_k< rep_scale_op_t >,
                    dim3( ( n + t - 1 ) / t ), dim3( t ), 0, nullptr,
                    sc, st.n_face );
    }
    dev.sync_all_streams(  );
    return st.p_z;
};

__host__ rep_solve_stat_t rep_dev_solve
( device::base_t & dev, const cplx_t & c, const rep_dev_t & st,
  const cell_blk_t * p_blk, float_t * p_rep,
  const float_t * p_x0, const float_t & tol, const int & it_max )
{
    rep_solve_stat_t out;
    const int n = st.n_face;
    float_t * p_rz  = st.p_sc + 0;
    float_t * p_rn  = st.p_sc + 1;
    float_t * p_al  = st.p_sc + 2;
    float_t * p_be  = st.p_sc + 3;
    float_t * p_rr  = st.p_sc + 4;   //  also pAp
    float_t * p_b2s = st.p_sc + 5;
    float_t * p_ok  = st.p_sc + 6;
    float_t * p_res = st.p_sc + 7;
    //  the starting point and its residual
    if( p_x0 == nullptr )
    {
        dev.f_mset( p_rep, 0, sizeof( float_t ) * size_t( n ), nullptr );
        real_copy( dev, st.p_rhs, st.p_r, n, st.n_th );
    }
    else
    {
        real_copy( dev, p_x0, p_rep, n, st.n_th );
        rep_dev_matvec( dev, c, st, p_blk, p_rep, st.p_r );
        rep_resid_op_t rs;
        rs.p_rhs = st.p_rhs;
        rs.p_r   = st.p_r;
        const int m = n > 0 ? n : 1;
        const int t = std::min( st.n_th, m );
        dev.launch( rep_cell_k< rep_resid_op_t >,
                    dim3( ( m + t - 1 ) / t ), dim3( t ), 0, nullptr,
                    rs, n );
    }
    real_dot_dev( dev, st.p_rhs, st.p_rhs, st.p_part, n, st.n_blk, p_b2s );
    {
        cg_b2s_t op;
        op.p_b2  = p_b2s;
        op.p_b2s = p_b2s;
        dev.launch( real_one_k< cg_b2s_t >, dim3( 1 ), dim3( 1 ),
                    0, nullptr, op );
    }
    const float_t * z = rep_dev_pre( dev, c, st, p_blk, st.p_r );
    real_dot_dev( dev, st.p_r, z, st.p_part, n, st.n_blk, p_rz );
    real_copy( dev, z, st.p_p, n, st.n_th );
    real_dot_dev( dev, st.p_r, st.p_r, st.p_part, n, st.n_blk, p_rr );
    {
        cg_res_t op;
        op.p_rr  = p_rr;
        op.p_b2s = p_b2s;
        op.p_res = p_res;
        dev.launch( real_one_k< cg_res_t >, dim3( 1 ), dim3( 1 ),
                    0, nullptr, op );
    }
    float_t h[ 8 ];
    dev.cp( h, st.p_sc, 8 );
    out.res = h[ 7 ];
    while( out.res > tol && out.it < it_max )
    {
        rep_dev_matvec( dev, c, st, p_blk, st.p_p, st.p_ap );
        //  pAp goes into the `rr` slot:  rr is recomputed from
        //  scratch at the end of the iteration anyway.
        real_dot_dev( dev, st.p_p, st.p_ap, st.p_part, n, st.n_blk, p_rr );
        {
            cg_alpha_t op;
            op.p_rz  = p_rz;
            op.p_pap = p_rr;
            op.p_al  = p_al;
            op.p_ok  = p_ok;
            dev.launch( real_one_k< cg_alpha_t >, dim3( 1 ), dim3( 1 ),
                        0, nullptr, op );
        }
        real_axpy_dev( dev, st.p_p,  p_rep, p_al, float_t(  1. ), n,
                       st.n_th, p_ok );
        real_axpy_dev( dev, st.p_ap, st.p_r, p_al, float_t( -1. ), n,
                       st.n_th, p_ok );
        z = rep_dev_pre( dev, c, st, p_blk, st.p_r );
        real_dot_dev( dev, st.p_r, z, st.p_part, n, st.n_blk, p_rn );
        {
            cg_beta_t op;
            op.p_rn = p_rn;
            op.p_rz = p_rz;
            op.p_be = p_be;
            dev.launch( real_one_k< cg_beta_t >, dim3( 1 ), dim3( 1 ),
                        0, nullptr, op );
        }
        real_xpay_dev( dev, z, st.p_p, p_be, n, st.n_th, p_ok );
        real_dot_dev( dev, st.p_r, st.p_r, st.p_part, n, st.n_blk, p_rr );
        {
            cg_res_t op;
            op.p_rr  = p_rr;
            op.p_b2s = p_b2s;
            op.p_res = p_res;
            dev.launch( real_one_k< cg_res_t >, dim3( 1 ), dim3( 1 ),
                        0, nullptr, op );
        }
        ++ out.it;
        //  THE host read of an iteration.
        dev.cp( h, st.p_sc, 8 );
        out.res = h[ 7 ];
        if( !( h[ 6 ] > float_t( 0. ) ) )
        {
            //  A non-positive curvature:  alpha was zero and the
            //  guard kept both updates off,  so the state is
            //  clean.
            break;
        }
    }
    return out;
};

__host__ void ni_ie_represent_dev
( device::base_t & dev, const cplx_t & c,
  const cell_blk_t * p_blk, const float2_t * p_a, rep_dev_t & st,
  float_t * p_rep, const float_t & tol, const int & it_max,
  int & it, float_t & res, const int & n_th_lim,
  const float_t * p_x0 )
{
    rep_dev_ensure( dev, c, st, n_th_lim );
    rep_dev_rhs( dev, c, st, p_blk, p_a );
    const rep_solve_stat_t s = rep_dev_solve
        ( dev, c, st, p_blk, p_rep, p_x0, tol, it_max );
    it  = s.it;
    res = s.res;
    return;
};

__host__ std::tuple< dim3, dim3, int >
rep_rhs_raw_op_t::resource( const int & n_th_lim ) const
{
    const int n = int( c.n_cell );
    const int t = std::min( n_th_lim > 0 ? n_th_lim : 128,
                            n > 0 ? n : 1 );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ void rep_dev_rhs_raw
( device::base_t & dev, const cplx_t & c, rep_dev_t & st,
  const cell_blk_t * p_blk, const float_t * p_a )
{
    dev.f_mset( st.p_rhs, 0,
                sizeof( float_t ) * size_t( st.n_face ), nullptr );
    rep_rhs_raw_op_t op;
    op.c = c;
    op.p_blk = p_blk;
    op.p_a = p_a;
    op.p_deg = st.p_deg;
    op.p_r = st.p_rhs;
    const auto [ n_bl, n_th, s_sh ] = op.resource( st.n_th );
    dev.launch( rep_rhs_raw_k< rep_rhs_raw_op_t >, n_bl, n_th, s_sh,
                nullptr, op, int( c.n_cell ) );
    return;
};

__host__ void ni_ie_represent_dev_raw
( device::base_t & dev, const cplx_t & c,
  const cell_blk_t * p_blk, const float_t * p_a, rep_dev_t & st,
  float_t * p_rep, const float_t & tol, const int & it_max,
  int & it, float_t & res, const int & n_th_lim,
  const float_t * p_x0 )
{
    rep_dev_ensure( dev, c, st, n_th_lim );
    rep_dev_rhs_raw( dev, c, st, p_blk, p_a );
    const rep_solve_stat_t s = rep_dev_solve
        ( dev, c, st, p_blk, p_rep, p_x0, tol, it_max );
    it  = s.it;
    res = s.res;
    return;
};

};    // namespace mhd_ee::ni::edge_mg