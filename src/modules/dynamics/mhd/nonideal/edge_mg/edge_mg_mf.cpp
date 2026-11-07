#include "edge_mg_mf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <vector>

////////////////////////////////////////////////////////////
//  The host half of the matrix-free hierarchy:  the launch
//  geometry of every op struct,  the allocation helpers,
//  the build / refill of the device hierarchy and the step
//  itself ( gather / apply / smooth / V-cycle / PCG ).
//
//  What is NOT here:  anything the device executes.  The op
//  structs,  their `__device__` bodies and the kernels they
//  are launched through stay in the header;  nothing in
//  this file computes a solution -- it dispatches,  and
//  reads back the O(1) scalars a step is allowed to take.

namespace mhd_ee::ni::edge_mg
{

__host__ std::tuple< dim3, dim3, int >
blk_agg_t::resource( const int & n_th_lim ) const
{
    const int n = n_cell > 0 ? n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
blk_diag_t::resource( const int & n_th_lim ) const
{
    const int n = int( c.n_cell );
    return std::make_tuple( dim3( unsigned( n ) ), dim3( 1 ),
                            n_th_lim > 0 ? n_th_lim : 128 );
};

__host__ std::tuple< dim3, dim3, int >
lam_start_t::resource( const int & n_th_lim ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
diag_upd_t::resource( const int & n_th_lim ) const
{
    const int n_ = n > 0 ? n : 1;
    return std::make_tuple( dim3( unsigned( n_ ) ), dim3( 1 ),
                            n_th_lim > 0 ? n_th_lim : 128 );
};

__host__ std::tuple< dim3, dim3, int >
diag_inv_t::resource( const int & n_th_lim ) const
{
    const int n_ = n > 0 ? n : 1;
    return std::make_tuple( dim3( unsigned( n_ ) ), dim3( 1 ),
                            n_th_lim > 0 ? n_th_lim : 128 );
};

__host__ std::tuple< dim3, dim3, int >
mf_blk_t::resource( const int & n_th_lim ) const
{
    const int n = c.n_cell > 0 ? c.n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
blk_blk_t::resource( const int & n_th_lim ) const
{
    const int n = c.n_cell > 0 ? c.n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
resid_fix_t::resource( const int & n_th_lim ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
gather_a_t::resource( const int & n_th_lim ) const
{
    const int n = c.n_edge > 0 ? c.n_edge : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
expand_a_t::resource( const int & n_th_lim ) const
{
    const int n = c.n_edge > 0 ? c.n_edge : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ int * mf_t::_up_int
( const std::vector< int > & v ) const
{
    if( v.empty(  ) )
    {
        return nullptr;
    }
    int * p = ( int * ) dev->f_malloc( sizeof( int ) * v.size(  ) );
    dev->f_cp( p, v.data(  ), sizeof( int ) * v.size(  ) );
    return p;
};

__host__ float_t * mf_t::_up_real
( const std::vector< float_t > & v ) const
{
    if( v.empty(  ) )
    {
        return nullptr;
    }
    float_t * p = ( float_t * ) dev->f_malloc( sizeof( float_t ) *
        v.size(  ) );
    dev->f_cp( p, v.data(  ), sizeof( float_t ) * v.size(  ) );
    return p;
};

__host__ float_t * mf_t::_alloc
( const int & n ) const
{
    if( n <= 0 )
    {
        return nullptr;
    }
    return ( float_t * ) dev->f_malloc( sizeof( float_t ) *
        size_t( n ) );
};

////////////////////////////////////////////////////////////
//  The out-of-line pieces ( the cycle and what uses it are
//  mutually recursive ).
__host__ void mf_op_t::apply( const float_t * p_x,
                              float_t * p_y ) const
{
    m->apply( l, p_x, p_y );
    return;
};

__host__ const float_t * mf_pre_t::apply( const float_t * p_r ) const
{
    if( cycle )
    {
        return m->cycle_apply( l, p_r );
    }
    m->jacobi( l, p_r, m->lv[ l ].p_z );
    return m->lv[ l ].p_z;
};

////////////////////////////////////////////////////////////
//  Count,  per compact dof of a level,  how many cells touch
//  it:  the divisor of a block-Jacobi sweep.  A topology
//  invariant ( it does not see dt ),  so it is built once.
__host__ std::vector< int > mf_cnt
( const cplx_t & c,  const std::vector< int > & keep,
  const std::vector< int > & comp,  const int & n_compact )
{
    std::vector< int > cnt( size_t( n_compact ), 0 );
    for( int ic = 0; ic < c.n_cell; ++ ic )
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int ge[ 12 ];
        c.cell_edges( i, ge );
        for( int e = 0; e < 12; ++ e )
        {
            const int g = ge[ e ];
            if( keep[ g ] != 0 && comp[ g ] >= 0 )
            {
                cnt[ size_t( comp[ g ] ) ] += 1;
            }
        }
    }
    return cnt;
};

////////////////////////////////////////////////////////////
//  The aggregation plan of level `l`,  read off the P CSR of
//  level `l - 1`.
//
//  Everything is derived from the ACTUAL prolongation rows,
//  not from a second reading of the weight rules:  a
//  re-derivation could disagree with P and the block structure
//  would be wrong where nothing looks.
__host__ void mf_t::_plan
( const int & l,  const std::vector< int > & keep_f,
  const std::vector< int > & comp_f,  const cplx_t & cf,
  const std::vector< int > & comp_c,  const csr_t & P )
{
    const cplx_t & cc = lv[ l ].c;
    int ratio[ 3 ];
    for( int a = 0; a < 3; ++ a )
    {
        //  n,  not NN:  a physical wall gives NN = n + 1 at
        //  both levels,  so the 2:1 relation only holds for the
        //  cell counts ( the extra node is the wall ).
        if( cc.n[ a ] == cf.n[ a ] )
        {
            ratio[ a ] = 1;
        }
        else if( cc.n[ a ] * 2 == cf.n[ a ] )
        {
            ratio[ a ] = 2;
        }
        else
        {
            throw std::runtime_error( "mf_t: the two levels are not 2:1" );
        }
    }
    const int n_cf = int( cf.n_cell );
    const int n_cc = int( cc.n_cell );
    if( int( P.n ) != int( lv[ l - 1 ].n_compact ) )
    {
        throw std::runtime_error
            ( "mf_t: the prolongation has the wrong number of rows" );
    }
    std::vector< int > par( size_t( n_cf ), 0 );
    std::vector< int > nch( size_t( n_cc ) + 1, 0 );
    for( int ic = 0; ic < n_cf; ++ ic )
    {
        const idx_t i( ic % int( cf.n[ 0 ] ),
                      ( ic / int( cf.n[ 0 ] ) ) % int( cf.n[ 1 ] ),
                       ic / int( cf.n[ 0 ] * cf.n[ 1 ] ) );
        idx_t pc( 0, 0, 0 );
        for( int a = 0; a < 3; ++ a )
        {
            pc[ a ] = i[ a ] / ratio[ a ];
        }
        const int p = int( pc[ 0 ] + cc.n[ 0 ] *
            ( pc[ 1 ] + cc.n[ 1 ] * pc[ 2 ] ) );
        par[ size_t( ic ) ] = p;
        nch[ size_t( p + 1 ) ] += 1;
    }
    for( int c = 0; c < n_cc; ++ c )
    {
        nch[ size_t( c + 1 ) ] += nch[ size_t( c ) ];
    }
    std::vector< int > child( size_t( n_cf ), 0 );
    std::vector< int > slot( nch.begin(  ), nch.end(  ) - 1 );
    //  The plan rows are indexed by the SLOT ( the children are
    //  grouped by their parent,  so the slot order is not the
    //  cell order ).  `pos` maps a fine cell to its slot.
    std::vector< int > pos( size_t( n_cf ), 0 );
    for( int ic = 0; ic < n_cf; ++ ic )
    {
        const int p = par[ size_t( ic ) ];
        const int s = slot[ size_t( p ) ] ++;
        child[ size_t( s ) ] = ic;
        pos[ size_t( ic ) ] = s;
    }
    //  per ( child, local edge ) at most two entries
    std::vector< int >      cidx( size_t( n_cf ) * 48, -1 );
    std::vector< float_t >     w( size_t( n_cf ) * 48, float_t( 0. ) );
    std::vector< int >        ns( size_t( n_cf ) * 12, 0 );
    for( int ic = 0; ic < n_cf; ++ ic )
    {
        const int p = par[ size_t( ic ) ];
        idx_t ip( p % int( cc.n[ 0 ] ),
                 ( p / int( cc.n[ 0 ] ) ) % int( cc.n[ 1 ] ),
                  p / int( cc.n[ 0 ] * cc.n[ 1 ] ) );
        int ge_c[ 12 ];
        cc.cell_edges( ip, ge_c );
        int col[ 12 ];
        for( int E = 0; E < 12; ++ E )
        {
            col[ E ] = comp_c[ size_t( ge_c[ E ] ) ];
        }
        const idx_t ii( ic % int( cf.n[ 0 ] ),
                       ( ic / int( cf.n[ 0 ] ) ) % int( cf.n[ 1 ] ),
                        ic / int( cf.n[ 0 ] * cf.n[ 1 ] ) );
        int ge_f[ 12 ];
        cf.cell_edges( ii, ge_f );
        for( int e = 0; e < 12; ++ e )
        {
            const int g = ge_f[ e ];
            const int q = comp_f[ size_t( g ) ];
            if( q < 0 || keep_f[ size_t( g ) ] == 0 )
            {
                continue;
            }
            for( int z = P.ptr[ size_t( q ) ]; z < P.ptr[ size_t( q + 1 ) ];
                 ++ z )
            {
                const int cq = P.idx[ size_t( z ) ];
                int E = -1;
                for( int t = 0; t < 12; ++ t )
                {
                    if( col[ t ] == cq )
                    {
                        E = t;
                        break;
                    }
                }
                if( E < 0 )
                {
                    throw std::runtime_error( "mf_t: a prolongation row leaves "
                        "its parent cell -- the per-cell block structure the "
                        "device hierarchy assumes is broken" );
                }
                const size_t base = size_t( pos[ size_t( ic ) ] ) * 12
                                  + size_t( e );
                const int k = ns[ base ];
                if( k > 3 )
                {
                    throw std::runtime_error( "mf_t: a prolongation row of a "
                        "cell has more than four entries" );
                }
                cidx[ base * 4 + size_t( k ) ]  = E;
                w[ base * 4 + size_t( k ) ]     = P.val[ size_t( z ) ];
                ns[ base ] = k + 1;
            }
        }
    }
    mf_agg_t & ag = lv[ l ].agg;
    ag.n_child = n_cf;
    ag.p_start = _up_int( nch );
    ag.p_child = _up_int( child );
    ag.p_cidx  = _up_int( cidx );
    ag.p_w     = _up_real( w );
    return;
};

////////////////////////////////////////////////////////////
//  build
__host__ void mf_t::build
( device::base_t & dev_,  const hier_t & h,  const int & n_th_lim_ )
{
    if( dev != nullptr )
    {
        throw std::runtime_error( "mf_t::build: already built" );
    }
    dev = & dev_;
    n_th_lim = n_th_lim_;
    n_lvl = int( h.L.size(  ) );
    if( n_lvl < 1 )
    {
        throw std::runtime_error( "mf_t::build: the host hierarchy is empty" );
    }
    lv.resize( size_t( n_lvl ) );
    dev->sync_all_streams(  );
    int n_max = 0;
    for( int l = 0; l < n_lvl; ++ l )
    {
        mf_lvl_t & lv_ = lv[ size_t( l ) ];
        const mg_lvl_t & gl = h.L[ size_t( l ) ];
        if( gl.keep.size(  ) != size_t( gl.c.n_edge ) ||
            gl.comp.size(  ) != size_t( gl.c.n_edge ) )
        {
            throw std::runtime_error( "mf_t::build: the level numbering does "
                "not cover every edge" );
        }
        lv_.c = gl.c;
        lv_.n_compact = gl.n_compact;
        n_max = std::max( n_max, gl.n_compact );
        lv_.p_keep = _up_int( gl.keep );
        lv_.p_comp = _up_int( gl.comp );
        lv_.p_cnt  = _up_int( mf_cnt( gl.c, gl.keep, gl.comp,
                                     gl.n_compact ) );
        lv_.p_x = _alloc( gl.n_compact );
        lv_.p_b = _alloc( gl.n_compact );
        lv_.p_r = _alloc( gl.n_compact );
        lv_.p_z = _alloc( gl.n_compact );
        lv_.p_w = _alloc( gl.n_compact );
        lv_.p_d = _alloc( gl.n_compact );
        lv_.p_inv = _alloc( gl.n_compact );
    }
    n_edge0 = int( lv[ 0 ].c.n_edge );
    p_xraw0 = _alloc( n_edge0 );
    p_part = _alloc( n_blk );
    for( int i = 0; i < 2; ++ i )
    {
        //  Eight contiguous floats per depth,  so the ONE host
        //  read of an iteration is a single 8-scalar copy.
        p_sc[ i ][ 0 ] = _alloc( 8 );
        for( int j = 1; j < 8; ++ j )
        {
            p_sc[ i ][ j ] = p_sc[ i ][ 0 ] + j;
        }
    }
    for( int l = 0; l < n_lvl; ++ l )
    {
        lv[ size_t( l ) ].p_lam = _alloc( 1 );
        lv[ size_t( l ) ].p_om  = _alloc( 1 );
    }
    for( int i = 0; i < 2; ++ i )
    {
        for( int j = 0; j < 3; ++ j )
        {
            p_scr[ i ][ j ] = _alloc( n_max );
        }
    }
    //  level > 0:  blocks + the plan that builds them
    for( int l = 1; l < n_lvl; ++ l )
    {
        mf_lvl_t & lv_ = lv[ size_t( l ) ];
        lv_.p_ablk = _alloc( int( lv_.c.n_cell ) * 144 );
        _plan( l, h.L[ size_t( l - 1 ) ].keep, h.L[ size_t( l - 1 ) ].comp,
               h.L[ size_t( l - 1 ) ].c, h.L[ size_t( l ) ].comp,
               h.L[ size_t( l - 1 ) ].P );
    }
    //  prolongation operators,  fine rows x coarse cols
    for( int l = 0; l + 1 < n_lvl; ++ l )
    {
        upload_csr( *( dev ), h.L[ size_t( l ) ].P, lv[ size_t( l ) ].P );
    }
    //  The outer iterate starts at zero,  so the first step is a
    //  cold solve;  every later step warm-starts from it.
    for( int l = 0; l < n_lvl; ++ l )
    {
        dev->f_mset( lv[ size_t( l ) ].p_x, 0,
                     sizeof( float_t ) * size_t( lv[ size_t( l ) ].n_compact ),
                     nullptr );
    }
    dev->sync_all_streams(  );
    return;
};

////////////////////////////////////////////////////////////
//  free
__host__ void mf_t::free(  )
{
    if( dev == nullptr )
    {
        return;
    }
    for( int l = 0; l < n_lvl; ++ l )
    {
        mf_lvl_t & lv_ = lv[ size_t( l ) ];
        int * pi[ 3 ] = { lv_.p_keep, lv_.p_comp, lv_.p_cnt };
        for( int i = 0; i < 3; ++ i )
        {
            if( pi[ i ] != nullptr )
            {
                dev->f_free( pi[ i ] );
            }
        }
        if( lv_.p_ablk != nullptr )
        {
            dev->f_free( lv_.p_ablk );
        }
        float_t * pf[ 7 ] = { lv_.p_x, lv_.p_b, lv_.p_r, lv_.p_z,
                              lv_.p_w, lv_.p_d, lv_.p_inv };
        for( int i = 0; i < 7; ++ i )
        {
            if( pf[ i ] != nullptr )
            {
                dev->f_free( pf[ i ] );
            }
        }
        mf_agg_t & ag = lv_.agg;
        int * pa[ 4 ] = { ag.p_start, ag.p_child, ag.p_cidx, nullptr };
        for( int i = 0; i < 3; ++ i )
        {
            if( pa[ i ] != nullptr )
            {
                dev->f_free( pa[ i ] );
            }
        }
        if( ag.p_w != nullptr )
        {
            dev->f_free( ag.p_w );
        }
        if( lv_.p_lam != nullptr )
        {
            dev->f_free( lv_.p_lam );
            lv_.p_lam = nullptr;
        }
        if( lv_.p_om != nullptr )
        {
            dev->f_free( lv_.p_om );
            lv_.p_om = nullptr;
        }
        free_csr( *( dev ), lv_.P );
    }
    float_t * pt[ 7 ] = { p_part, p_xraw0 };
    for( int i = 0; i < 2; ++ i )
    {
        if( pt[ i ] != nullptr )
        {
            dev->f_free( pt[ i ] );
        }
    }
    for( int i = 0; i < 2; ++ i )
    {
        for( int j = 0; j < 3; ++ j )
        {
            if( p_scr[ i ][ j ] != nullptr )
            {
                dev->f_free( p_scr[ i ][ j ] );
                p_scr[ i ][ j ] = nullptr;
            }
        }
        if( p_sc[ i ][ 0 ] != nullptr )
        {
            dev->f_free( p_sc[ i ][ 0 ] );
        }
        for( int j = 0; j < 8; ++ j )
        {
            p_sc[ i ][ j ] = nullptr;
        }
    }
    lv.clear(  );
    n_lvl = 0;
    dev = nullptr;
    return;
};

////////////////////////////////////////////////////////////
//  refill:  the only per-step work of the hierarchy itself.
//
//  Level 1 is aggregated straight from the per-step cell
//  blocks ( which are already on the device ),  and every
//  higher level from the level below,  so the whole coarse
//  side of a step is `n_lvl - 1` deterministic launches and
//  not one transfer.
__host__ void mf_t::refill( const cell_blk_t * p_blk,
                            const float_t & dt_ )
{
    p_blk0 = p_blk;
    dt = dt_;
    for( int l = 1; l < n_lvl; ++ l )
    {
        blk_agg_t op;
        op.cc = lv[ size_t( l ) ].c;
        if( l == 1 )
        {
            op.p_blk = p_blk;
            op.dt = dt;
        }
        else
        {
            op.p_ab = lv[ size_t( l - 1 ) ].p_ablk;
        }
        const mf_agg_t & ag = lv[ size_t( l ) ].agg;
        op.p_start = ag.p_start;
        op.p_child = ag.p_child;
        op.p_cidx = ag.p_cidx;
        op.p_w = ag.p_w;
        op.p_out = lv[ size_t( l ) ].p_ablk;
        op.n_cell = int( op.cc.n_cell );
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( blk_agg_k< blk_agg_t >, n_bl, n_th, s_sh, nullptr,
                     op, op.n_cell );
    }
    dev->sync_all_streams(  );
    refill_diag(  );
    lam_est(  );
    return;
};

////////////////////////////////////////////////////////////
//  level 0 vector plumbing
__host__ void mf_t::gather( const dat_3d_t< float_t > & a,
                            float_t * p_x ) const
{
    gather_a_t op;
    op.c      = lv[ 0 ].c;
    op.p_keep = lv[ 0 ].p_keep;
    op.p_comp = lv[ 0 ].p_comp;
    op.a      = a;
    op.p_x    = p_x;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
    dev->launch( gather_a_k< gather_a_t >, n_bl, n_th, s_sh, nullptr,
                 op, n_edge0 );
    dev->sync_all_streams(  );
    return;
};

__host__ void mf_t::expand( const dat_3d_t< float_t > & a,
                            const float_t * p_x,
                            float_t * p_raw ) const
{
    expand_a_t op;
    op.c      = lv[ 0 ].c;
    op.p_comp = lv[ 0 ].p_comp;
    op.p_x    = p_x;
    op.a      = a;
    op.p_raw  = p_raw;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
    dev->launch( expand_a_k< expand_a_t >, n_bl, n_th, s_sh, nullptr,
                 op, n_edge0 );
    dev->sync_all_streams(  );
    return;
};

////////////////////////////////////////////////////////////
//  apply / smoother / residual,  any level
__host__ void mf_t::apply_dt( const int & l,  const float_t * p_x,
                              float_t * p_y,
                              const float_t & dt_ ) const
{
    if( l == 0 )
    {
        mf_edge_t op;
        op.c      = lv[ 0 ].c;
        op.p_blk  = p_blk0;
        op.p_keep = lv[ 0 ].p_keep;
        op.p_comp = lv[ 0 ].p_comp;
        op.p_x    = p_x;
        op.p_y    = p_y;
        op.dt     = dt_;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( mf_edge_k< mf_edge_t >, n_bl, n_th, s_sh, nullptr,
                     op, int( op.c.n_cell ) );
    }
    else
    {
        mf_blk_t op;
        op.c      = lv[ size_t( l ) ].c;
        op.p_ab   = lv[ size_t( l ) ].p_ablk;
        op.p_keep = lv[ size_t( l ) ].p_keep;
        op.p_comp = lv[ size_t( l ) ].p_comp;
        op.p_x    = p_x;
        op.p_y    = p_y;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( mf_blk_k< mf_blk_t >, n_bl, n_th, s_sh, nullptr,
                     op, int( op.c.n_cell ) );
    }
    dev->sync_all_streams(  );
    return;
};

__host__ void mf_t::apply( const int & l,  const float_t * p_x,
                           float_t * p_y ) const
{
    apply_dt( l, p_x, p_y, dt );
    return;
};

__host__ void mf_t::rhs_m1( const float_t * p_a,
                            float_t * p_b ) const
{
    dev->f_mset( p_b, 0, sizeof( float_t ) * size_t( lv[ 0 ].n_compact ),
                 nullptr );
    apply_dt( 0, p_a, p_b, float_t( 0. ) );
    return;
};

__host__ void mf_t::jacobi( const int & l,  const float_t * p_r,
                            float_t * p_z ) const
{
    const mf_lvl_t & lv_ = lv[ size_t( l ) ];
    dev->f_mset( p_z, 0, sizeof( float_t ) * size_t( lv_.n_compact ),
                 nullptr );
    if( l == 0 )
    {
        blk_solve_t op;
        op.c      = lv_.c;
        op.p_blk  = p_blk0;
        op.p_keep = lv_.p_keep;
        op.p_comp = lv_.p_comp;
        op.p_r    = p_r;
        op.p_x    = p_z;
        op.dt     = dt;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( blk_solve_k< blk_solve_t >, n_bl, n_th, s_sh, nullptr,
                     op, int( op.c.n_cell ) );
    }
    else
    {
        blk_blk_t op;
        op.c      = lv_.c;
        op.p_ab   = lv_.p_ablk;
        op.p_keep = lv_.p_keep;
        op.p_comp = lv_.p_comp;
        op.p_r    = p_r;
        op.p_x    = p_z;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( blk_blk_k< blk_blk_t >, n_bl, n_th, s_sh, nullptr,
                     op, int( op.c.n_cell ) );
    }
    blk_scale_t sc;
    sc.p_cnt = lv_.p_cnt;
    sc.p_x   = p_z;
    sc.n     = lv_.n_compact;
    const auto [ s_bl, s_th, s_sh2 ] = sc.resource( n_th_lim );
    dev->launch( blk_scale_k< blk_scale_t >, s_bl, s_th, s_sh2, nullptr,
                 sc, sc.n );
    dev->sync_all_streams(  );
    return;
};

__host__ void mf_t::resid( const int & l,  const float_t * p_x,
                           const float_t * p_b,
                           float_t * p_r ) const
{
    const mf_lvl_t & lv_ = lv[ size_t( l ) ];
    dev->f_mset( p_r, 0, sizeof( float_t ) * size_t( lv_.n_compact ),
                 nullptr );
    apply( l, p_x, p_r );
    resid_fix_t op;
    op.p_b = p_b;
    op.p_r = p_r;
    op.n   = lv_.n_compact;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
    dev->launch( resid_fix_k< resid_fix_t >, n_bl, n_th, s_sh, nullptr,
                 op, op.n );
    dev->sync_all_streams(  );
    return;
};

__host__ float_t mf_t::_read_res( const int & depth ) const
{
    float_t h;
    dev->cp( & h, p_sc[ size_t( depth ) ][ 7 ], 1 );
    return h;
};

////////////////////////////////////////////////////////////
//  The smoothing:  DAMPED diagonal Jacobi.
//
//  A per-cell 12x12 block over-corrects like h^-2 here,
//  and no fixed damping is stable across meshes.  The
//  diagonal has a mesh-independent lam_max( D^-1 A ) of
//  about 3.8,  so the smoother is  x += omega D^-1 ( b - A x )
//  with  omega = 2 / ( lam_max ( 1 + 1 / 8 ) ):  the low end
//  of the interval is lam_max / 8,  the usual choice for a
//  smoothing ( not a solving ) polynomial.
__host__ void mf_t::refill_diag(  )
{
    for( int l = 0; l < n_lvl; ++ l )
    {
        mf_lvl_t & lv_ = lv[ size_t( l ) ];
        dev->f_mset( lv_.p_d, 0, sizeof( float_t ) * size_t( lv_.n_compact ),
                     nullptr );
        blk_diag_t op;
        op.c = lv_.c;
        if( l == 0 )
        {
            op.p_blk = p_blk0;
            op.dt = dt;
        }
        else
        {
            op.p_ab = lv_.p_ablk;
        }
        op.p_keep = lv_.p_keep;
        op.p_comp = lv_.p_comp;
        op.p_d = lv_.p_d;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( blk_diag_k< blk_diag_t >, n_bl, n_th, s_sh, nullptr,
                     op, int( op.c.n_cell ) );
        diag_inv_t iv;
        iv.p_d = lv_.p_d;
        iv.p_inv = lv_.p_inv;
        iv.n = lv_.n_compact;
        const auto [ i_bl, i_th, i_sh ] = iv.resource( n_th_lim );
        dev->launch( diag_inv_k< diag_inv_t >, i_bl, i_th, i_sh, nullptr,
                     iv, iv.n );
    }
    dev->sync_all_streams(  );
    return;
};

//  lam_max( D^-1 A ) by power iteration.  D^-1 A is a
//  similarity of an SPD matrix,  so its spectrum is real and
//  positive and the Rayleigh quotient converges to the top of
//  it.  TWO details that are not cosmetic:
//
//  ( 1 )  the START VECTOR must excite the top of the
//  spectrum.  The all-ones start is ( nearly ) the smoothest
//  vector there is,  so its power iterates never leave the
//  bottom of the spectrum:  the Rayleigh quotient comes out
//  near lam_min and the damping factor 2/( 1.125 lam ) then
//  over-relaxes -- which is exactly the divergence the
//  N = 128 axis mesh showed.  An alternating start has half
//  its energy in the highest mode and converges in a dozen
//  iterations.
//
//  ( 2 )  the sweep count matters for the same reason:  a
//  power iteration that has not converged UNDER-estimates
//  lam_max,  and an under-estimate is an over-relaxation.
__host__ void mf_t::lam_est(  )
{
    for( int l = 0; l < n_lvl; ++ l )
    {
        mf_lvl_t & lv_ = lv[ size_t( l ) ];
        const int n = lv_.n_compact;
        float_t * p_v = lv_.p_w;
        float_t * p_y = lv_.p_r;
        float_t * p_t = lv_.p_z;
        float_t * p_num = p_sc[ 0 ][ 0 ];
        float_t * p_den = p_sc[ 0 ][ 1 ];
        lam_start_t st;
        st.p_v = p_v;
        st.n   = n;
        const auto [ n_bl, n_th, s_sh ] = st.resource( n_th_lim );
        dev->launch( real_k< lam_start_t >, n_bl, n_th, s_sh, nullptr,
                     st, st.n );
        for( int it = 0; it < 24; ++ it )
        {
            dev->f_mset( p_y, 0, sizeof( float_t ) * size_t( n ), nullptr );
            apply( l, p_v, p_y );
            real_dmul( *( dev ), lv_.p_inv, p_y, p_t, n, n_th_lim );
            real_dot_dev( *( dev ), p_v, p_t, p_part, n, n_blk, p_num );
            real_dot_dev( *( dev ), p_v, p_v, p_part, n, n_blk, p_den );
            lam_div_t op;
            op.p_num = p_num;
            op.p_den = p_den;
            op.p_lam = lv_.p_lam;
            dev->launch( real_one_k< lam_div_t >, dim3( 1 ), dim3( 1 ),
                         0, nullptr, op );
            real_copy( *( dev ), p_t, p_v, n, n_th_lim );
        }
        //  The damping factor is derived from lam on the device,
        //  so a step never has to bring lam back.  The host copy
        //  is refreshed only for the setup-time diagnostic.
        cg_om_t om;
        om.p_lam = lv_.p_lam;
        om.p_om  = lv_.p_om;
        dev->launch( real_one_k< cg_om_t >, dim3( 1 ), dim3( 1 ),
                     0, nullptr, om );
    }
    dev->sync_all_streams(  );
    return;
};

//  x += omega D^-1 ( b - A x ),  repeated n_smooth times.
__host__ void mf_t::smooth( const int & l,  const float_t * p_b,
                            float_t * p_x )
{
    mf_lvl_t & lv_ = lv[ size_t( l ) ];
    const int n = lv_.n_compact;
    //  omega = 2 / ( 1.35 lam )  lives in `lv_.p_om`,  built by
    //  `lam_est` on the device:  the host does not compute it.
    for( int s = 0; s < n_smooth; ++ s )
    {
        resid( l, p_x, p_b, lv_.p_r );
        diag_upd_t op;
        op.p_inv = lv_.p_inv;
        op.p_r = lv_.p_r;
        op.p_z = lv_.p_z;
        op.p_om = lv_.p_om;
        op.n = n;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( diag_upd_k< diag_upd_t >, n_bl, n_th, s_sh, nullptr,
                     op, op.n );
        real_axpy( *( dev ), lv_.p_z, p_x, float_t( 1. ), n, n_th_lim );
    }
    dev->sync_all_streams(  );
    return;
};

////////////////////////////////////////////////////////////
//  The V-cycle,  in place on  `lv[l].p_x`  writing from
//  `lv[l].p_b`.
//
//  smooth ( n_smooth block-Jacobi )  ->  r = b - A x  ->
//  b_c = P^T r  ->  recurse  ->  x += P x_c  ->  smooth.
//
//  The coarsest level is solved to a tight tolerance by the
//  same device PCG with the block-Jacobi preconditioner.
//  A dense coarsest factorisation would be faster per solve but
//  would put the coarsest solve back on the bus;  and a
//  preconditioner only sets the rate,  never the answer,  so
//  the outer solve is unaffected by the choice.
__host__ void mf_t::vcycle( const int & l,  float_t * p_x )
{
    mf_lvl_t & lv_ = lv[ size_t( l ) ];
    if( l == 0 )
    {
        ++ n_vcyc;
    }
    if( l + 1 >= n_lvl )
    {
        const pcg_stat_t cst = pcg( l, false, p_x, lv_.p_b, p_scr[ 1 ][ 0 ],
                                    p_scr[ 1 ][ 1 ], p_scr[ 1 ][ 2 ],
                                    coarse_tol, coarse_iter, false, 1 );
        n_coarse_it += cst.it;
        return;
    }
    for( int s = 0; s < n_smooth; ++ s )
    {
        smooth( l, lv_.p_b, p_x );
    }
    resid( l, p_x, lv_.p_b, lv_.p_r );
    {
        mf_lvl_t & lc = lv[ size_t( l + 1 ) ];
        dev->f_mset( lc.p_b, 0, sizeof( float_t ) * size_t( lc.n_compact ),
                     nullptr );
        matvec_t_t op;
        op.S   = lv[ l ].P;
        op.p_x = lv_.p_r;
        op.p_y = lc.p_b;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( matvec_t_k< matvec_t_t >, n_bl, n_th, s_sh, nullptr,
                     op, op.S.n );
        dev->f_mset( lc.p_w, 0, sizeof( float_t ) * size_t( lc.n_compact ),
                     nullptr );
        dev->sync_all_streams(  );
    }
    vcycle( l + 1, lv[ size_t( l + 1 ) ].p_w );
    {
        matvec_add_t op;
        op.S   = lv_.P;
        op.p_x = lv[ size_t( l + 1 ) ].p_w;
        op.p_y = p_x;
        const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
        dev->launch( matvec_k< matvec_add_t >, n_bl, n_th, s_sh, nullptr,
                     op, op.S.n );
    }
    for( int s = 0; s < n_smooth; ++ s )
    {
        smooth( l, lv_.p_b, p_x );
    }
    dev->sync_all_streams(  );
    return;
};

__host__ const float_t * mf_t::cycle_apply( const int & l,
                                             const float_t * p_r )
{
    mf_lvl_t & lv_ = lv[ size_t( l ) ];
    real_copy( *( dev ), p_r, lv_.p_b, lv_.n_compact, n_th_lim );
    dev->f_mset( lv_.p_w, 0, sizeof( float_t ) * size_t( lv_.n_compact ),
                 nullptr );
    vcycle( l, lv_.p_w );
    return lv_.p_w;
};

////////////////////////////////////////////////////////////
//  The device PCG.
//
//  Same reductions,  same residual definition and same break
//  on a non-positive curvature as the face solve,  with two
//  things that matter here:
//  the work vectors are OWNED by the hierarchy instead of
//  being allocated per solve,  and `warm` decides whether
//  `x` starts from the caller's value.  A cold start is what
//  makes a preconditioner's non-linearity harmless,  so the
//  coarse solve is always cold;  the outer solve is warm
//  because a step's previous solution is the best guess it
//  will ever have.
__host__ pcg_stat_t mf_t::pcg
( const int & l,  const bool & cycle,  float_t * p_x,
  const float_t * p_b,  float_t * p_r,  float_t * p_p,
  float_t * p_ap,  const float_t & tol,  const int & it_max,
  const bool & warm,  const int & depth )
{
    //
    //  Every scalar of the iteration is computed ON THE DEVICE
    //  ( one-thread kernels:  alpha,  beta,  the residual,  the
    //  b2 guard ) and every vector update reads its coefficient
    //  from the device as well.  The host's part of an iteration
    //  is exactly one 8-scalar copy -- `res` for the loop test
    //  and `ok` for the curvature breakdown -- plus the launches.
    //  That is the O(1)-scalar shape a step is allowed to have,
    //  and the partials are still summed in ORDER,  so the
    //  result does not depend on the thread schedule.
    //
    pcg_stat_t st;
    const int n = lv[ size_t( l ) ].n_compact;
    float_t * p_rz  = p_sc[ size_t( depth ) ][ 0 ];
    float_t * p_rn  = p_sc[ size_t( depth ) ][ 1 ];
    float_t * p_al  = p_sc[ size_t( depth ) ][ 2 ];
    float_t * p_be  = p_sc[ size_t( depth ) ][ 3 ];
    float_t * p_rr  = p_sc[ size_t( depth ) ][ 4 ];
    float_t * p_b2s = p_sc[ size_t( depth ) ][ 5 ];
    float_t * p_ok  = p_sc[ size_t( depth ) ][ 6 ];
    float_t * p_res = p_sc[ size_t( depth ) ][ 7 ];
    if( !warm )
    {
        dev->f_mset( p_x, 0, sizeof( float_t ) * size_t( n ), nullptr );
    }
    resid( l, p_x, p_b, p_r );
    real_dot_dev( *( dev ), p_b, p_b, p_part, n, n_blk, p_b2s );
    {
        cg_b2s_t op;
        op.p_b2  = p_b2s;
        op.p_b2s = p_b2s;
        dev->launch( real_one_k< cg_b2s_t >, dim3( 1 ), dim3( 1 ),
                     0, nullptr, op );
    }
    mf_pre_t pre;
    pre.dev = dev;
    pre.m = this;
    pre.l = l;
    pre.cycle = cycle;
    const float_t * z = pre.apply( p_r );
    real_dot_dev( *( dev ), p_r, z, p_part, n, n_blk, p_rz );
    real_copy( *( dev ), z, p_p, n, n_th_lim );
    real_dot_dev( *( dev ), p_r, p_r, p_part, n, n_blk, p_rr );
    {
        cg_res_t op;
        op.p_rr  = p_rr;
        op.p_b2s = p_b2s;
        op.p_res = p_res;
        dev->launch( real_one_k< cg_res_t >, dim3( 1 ), dim3( 1 ),
                     0, nullptr, op );
    }
    st.n_red = 2;
    st.res = _read_res( depth );
    while( st.res > tol && st.it < it_max )
    {
        dev->f_mset( p_ap, 0, sizeof( float_t ) * size_t( n ), nullptr );
        apply( l, p_p, p_ap );
        //  pAp goes into the `rr` slot:  rr is recomputed from
        //  scratch at the end of the iteration anyway.
        real_dot_dev( *( dev ), p_p, p_ap, p_part, n, n_blk, p_rr );
        {
            cg_alpha_t op;
            op.p_rz  = p_rz;
            op.p_pap = p_rr;
            op.p_al  = p_al;
            op.p_ok  = p_ok;
            dev->launch( real_one_k< cg_alpha_t >, dim3( 1 ), dim3( 1 ),
                         0, nullptr, op );
        }
        real_axpy_dev( *( dev ), p_p,  p_x, p_al, float_t(  1. ), n,
                       n_th_lim, p_ok );
        //  On a breakdown ( `ok` clear ) nothing moves:  alpha is
        //  zero,  and the guard makes even that a no-op,  so the
        //  iteration stops on a clean state instead of on NaN.
        real_axpy_dev( *( dev ), p_ap, p_r, p_al, float_t( -1. ), n,
                       n_th_lim, p_ok );
        z = pre.apply( p_r );
        real_dot_dev( *( dev ), p_r, z, p_part, n, n_blk, p_rn );
        {
            cg_beta_t op;
            op.p_rn = p_rn;
            op.p_rz = p_rz;
            op.p_be = p_be;
            dev->launch( real_one_k< cg_beta_t >, dim3( 1 ), dim3( 1 ),
                         0, nullptr, op );
        }
        real_xpay_dev( *( dev ), z, p_p, p_be, n, n_th_lim, p_ok );
        real_dot_dev( *( dev ), p_r, p_r, p_part, n, n_blk, p_rr );
        {
            cg_res_t op;
            op.p_rr  = p_rr;
            op.p_b2s = p_b2s;
            op.p_res = p_res;
            dev->launch( real_one_k< cg_res_t >, dim3( 1 ), dim3( 1 ),
                         0, nullptr, op );
        }
        ++ st.it;
        st.n_red += 2;
        //  THE host read of the iteration.
        float_t h[ 8 ];
        dev->cp( h, p_sc[ size_t( depth ) ][ 0 ], 8 );
        st.res = h[ 7 ];
        if( std::getenv( "MF_PCG_TRACE" ) != nullptr )
        {
            std::printf( "CXX pcg-trace l %d d %d it %d res %.3e\n",
                         l, depth, st.it, double( st.res ) );
            std::fflush( stdout );
        }
        if( !( h[ 6 ] > float_t( 0. ) ) )
        {
            //  A non-positive curvature:  the iteration cannot
            //  go on.  alpha was set to zero,  so nothing moved.
            break;
        }
    }
    return st;
};

__host__ pcg_stat_t mf_t::solve( float_t * p_x,  const float_t * p_b,
                                 const float_t & tol,
                                 const int & it_max,
                                 const bool & warm )
{
    n_vcyc      = 0;
    n_coarse_it = 0;
    const pcg_stat_t st = pcg( 0, true, p_x, p_b, p_scr[ 0 ][ 0 ],
                               p_scr[ 0 ][ 1 ], p_scr[ 0 ][ 2 ], tol,
                               it_max, warm, 0 );
    if( std::getenv( "MF_COARSE_STATS" ) != nullptr )
    {
        std::printf( "CXX mf-coarse:  %lld V-cycles,  %lld coarsest PCG "
                     "iterations ( %.2f per cycle ),  outer it %d\n",
                     n_vcyc, n_coarse_it,
                     n_vcyc > 0 ? double( n_coarse_it ) / double( n_vcyc )
                                : 0.,
                     st.it );
        std::fflush( stdout );
    }
    return st;
};

};    // namespace mhd_ee::ni::edge_mg
