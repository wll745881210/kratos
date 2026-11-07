#include "edge_mg_dev_vec.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

////////////////////////////////////////////////////////////
//  The host half of the scalar device-vector primitives.
//
//  These are the launch wrappers:  each one fills the op
//  object from the header,  takes the geometry from its
//  `resource`,  and launches.  The ops themselves,  and
//  the kernel templates they launch,  stay in the header,
//  because a kernel needs them per type,  not per TU.

namespace mhd_ee::ni::edge_mg
{

__host__ std::tuple< dim3, dim3, int >
real_copy_t::resource( const int & n_th_lim, const int & n ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
real_axpy_t::resource( const int & n_th_lim, const int & n ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
real_xpay_t::resource( const int & n_th_lim, const int & n ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
real_dmul_t::resource( const int & n_th_lim, const int & n ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
real_axpby_t::resource( const int & n_th_lim, const int & n ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ void real_dot_dev
( device::base_t & dev, const float_t * p_x, const float_t * p_y,
  float_t * p_part, const int & n, const int & n_blk,
  float_t * p_out, const int & n_th_lim )
{
    real_dot_part_t op;
    op.p_x       = p_x;
    op.p_y       = p_y;
    op.n         = n;
    op.chunk_len = ( n + n_blk - 1 ) / n_blk;
    op.p_part    = p_part;
    ( void ) n_th_lim;
    dev.launch( real_dot_part_k< real_dot_part_t >, dim3( n_blk ), dim3( 1 ),
                0, nullptr, op, n_blk );
    real_sum_part_t sum;
    sum.p_part = p_part;
    sum.n_blk  = n_blk;
    sum.p_out  = p_out;
    dev.launch( real_one_k< real_sum_part_t >, dim3( 1 ), dim3( 1 ),
                0, nullptr, sum );
    return;
};

__host__ std::tuple< dim3, dim3, int >
real_axpy_dev_t::resource( const int & n_th_lim, const int & n ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
real_xpay_dev_t::resource( const int & n_th_lim, const int & n ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ void real_axpy_dev
( device::base_t & dev, const float_t * p_x, float_t * p_y,
  const float_t * p_a, const float_t & s,
  const int & n, const int & n_th_lim,
  const float_t * p_ok )
{
    real_axpy_dev_t op;
    op.p_x  = p_x;
    op.p_y  = p_y;
    op.p_a  = p_a;
    op.s    = s;
    op.p_ok = p_ok;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim, n );
    dev.launch( real_k< real_axpy_dev_t >, n_bl, n_th, s_sh, nullptr, op, n );
    return;
};

__host__ void real_xpay_dev
( device::base_t & dev, const float_t * p_x, float_t * p_y,
  const float_t * p_be, const int & n, const int & n_th_lim,
  const float_t * p_ok )
{
    real_xpay_dev_t op;
    op.p_x  = p_x;
    op.p_y  = p_y;
    op.p_be = p_be;
    op.p_ok = p_ok;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim, n );
    dev.launch( real_k< real_xpay_dev_t >, n_bl, n_th, s_sh, nullptr, op, n );
    return;
};

__host__ void real_copy
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const int & n,
  const int & n_th_lim )
{
    real_copy_t op;
    op.p_x = p_x;
    op.p_y = p_y;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim, n );
    dev.launch( real_k< real_copy_t >, n_bl, n_th, s_sh, nullptr, op, n );
    return;
};

__host__ void real_axpy
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const float_t & a,
  const int & n, const int & n_th_lim )
{
    real_axpy_t op;
    op.p_x = p_x;
    op.p_y = p_y;
    op.a   = a;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim, n );
    dev.launch( real_k< real_axpy_t >, n_bl, n_th, s_sh, nullptr, op, n );
    return;
};

__host__ void real_xpay
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const float_t & be,
  const int & n, const int & n_th_lim )
{
    real_xpay_t op;
    op.p_x = p_x;
    op.p_y = p_y;
    op.be  = be;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim, n );
    dev.launch( real_k< real_xpay_t >, n_bl, n_th, s_sh, nullptr, op, n );
    return;
};

__host__ void real_dmul
( device::base_t & dev, const float_t * p_a, const float_t *
    p_x,
  float_t * p_y, const int & n, const int & n_th_lim )
{
    real_dmul_t op;
    op.p_a = p_a;
    op.p_x = p_x;
    op.p_y = p_y;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim, n );
    dev.launch( real_k< real_dmul_t >, n_bl, n_th, s_sh, nullptr, op, n );
    return;
};

__host__ void real_axpby
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const float_t & a,
  const float_t & be, const int & n, const int & n_th_lim )
{
    real_axpby_t op;
    op.p_x = p_x;
    op.p_y = p_y;
    op.a   = a;
    op.be  = be;
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim, n );
    dev.launch( real_k< real_axpby_t >, n_bl, n_th, s_sh, nullptr, op, n );
    return;
};

__host__ float_t real_dot
( device::base_t & dev, const float_t * p_x, const float_t *
    p_y,
  float_t * p_part, const int & n, const int & n_blk,
  const int & n_th_lim )
{
    real_dot_part_t op;
    op.p_x       = p_x;
    op.p_y       = p_y;
    op.n         = n;
    op.chunk_len = ( n + n_blk - 1 ) / n_blk;
    op.p_part    = p_part;
    ( void ) n_th_lim;
    //  one block, one thread each: the stage-1 partial sum
    //  is serial per
    //  block,  which is what makes the order fixed
    dev.launch( real_dot_part_k< real_dot_part_t >, dim3( n_blk ), dim3( 1 ),
                0, nullptr, op, n_blk );
    dev.sync_all_streams(  );
    std::vector< float_t > h( n_blk );
    dev.f_cp( h.data(  ), p_part, sizeof( float_t ) * n_blk
        );
    dev.sync_all_streams(  );
    float_t s( 0. );
    for( int b = 0; b < n_blk; ++ b )
    {
        s += h[ b ];
    }
    return s;
};

__host__ void real_dot_batch_ws_t::free( device::base_t & dev )
{
    if( pp_x != nullptr )
    {
        dev.f_free( pp_x );
    }
    if( p_part != nullptr )
    {
        dev.f_free( p_part );
    }
    pp_x = nullptr;
    p_part = nullptr;
    h_part.clear(  );
    n_dot = 0;
    n_cap = 0;
    n_blk = 0;
    return;
};

__host__ void real_dot_batch_ws_t::set_x
( device::base_t & dev,
  const std::vector< const float_t * > & p_x )
{
    if( int( p_x.size(  ) ) != n_cap )
    {
        throw std::invalid_argument
            ( "real_dot_batch: wrong pointer-list"
              " size" );
    }
    if( n_cap > 0 )
    {
        dev.f_cp
            ( pp_x, p_x.data(  ),
              sizeof( float_t * ) * n_cap );
    }
    n_dot = n_cap;
    return;
};

__host__ void real_dot_batch_ws_t::set_active( const int & n_active )
{
    if( n_active < 0 || n_active > n_cap )
    {
        throw std::invalid_argument
            ( "real_dot_batch: invalid active prefix" );
    }
    n_dot = n_active;
    return;
};

__host__ void real_dot_batch_ws_t::init
( device::base_t & dev,
  const std::vector< const float_t * > & p_x,
  const int & n_blk_ )
{
    free( dev );
    if( n_blk_ <= 0 )
    {
        throw std::invalid_argument
            ( "real_dot_batch: n_blk must be"
              " positive" );
    }
    n_cap = int( p_x.size(  ) );
    n_dot = n_cap;
    n_blk = n_blk_;
    if( n_cap == 0 )
    {
        return;
    }
    const size_t n_part = size_t( n_cap ) * n_blk;
    pp_x = ( const float_t ** ) dev.f_malloc
        ( sizeof( float_t * ) * n_cap );
    p_part = ( float_t * ) dev.f_malloc
        ( sizeof( float_t ) * n_part );
    h_part.resize( n_part );
    set_x( dev, p_x );
    return;
};

__host__ real_dot_batch_stat_t real_dot_batch
( device::base_t & dev, real_dot_batch_ws_t & ws,
  const float_t * p_y, const int & n,
  std::vector< float_t > & dots, const int & n_th_lim
      )
{

    real_dot_batch_stat_t st;
    st.n_dot = ws.n_dot;
    dots.resize( ws.n_dot );
    if( ws.n_dot == 0 )
    {
        return st;
    }
    if( n < 0 || ws.n_blk <= 0 || ws.n_dot > ws.n_cap ||
        ws.pp_x == nullptr || ws.p_part == nullptr )
    {
        throw std::invalid_argument
            ( "real_dot_batch: invalid size or workspace" );
    }
    ( void ) n_th_lim;
    real_dot_batch_part_t op;
    op.pp_x      = ws.pp_x;
    op.p_y       = p_y;
    op.n         = n;
    op.chunk_len = ( n + ws.n_blk - 1 ) / ws.n_blk;
    op.n_blk     = ws.n_blk;
    op.p_part    = ws.p_part;
    const int n_task = ws.n_dot * ws.n_blk;
    dev.launch
        ( real_dot_batch_part_k< real_dot_batch_part_t >,
          dim3( n_task ), dim3( 1 ), 0, nullptr,
          op, n_task );
    dev.sync_all_streams(  );
    const size_t n_part = size_t( ws.n_dot ) * ws.n_blk;
    dev.f_cp
        ( ws.h_part.data(  ), ws.p_part,
          sizeof( float_t ) * n_part );
    for( int j = 0; j < ws.n_dot; ++ j )
    {
        const float_t * h = ws.h_part.data(  )
                         + size_t( j ) * ws.n_blk;
        float_t s( 0. );
        for( int b = 0; b < ws.n_blk; ++ b )
        {
            s += h[ b ];
        }
        dots[ j ] = s;
    }
    st.n_comm = 1;
    return st;
};

};    // namespace mhd_ee::ni::edge_mg