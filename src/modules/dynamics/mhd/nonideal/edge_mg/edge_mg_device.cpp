#include "edge_mg_device.h"

#include <algorithm>
#include <tuple>

#include "../../../../../../src/device/device.h"

////////////////////////////////////////////////////////////
//  The host half of the device path of the edge-MG leg.
//
//  `resource()` computes the launch geometry here;  it is
//  host code that never runs on the device.  `upload_csr`
//  and `free_csr` move the assembled operator in and out.
//  The op structs and their `apply`s,  the kernels and the
//  templates all stay in the header.

namespace mhd_ee::ni::edge_mg
{

__host__ std::tuple< dim3, dim3, int >
rhs_m12_t::resource( const int & n_th_lim ) const
{
    const int n = n_cell > 0 ? n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ void upload_csr
( device::base_t & dev, const csr_t & A, dev_csr_t & d )
{
    d.n    = A.n;
    d.ncol = A.ncol;
    d.ptr = ( int    * ) dev.f_malloc( sizeof( int ) * A.ptr.size(  ) );
    d.idx = ( int    * ) dev.f_malloc( sizeof( int ) * A.idx.size(  ) );
    d.val = ( float_t * ) dev.f_malloc( sizeof( float_t ) *
        A.val.size(  ) );
    dev.f_cp( d.ptr, A.ptr.data(  ), sizeof( int ) * A.ptr.size(  ) );
    dev.f_cp( d.idx, A.idx.data(  ), sizeof( int ) * A.idx.size(  ) );
    dev.f_cp( d.val, A.val.data(  ), sizeof( float_t ) *
        A.val.size(  ) );
    dev.sync_all_streams(  );
    return;
};

__host__ void free_csr( device::base_t & dev, dev_csr_t & d )
{
    if( d.ptr != nullptr )
    {
        dev.f_free( d.ptr );
    }
    if( d.idx != nullptr )
    {
        dev.f_free( d.idx );
    }
    if( d.val != nullptr )
    {
        dev.f_free( d.val );
    }
    d = dev_csr_t(  );
    return;
};

__host__ std::tuple< dim3, dim3, int >
matvec_t::resource( const int & n_th_lim ) const
{
    const int n = S.n > 0 ? S.n : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
matvec_t_t::resource( const int & n_th_lim ) const
{
    const int n = S.n > 0 ? S.n : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
mf_edge_t::resource( const int & n_th_lim ) const
{
    const int n = c.n_cell > 0 ? c.n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
blk_solve_t::resource( const int & n_th_lim ) const
{
    const int n = c.n_cell > 0 ? c.n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
blk_scale_t::resource( const int & n_th_lim ) const
{
    const int m = n > 0 ? n : 1;
    const int t = std::min( n_th_lim, m );
    return std::make_tuple( dim3( ( m + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
resid_t::resource( const int & n_th_lim ) const
{
    const int n = S.n > 0 ? S.n : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

__host__ std::tuple< dim3, dim3, int >
matvec_add_t::resource( const int & n_th_lim ) const
{
    const int n = S.n > 0 ? S.n : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ), dim3( t ), 0 );
};

};    // namespace mhd_ee::ni::edge_mg