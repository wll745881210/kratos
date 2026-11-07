#pragma once

#include "../../../utilities/functions.h"
#include "comm.h"

namespace multigrid::boundary
{
////////////////////////////////////////////////////////////
// Packing

template< class prx_T, class derived_T >
template< class bdt_T > __device__
void comm_t< prx_T, derived_T >::to_finer_finest
( const bdt_T & bdt ) const
{
    idx_t i_b, i_m, i_m0, di;
    if( ! get_self(  ).dispatch( i_m0, n_ceff_s ) )
        return;

    __dyn_shared__ ( f_t, var_sh );
    f_t * q_c  = var_sh + threadIdx.x * n_fld;

    const int r_dim( utils::shift( 1, bdt.geo.n_dim - 1 ) );
    for ( int k = 0; k < order; ++ k )
    {
        i_m0 [ axis[ 0 ] ] = k / 2;
        di   [ axis[ 0 ] ] = ( k & 1 );

        if( ( k & 1 ) == 0 )
        {
            for( int a = 0; a < bdt.geo.n_dim; ++ a )
                i_m[ a ] = i_m0[ a ] + offset_s [ a ];
            const auto * q = bdt.mgd.fld.at( i_m );
            for( int n = 0; n < n_fld; ++ n )
                q_c[ n ] = q[ n ];
        }
        for( int l = 0 ; l < r_dim ; ++ l )
        {
            utils::didx_refine( di, l, this-> axis );
            for( int a = 0; a < bdt.geo.n_dim; ++ a )
                i_b[ a ] = 2 * i_m0[ a ]
                         + offset_c[ a ] + di[ a ];
            auto * q = b_send.at( i_b ) ;
            for( int n = 0; n < n_fld; ++ n )
                q[ n ] = q_c[ n ];                
        }
    }
    return;
}

////////////////////////////////////////////////////////////
// Un-packing

template< class prx_T , class derived_T >
template< class bdt_T > __device__
void comm_t< prx_T, derived_T >::from_buf_finest
( const bdt_T & bdt ) const
{
    if( d_lvl < 0 )
        return from_buf_finest_refine ( bdt );
    
    idx_t i;
    if( ! get_self(  ).dispatch( i, n_ceff_r ) )
        return;
    i[ axis[ 0 ] ] = blockIdx.y;    
    const auto * p_buff = b_recv     .at( i );
    for( int a = 0; a < bdt.geo.n_dim; ++ a )
        i[ a ] += offset_r[ a ];
    auto       * p_mesh = bdt.mgd.fld.at( i );
    for( int n = 0; n < n_fld; ++ n )
        p_mesh[ n ] = p_buff[ n ];
    return;
}

template< class prx_T , class derived_T >
template< class bdt_T > __device__
void comm_t< prx_T, derived_T >::from_buf_finest_refine
( const bdt_T & bdt ) const
{
    idx_t i;
    if( ! get_self(  ).dispatch( i, n_ceff_r ) )
        return;
    i[ axis[ 0 ] ]   = rhs *
                     ( bdt.geo.n_ceff[ axis[ 0 ] ] - 1 );
    const int     di = rhs * 2 - 1;
    const auto * q_b = bdt.mgd.fld.at( i );
    i[ axis[ 0 ] ]  -= di;
    const auto * q_d = bdt.mgd.fld.at( i );
    i[ axis[ 0 ] ]  += di  * 2;
    auto       * p_m = bdt.mgd.fld.at( i );
    auto       * q_t = p_m;
    if( ! same_rank )
    {
        i [ axis[ 0 ] ]  = 0;
        q_t = b_recv.at( i );
    }
    for( int n = 0; n < n_fld; ++ n )
    {
        const  auto  q_s = q_b[ n ]
            + ( q_b[ n ] - q_d[ n ] ) / 2;
        p_m[ n ] = ( q_s + q_t[ n ] ) / 2;
    }
    return;
}

};                         // namespace multigrid::boundary
