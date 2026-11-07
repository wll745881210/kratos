#pragma once

#include "../../../utilities/functions.h"
#include "comm.h"

namespace multigrid::boundary
{
////////////////////////////////////////////////////////////
// Job dispatching and resources

template < class prx_T, class derived_T > __device__ 
bool comm_t< prx_T, derived_T >::dispatch
( idx_t & i, const idx_t & n_ceff ) const
{
    utils::divmod( i[ axis[ 2 ] ], i[ axis[ 1 ] ] ,
                   utils:: th_id(  ), n_ceff[ axis[ 1 ] ] );
    return   i[ axis[ 2 ] ] < n_ceff[ axis[ 2 ] ] ;
}

////////////////////////////////////////////////////////////
// Actions of communication boundary

template< class prx_T , class derived_T >
template< class bdt_T > __device__ void comm_t
< prx_T, derived_T >::to_same( const bdt_T & b_d ) const
{
    idx_t i_m, i_b;
    if( ! get_self(  ).dispatch( i_b, n_ceff_s ) )
        return;
    for( int a = 0  ; a  < i_m. size(  ); ++ a )
    {
        i_m[ a ] = i_b [ a ]  + offset_s[ a ] ;
        i_b[ a ] = i_b [ a ]  + offset_c[ a ] ;
    }

    for(  int k = 0; k < order; ++ k )
    {
        i_m[ axis[ 0 ] ]  = k + offset_s[ axis[ 0 ] ];
        i_b[ axis[ 0 ] ]  = k + offset_c[ axis[ 0 ] ];
        auto       * p_buff = b_send     .at( i_b );
        const auto * p_mesh = b_d.mgd.fld.at( i_m );
        for( int n = 0; n < n_fld; ++ n )
            p_buff[ n ] = p_mesh[ n ];
        if( just_restrict )
            for( int n = n_fld; n < b_send.n_int; ++ n )
                p_buff[ n ] = p_mesh[ n - n_fld ];
    }
    return;
}

template< class prx_T, class derived_T >
template< class bdt_T > __device__ void comm_t
< prx_T, derived_T >::to_finer ( const bdt_T & bdt ) const
{
    idx_t i_b, i_m, i_m0, di;
    if( ! get_self(  ).dispatch( i_m0, n_ceff_s ) )
        return;
    __dyn_shared__ ( f_t, var_sh );
    f_t * q_c  = var_sh + threadIdx.x * n_fld;
    const int order_2( ( order + ( order & 1 ) ) / 2 );
    for ( int k = 0; k < order_2; ++ k )
    {
        i_m0[ axis[ 0 ] ] =   k / 2  ;
        di  [ axis[ 0 ] ] = ( k & 1 );
        if( ( k & 1 ) ==  0 )
        {
            for( int a = 0; a < bdt.geo.n_dim; ++ a )
                i_m[ a ] =  i_m0[ a ] + offset_s[ a ];
            const auto * p  = bdt.mgd . fld.at( i_m );
            for( int n = 0; n < n_fld ; ++ n )
                q_c[ n ] = p[ n ];
        }        
        for( int l = 0 ; l < m_dim ; ++ l )
        {
            utils::muldim( di, l, bdt.geo.n_dim );
            for( int a = 0; a < bdt.geo.n_dim; ++ a )
                i_b[ a ] = 2 * i_m0[ a ]
                         + offset_c[ a ] + di[ a ];
            int j = i_b[ axis[ 0 ] ] + same_rank * order;
            if( j < 0 || j >= b_send.n_cell[ axis[ 0 ] ] )
                continue;            
            auto * q = b_send.at( i_b ) ;
            for( int n = 0 ; n < n_fld; ++ n )
            {
                q[ n ] = q_c[ n ];
                if( just_restrict && b_send.n_int > n_fld )
                    q[ n_fld + n ] = q_c[ n ];
            }
        }
    }
    return;
}

template< class prx_T, class derived_T >
template< class bdt_T > __device__ void comm_t
< prx_T, derived_T >::to_coarser
( const bdt_T & bdt ) const
{
    idx_t i_b, i_m, i0, di;
    if( ! get_self(  ).dispatch( i0, n_ceff_s ) )
        return;

    __dyn_shared__ ( f_t, var_sh );
    auto * q = var_sh + n_fld * threadIdx.x;
    for( int a = 0; a < 3; ++ a )
    {
        i_m[ a ]  = 2 * i0[ a ] + offset_s[ a ];
        i_b[ a ]  =     i0[ a ] + offset_c[ a ];
    }
    f_t norm ;
    for( int n = 0; n < n_fld; ++ n )
        q[ n ] = 0;
    for( int k = 0; k < order; ++ k )
    {
        i_b[ axis[ 0 ] ] = k + offset_c[ axis[ 0 ] ] ;
        i_m[ axis[ 0 ] ] = 2 * ( k + i0[ axis[ 0 ] ] )
                             + offset_s[ axis[ 0 ] ] ;
        norm = 0 ;
        for( int l = 0; l < m_dim; ++ l )
        {
            utils::muldim( di, l, bdt.geo.n_dim );
            for( int a = 0; a < 3; ++ a )
                i_m[ a ] += di[ a ];
            const auto  dv = bdt.geo.volume( i_m );
            const auto * p = bdt.mgd.fld.at( i_m );
            norm += dv ;
            for( int n = 0 ; n < n_fld; ++ n )
                q[ n ] += p[ n ] * dv;

            for( int a = 0; a < bdt.geo.n_dim; ++ a )
                i_m[ a ] -= di[ a ];
        }
        auto * p_buff = b_send.at ( i_b );
        for( int n = 0; n < n_fld; ++ n )
            p_buff[ n ] = q [ n ] / norm;
        if( just_restrict )
            for( int n = n_fld; n < b_send.n_int; ++ n )
                p_buff[ n ] = q[ n - n_fld ] / norm;
    }
    return;
}

////////////////////////////////////////////////////////////
// Unpack from the receiving buffer

template< class prx_T , class derived_T >
template< class bdt_T > __device__ void comm_t
< prx_T, derived_T >::from_buf( const bdt_T & b_d ) const
{
    idx_t i_m, i_b;
    if( ! get_self(  ).dispatch( i_b, n_ceff_r ) )
        return ;
    
    for( int a = 0; a < i_m.size(  ); ++ a )
        i_m[ a ] =  i_b[ a ] + offset_r[ a ];
    for( int k = 0; k < order; ++ k )
    {
        i_m[ axis[ 0 ] ]    = k + offset_r[ axis[ 0 ] ];
        i_b[ axis[ 0 ] ]    = k ;
        auto * p_buff = b_recv     .at ( i_b );
        auto * p_mesh = b_d.mgd.fld.at ( i_m );
        for( int n = 0; n < n_fld; ++ n )
            p_mesh[ n ] = p_buff[ n ];
        if( just_restrict )        
            for( int n = n_fld; n< b_d.mgd.fld.n_int; ++ n )
                p_mesh[ n ] = p_buff[ n - n_fld ];
    }
    return;
}

};                   // namespace multigrid::boundary
