#pragma once

#include "../../../../src/types.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Local interpolator for coolings

template< class f_T = type::float_t, int hwidth = 2 >
struct intp_t
{
    ////////// Types and Data //////////
    using f_t  = f_T ;
    f_t lnT_ref[ 2 * hwidth + 1 ];
    f_t lnL_ref[ 2 * hwidth + 1 ];

    static constexpr f_t f_min = 1e-38f ;

    template< class g_T > __device__ __forceinline__
    int find_idx( const g_T & lnT ) const
    {
        int i;
        if( lnT < lnT_ref[ 0 ] )
            i = 0;
        else if( lnT > lnT_ref[ 2 * hwidth ] )
            i = 2 * hwidth - 1;
        else
            for( i = 0; i < 2 * hwidth; ++ i )
                if( lnT_ref[ i + 1 ] > lnT )
                    break;
        return i;
    };

    ////////// Device-side Function //////////    
    template< class g_T > __device__ __forceinline__
    g_T dlnL_dlnT( int & i, const g_T & lnT ) const
    {
        i = find_idx( lnT ); 
        // if( lnT < lnT_ref[ 0 ] ||
        //     lnT > lnT_ref[ 2 * hwidth ] )
        //     return 0; // Zero derivative, not zero values!
        return ( lnL_ref[ i + 1 ] - lnL_ref[ i ] )
             / ( lnT_ref[ i + 1 ] - lnT_ref[ i ] );
    };

    template< class g_T > __device__ __forceinline__
    g_T dlnL_dlnT( const g_T & lnT ) const
    {
        int i;
        return dlnL_dlnT( i, lnT );
    };

    template< class g_T > __device__ __forceinline__
    g_T lnL( const g_T & lnT )
    {
        int i ;
        const auto dln = dlnL_dlnT( i, lnT );
        return lnL_ref[ i ]  + dln * ( lnT - lnT_ref[ i ] );
    };

    template< class g_T > __device__ __forceinline__
    g_T L( const g_T & lnT )
    {
        const auto res = expf( lnL( lnT ) );
        return ( res < 1e2f * f_min ? 0 : res );
    };

    template  < class dat_T, class  bdt_T , class eos_T,
                class rls_T, class cool_T > __device__
    void setup( dat_T        &   d , const bdt_T & bdt,
                const  eos_T & eos , const rls_T & rls,
                const cool_T &   c )
    {
        const auto   T_old = d.T;
        const auto lnT_old = logf(  T_old );
        const auto   n_pts = 2 * hwidth + 1;
        for( int i = 0; i < n_pts; ++ i )
        {
            lnT_ref[ i ] = lnT_old + 1.0f * ( i - hwidth );
            d.T  = c.exp_( lnT_ref [ i ] );
            const auto L_ref = __max1
            ( c.rate_gen( d, bdt, eos, rls ), 10 * f_min );
            lnL_ref[ i ] = logf( L_ref );
        }
        d.T  = T_old;
        return;
    };
};

};                    // namespace extension::chemistry

