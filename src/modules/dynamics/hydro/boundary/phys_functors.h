#pragma once

#include "phys.h"

namespace dynamics::hydro::boundary
{
////////////////////////////////////////////////////////////
// Functors for most commonly used hydrodynamic boundaries

template< class prx_T  = dynamics::hydro::proxy_t >
struct free_t : phys_base_t< prx_T, free_t< prx_T > >
{
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
        return;
    };
};

template < class   prx_T = dynamics::hydro::proxy_t >
struct fixed_t : phys_base_t< prx_T, fixed_t< prx_T > >
{
    //  Dirichlet boundary: the ghost state is fixed to a
    //  prescribed PRIMITIVE state ( the boundary functors
    //  operate on hyd.w ), filled by the keeper at init
    //  from [init] rho0 / pre0 / vel0, i.e. the ambient
    //  medium.  Use at boundaries with physical inflow:
    //  outflow_t suppresses inflow by clamping the inward
    //  normal momentum, while free_t extrapolates the
    //  interior state.  Anything more elaborate ( spatially
    //  varying or time-dependent injection ) belongs in a
    //  problem-specific functor in the pgen.
    type::float2_t w_fix[ 5 ];

    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = w_fix[ n ];
        return;
    };
};

template < class   prx_T = dynamics::hydro::proxy_t >
struct outflow_t : phys_base_t< prx_T, outflow_t< prx_T > >
{
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        const bool right = this->side & 1;
        const int  i_mom ( this->side / 2 + idx::mom );
        if( d( i_mom ) * ( right ? -1 : 1 ) > 0 )
            d( i_mom ) = 0;        
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
            // if( i_mom - idx::mom == 0 && right )
            //     std::cerr << "HEY\n";
        }
        return;
    };
};

template < class   prx_T = dynamics::hydro::proxy_t >
struct reflect_t : phys_base_t< prx_T, reflect_t< prx_T > >
{
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        const int i_mom( idx::mom + this->side / 2 );
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            auto *  dl = d.at(  i     );
            auto *  dr = d.at( -i - 1 );
            for( int n = 0; n < d.n_int; ++ n )
                dl[ n ] = dr[ n ] * ( n == i_mom ? -1 : 1 );
        }
        return;
    };
};

};  // namespace dynamics::hydro::boundary
