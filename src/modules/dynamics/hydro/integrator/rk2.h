#pragma once

#include "integrator.h"
#include <type_traits>

namespace dynamics::hydro::integrator
{
////////////////////////////////////////////////////////////
// Second-order Runge-Kutta integrator for hydro only

template < typename derived_T = crtp::dummy_t >
struct rk2_t : crtp::helper_t< rk2_t, derived_T, base_t >
{
    ////////// Temporal order //////////
    virtual __host__ int time_order(  ) const { return 2; };
    __device__ __forceinline__ 
    constexpr int    time_order_d  (  ) const { return 2; };

    ////////// Actuator //////////
    template< class f2_T, class cons_T, class idx_T >
    __device__ f2_T * load
    ( f2_T *  v , const  cons_T & u, const idx_T & i,
      const int & step ) const
    {
        const auto & n_var = u.n_int;
        auto  *      u_src = u.at( i,     step );
        auto  *      u_tgt = u.at( i, 1 - step );
        if( step == 0 )
            for( int n = 0; n < n_var; ++ n )
                v[ n ] =   u_src[ n ];
        else
            for( int n = 0; n < n_var; ++ n )
                v[ n ] = ( u_src[ n ] + u_tgt[ n ] ) * 0.5f;
        return  u_tgt;
    };

    template< class f_T = type::float_t > __forceinline__
    __host__ f_T step_size( const int & step ) const
    {
        return ( step == 0 ? ( f_T ) 1 : ( f_T ) 0.5f );
    };
};

}; // namespace dynamics::hydro::integrator
