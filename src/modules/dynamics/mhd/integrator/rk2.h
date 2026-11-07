#pragma once

#include "../../hydro/integrator/rk2.h"
#include "integrator.h"
#include <type_traits>

namespace dynamics::mhd::integrator
{
////////////////////////////////////////////////////////////
// Second-order Runge-Kutta integrator for MHD

template < typename derived_T = crtp::dummy_t >
struct rk2_t : crtp::helper_t< rk2_t, derived_T, base_t >
{
    ////////// Temporal order //////////
    __host__ virtual int time_order(  ) const { return 2; };

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
    
    //  mask:  bit  a  marks the components whose  db  was
    // actually computed at this slot  ( the seam / boundary
    // rows compute only the aligned component;  the others
    // belong to the comm packs --  writing the uncomputed
    //  db  slots dumps uninitialized stack content into the
    // ghost band,  the sl3 pure - MHD nondeterminism ).
    template< class fld_T, class idx_T, class f2_T >
    __forceinline__ __device__ void set_field
    ( const fld_T &  fld , const idx_T &  i ,
      const int   & step , const  f2_T * db ,
      const int   & mask ) const
    {
        auto * b_src = fld.bf.at( i,     step );
        auto * b_tgt = fld.bf.at( i, 1 - step );
        for( int a = 0; a < 3; ++ a )
            if( mask & ( 1 << a ) )
                b_tgt[ a ] = db[ a ] +
                    ( step  == 0 ? b_src[ a ]
                      : ( b_src[ a ] + b_tgt[ a ] ) / 2 );
        return;
    };
};

};                   // namespace dynamics::mhd::integrator
