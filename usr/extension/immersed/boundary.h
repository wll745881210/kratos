#pragma once

#include "../../../src/modules/dynamics/hydro/boundary/phys.h"

namespace extension::immersed
{
////////////////////////////////////////////////////////////
// Fixed-state inflow (freestream) boundary
//
// Fills the ghost cells of the chosen side with the constant
// primitive state ( rho0, pre0, vel0 ) read from [prob].

template< class prx_T = dynamics::hydro::proxy_t >
struct inflow_t
    : dynamics::hydro::boundary::phys_base_t
      < prx_T, inflow_t< prx_T > >
{
    ////////// Types //////////
    using float2_t = type::float2_t;
    using coord2_t = type::coord2_t;

    ////////// Data //////////
    float2_t  rho0;
    float2_t  pre0;
    coord2_t  vel0;

    ////////// Initializer //////////
    __host__ inflow_t(  )
        : rho0( 1 ), pre0( 1 ), vel0( 0 ) {  };

    __host__ virtual void init
    ( const input & args, mesh::mod_base_t & mod ) override
    {
        rho0 = args.get< float2_t >( "prob", "rho0", 1 );
        pre0 = args.get< float2_t >( "prob", "pre0", 1 );
        vel0 = args.get< coord2_t >( "prob", "vel0", 0 );
        return;
    };

    ////////// Device-side interface //////////
    template< class d1d_T, class bdt_T > __forceinline__ __device__
    void hydro( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            d( i, idx::rho ) = rho0;
            d( i, idx::ene ) = pre0;
            // Primitive state: idx::mom + a holds a velocity.
            for( int a = 0; a < 3; ++ a )
                d( i, idx::mom + a ) = vel0[ a ];
        }
        return;
    };
};

};                              // namespace extension::immersed
