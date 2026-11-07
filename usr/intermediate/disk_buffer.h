#pragma once

#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"
#include "../../src/modules/dynamics/indices.h"
#include "../../src/types.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  PPD buffer zone ( design.md sec. 2.1;  WBG19 pre_proc,
//  microphys disk_wind / disk_source.cpp ):
//
//  * Buffer r < r_damp + l_damp:  poloidal momentum damped
//    mom_{ r, th } *= max( 1 - damp_fac dt omega_K , 0 ),
//    the removed kinetic energy leaves ene.
//  * The density FLOOR lives in the integrator ( stp_t,
//    composition - preserving;  it used to be here and
//    corrupted the composition at floor cells ).

struct disk_buffer_t
{
    ////////// Types //////////
    using float_t  = type::  float_t;
    using float2_t = type::float2_t;

    ////////// Data //////////
    bool      on;
    float2_t  r_damp;      // 1 au
    float2_t  l_damp;      // 0.5 au
    float2_t  gm;          // G * M_star ( code )
    float2_t  damp_fac;    // 10 ( WBG19 dt * 10 )
    float2_t  rho_floor;   // rho_floor_dyn
    float2_t  floor_r0;
    float2_t  floor_idx;

    ////////// Functions //////////
    __host__ disk_buffer_t(  )
    {
        on        = true;
        r_damp    = 1.0;
        l_damp    = 0.5;
        gm        = 0;
        damp_fac  = 10.0;
        rho_floor = 5e7;
        floor_r0  = 1.0;
        floor_idx = -3.0;
        return;
    };

    __host__ void read( const input & args )
    {
        phys::unit_t< float2_t > unit;
        unit.init( args );
        on        = args.get< int >
                        ( "buffer", "on", 1 ) > 0;
        r_damp    = args.get< float2_t >
                        ( "buffer", "r_damp",    1.0 );
        l_damp    = args.get< float2_t >
                        ( "buffer", "l_damp",    0.5 );
        damp_fac  = args.get< float2_t >
                        ( "buffer", "damp_fac", 10.0 );
        rho_floor = args.get< float2_t >
                        ( "buffer", "rho_floor_dyn", 5e7 );
        floor_r0  = args.get< float2_t >
                        ( "buffer", "rho_floor_r0",  1.0 );
        floor_idx = args.get< float2_t >
                        ( "buffer", "rho_floor_idx", -3.0 );
        const auto mstar = args.get< float2_t >
                        ( "prob", "mstar_cgs", 2e33 );
        gm = phys::cgs::G * mstar / unit.derive( 0, 3, -2 );
        return;
    };

    // Buffer membership ramp ( WBG19 f_damp ):  0 at r_damp,
    // 1 at r_damp + l_damp.
    template< class f_T >
    __host__ __device__ __forceinline__ f_T ramp
    ( const f_T & r ) const
    {
        const f_T f = ( r - r_damp ) / l_damp;
        return f < 0 ? f_T( 0 ) : ( f > 1 ? f_T( 1 ) : f );
    };

    template< class bdt_T, class idx_T >
    __device__ __forceinline__ void apply
    ( const bdt_T & b_d, const idx_T & i ) const
    {
        //  The density floor moved to the integrator ( stp_t,
        //  composition - preserving;  see disk_intg.h ).
        //  This kernel now does only the damping.
        if( ! on )
            return;
        using f_t = typename decltype( b_d.hyd )::f_t;
        const auto r = f_t( b_d.geo.x_cc( 0, i[ 0 ] ) );
        auto * u = b_d.hyd.u.at( i, 0 );
        // WBG19 pre_proc: damp v_r everywhere while B = 0
        // ( the relaxation device ), otherwise only inside
        // the buffer r < r_damp + l_damp.
        f_t b2( 0 );
        for( int a = 0; a < 3; ++ a )
            b2 += u[ idx::bcc + a ] * u[ idx::bcc + a ];
        if( b2 > f_t( 1e-32 ) && r >= r_damp + l_damp )
            return;
        const f_t w_orb = sqrt( gm / ( r * r * r ) );
        const f_t f0 = 1 - damp_fac * b_d.dt * w_orb;
        const f_t f  = f0 < 0 ? f_t( 0 ) : f0;
        const f_t rho = u[ idx::rho ];
        // B = 0 ( hydro relaxation ):  damp BOTH poloidal
        // components --  the full - sphere midplane allows
        // the vertical - collapse pile - up that the WBG19
        // midplane reflection forbade.   With B on ( or in
        // the buffer ),  damp v_r only ( the WBG19 rule ).
        const int n_pol = ( b2 > f_t( 1e-32 ) ) ? 1 : 2;
        for( int a = 0; a < n_pol; ++ a )
        {
            const f_t p0 = u[ idx::mom + a ];
            u[ idx::mom + a ] *= f;
            const f_t p1 = u[ idx::mom + a ];
            u[ idx::ene ] += ( p1 * p1 - p0 * p0 )
                           / ( 2 * rho );
        }
        return;
    };
};

////////////////////////////////////////////////////////////
//  Thin per-cell kernel ( style guide sec. 13.1 )

template< class bdt_T >
__global__ void disk_buffer_k
( const bdt_T b_d, const disk_buffer_t buf )
{
    const auto & geo = b_d.geo;
    const int    n0  = geo.n_ceff[ 0 ];
    const int    n1  = geo.n_ceff[ 1 ];
    const int    t   = blockIdx.x * blockDim.x
                     + threadIdx.x;
    if( t >= n0 * n1 * geo.n_ceff[ 2 ] )
        return;
    type::idx_t i;
    i[ 0 ] =   t % n0;
    i[ 1 ] = ( t / n0 ) % n1;
    i[ 2 ] =   t / ( n0 * n1 );
    buf.apply( b_d, i );
    return;
};

// Flat cell-launch resource ( the implicit_mg rsc_copy
// pattern ).
template< class bdt_T > __host__
auto disk_buffer_rsc( const bdt_T & b_d )
{
    const int n_cell = b_d.geo.n_ceff.prod(  );
    const int n_th   = std::min( n_cell, 256 );
    const dim3 n_bl  ( ( n_cell + n_th - 1 ) / n_th );
    return std::make_tuple( n_bl, dim3( n_th ), 0 );
};

};                              // namespace interm
