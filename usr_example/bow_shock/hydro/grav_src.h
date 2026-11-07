#pragma once

#include <cmath>
#include "../../../src/utilities/functions.h"

namespace friction
{
////////////////////////////////////////////////////////////
// Gravitational source type

struct star_t
{
    ////////// Types //////////
    using float_t = type::float_t;
    using coord_t = type::coord_t;

    ////////// Data  //////////
    float_t         g_m;          // G * M
    float_t    r_sink_2;          // pow( r_sink, 2 )
    float_t    r_soft_2;          // pow( r_soft, 2 )
    float_t    x_c[ 3 ];          // Location of the center

    float_t        gam1;          // (Adiabatic gamma) - 1
    float_t    mdot_4pi;
    float_t   v_outflow;
    float_t  pre_o_rho0;
    bool  is_outflowing;

    float_t   rho_floor;
    float_t   tmp_floor;

    __host__ virtual void init( const input & args )
    {
        g_m = args.get< float_t > ( "prob", "g_m", 1 );
        const auto pre_o_rho0 = args.get< float_t >
            ( "prob", "pre_o_rho_outflow", 1 );
        mdot_4pi = args.get< float_t >
            ( "prob",  "mdot", 1 ) / ( 4 * 3.141592654 );
        v_outflow = args.get< float_t >
            ( "prob", "v_outflow", 1 );
        gam1 = args.get< float_t >
            ( "dynamics","gamma", 1.4 ) - 1;
        is_outflowing = ( mdot_4pi > 0 );
        for( int a = 0; a < 3; ++ a )
            x_c[ a ] = 0;
        args( "prob", "x_c", x_c );

        float_t r_sink( 0 ), r_soft( 0 );
        if( args.found( "prob", "r_sink" ) )
        {
            r_sink = args.get< float_t >
                ( "prob", "r_sink" );
            r_soft = args.get< float_t >
                ( "prob", "r_sink", r_soft );
        }
        else
        {
            int     n_cell[ 3 ];
            coord_t x_max, x_min;
            args( "mesh", "x_max",          x_max );
            args( "mesh", "x_min",          x_min );
            args( "mesh", "n_cell_global", n_cell );

            int lvl( 0 );
            for( const auto & p :  args.get_prefixes(  ) )
                if( p.find( "refine_region" ) !=
                    std::string::npos )
                    lvl = std::max( lvl, args.get< int >
                                    ( p, "level", 0 ) );
            float_t dx ( 1e32 );
            for( int a = 0; a < 3; ++ a )
                dx = __min1( dx, 1.f / n_cell[ a ] *
                             ( x_max[ a ] - x_min[ a ] ) );
            dx /= pow( 2, lvl );

            r_sink  = args.get< float_t >
                ( "prob", "r_sink_cell",      2 );
            r_soft  = args.get< float_t >
                ( "prob", "r_soft_cell", r_sink );
            r_sink *= dx;
            r_soft *= dx;
        }
        r_sink_2 = pow( r_sink, 2 ) * utils::sgn( r_sink );
        r_soft_2 = pow( r_soft, 2 ) * utils::sgn( r_soft );
        return;
    };

    template< class x_T >
    __host__ __device__ bool in_sink
    ( float_t & r_2, float_t dx[  ], const x_T & x ) const
    {
        r_2 = 0.f;
        for( int a = 0; a < x.size(  ); ++ a )
        {
            dx[ a ] = x [ a ] - x_c[ a ];
            r_2    += dx[ a ] * dx [ a ];
        }
        return r_2 < r_sink_2;
    };
    template< class x_T >
    __host__ __device__ bool in_sink( const x_T & x ) const
    {
        float_t dx[ 3 ], r_2  ( 0.f );
        return in_sink ( r_2, dx, x );
    };
    template< class x_T, class f_T >
    __device__ bool get_acc
    ( x_T & acc, f_T & r_2, const x_T & x ) const
    {
        if( g_m == 0 )
            return false;
        f_T dx[ 3 ];
        if( in_sink( r_2, dx, x ) )
            return  true;

        if( r_2 < r_soft_2 )
        {
            auto r_s   = __sqrt1( r_2 + r_soft_2 );
            auto r_s_2 = ( r_2 + r_soft_2 )
                 * ( r_2 + r_soft_2 );
            const float_t corr_4th
              = ( r_2 + 2.5f * r_soft_2 ) / ( r_s_2 * r_s );
            for( int a = 0; a < acc.size(  ); ++ a )
                acc[ a ] = -g_m * dx[ a ] * corr_4th;
        }
        else
        {
            auto r_inv = 1  / __sqrt1( r_2 );
            for( int a = 0; a < acc.size(  ); ++ a )
                acc[ a ] = -g_m * dx[ a ] / r_2 * r_inv;
        }
        return false;
    };

    template< class f2_T, class x_T > __device__
    void set_flow( f2_T * u, const x_T & x ) const
    {
        if( ! is_outflowing )   // Is a sink
        {
            const float_t rho0( u[ idx::rho ] );
            const float_t ene0( u[ idx::ene ] );
            float_t e_g( u[ idx::ene ] );
            for( int a = 0; a < 3; ++ a )
            {
                const float_t p0  = u [ idx::mom + a ];
                e_g  -= p0 * p0 / ( 2 * rho0 );
                u[ idx::mom + a ] = 0 ;
            }
            const auto rho = __max1( rho0 / 2, rho_floor );
            u[ idx::ene ]  = __max1
                              ( e_g / 2, rho * tmp_floor );
            u[ idx::rho ]  = rho;
        }
        else                    // Is a source
        {
            coord_t dx;
            float_t r2( 0 );
            for( int a = 0; a < 3; ++ a )
            {
                dx[ a ] = x [ a ] - x_c[ a ];
                r2     += dx[ a ] *  dx[ a ];
            }
            const auto mom_r = mdot_4pi     / r2;
            const auto rho   = mom_r / v_outflow;
            const auto e_t   = pre_o_rho0 * rho / gam1
                + mom_r * mom_r / ( 2 * rho );

            const auto r = __sqrt1 ( r2 );
            for( int a = 0; a < 3; ++ a )
                u[ idx::mom + a ] = mom_r * dx[ a ] / r;
            u[ idx::rho ] = rho;
            u[ idx::ene ] = e_t;
        }
        return;
    }
};

};                              // namespace friction
