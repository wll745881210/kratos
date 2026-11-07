#pragma once

#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  The dust temperature provider ( usr_ext / ppd /
//  design.md decision 5 ).   Currently the WBG19 - style
//  prescription  T_dust( R ) = T_d0 ( R / r0 )^q  ( R = the
//  cylindrical distance to the star ).   THE UPGRADE POINT
//  for the future self - consistent dust radiative transfer
//  ( DDMC / FLD / M1 - closure ):  the provider 's eval takes
//  the block data and the cell index --  today it uses only
//  the position;  then it will read the radiation module 's
//  state ( the local flux / the solved temperature ).

struct disk_dust_t
{
    ////////// Types //////////
    using float2_t = type::float2_t;

    ////////// Data //////////
    float2_t  T_d0;   // T_dust at r0  [K]
    float2_t      q;  // power index
    float2_t     r0;  // reference radius ( code units )
    float2_t  T_cap;  // ceiling ( dust sublimates ~ 1500 K;
                      // R = r sin th -> 0 at the poles would
                      // otherwise diverge for q < 0 )
    //  The FLD mode ( disk_fld.h,  T11 ):  T_dust from the
    //  FLD energy density E ( stored E / lum_sc )  instead
    //  of the analytic profile.
    int       fld_on;
    float2_t  lum_sc;  //  the E - field rescale
    float2_t  a_c;     //  a_rad in code ( ene0 ) units
    float2_t  T_min;   //  the E - floor's temperature

    ////////// Functions //////////
    __host__ disk_dust_t(  )
    {
        T_d0 = 280;   q = -0.5;   r0 = 1;   T_cap = 1500;
        fld_on = 0;   lum_sc = 1e4;   a_c = 0;   T_min = 1;
        return;
    };

    __host__ void read( const input & args )
    {
        //  Defaults follow the disk 's midplane T( R ) so the
        //  accommodation holds the disk at its IC temperature
        //  ( the M1 self - consistency test ).
        T_d0  = args.get< float2_t >
                   ( "dust", "T_d0",   280.0 );
        q     = args.get< float2_t >
                   ( "dust", "T_pow",  -0.5 );
        r0    = args.get< float2_t >
                   ( "profile", "r0",   1.0 );
        T_cap = args.get< float2_t >
                   ( "dust", "T_cap", 1500.0 );
        fld_on = args.get< int >( "dust", "fld_on", 0 );
        lum_sc = args.get< float2_t >
                 ( "fld", "lum_sc", 1e4 );
        if( fld_on )
        {
            phys::unit_t< float2_t > unit;
            unit.init( args );
            //  a_rad = 4 sigma_sb / c  [ erg / cc / K^4 ]
            //  ->  code ( / ene0 ):
            const float2_t a_cgs = 4 * phys::cgs::sigma_sb
                                 / phys::cgs::c;
            a_c = a_cgs / unit.ene0;
        }
        return;
    };

    template< class bdt_T, class idx_T >
    __device__ __forceinline__ type::float_t eval
    ( const bdt_T & b_d, const idx_T & i ) const
    {
        if( fld_on )
        {
            const auto e = b_d.e_fld.at( i )[ 0 ];
            if( e > 0 )
            {
                //  T = ( E lum_sc / a )^{ 1/4 }:
                const auto T4 = lum_sc / a_c * e;
                const auto Tt = __sqrt1( __sqrt1( T4 ) );
                if( Tt > T_min )
                    return ( type::float_t ) Tt;
            }
            //  fall through to the analytic profile where
            //  E is unset / nonpositive
        }
        //  R = r sin( th ) in code units;  T in KELVIN ( the
        //  solver works in CGS ).
        const auto r  = b_d.geo.x_cc( 0, i[ 0 ] );
        const auto th = b_d.geo.x_cc( 1, i[ 1 ] );
        const auto R  = r * __sin1( th );
        const auto T   = T_d0 * pow( R / r0, q );
        return ( type::float_t ) ( T < T_cap ? T : T_cap );
    };
};

};                              // namespace interm
