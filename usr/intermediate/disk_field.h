#pragma once

#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"
#include "disk_profile.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  The poloidal magnetic field:  the Zanni et al. 2007 bending
//  configuration,  ported from the WBG19 Athena++ pgen
//  ( set_Bp.cpp )  but applied at t = 0 ( t_mag_on is NOT
//  carried over ).
//
//  A_phi( R, th ) = A0 ( R / r0 )^{ 1 - ( alpha + qT ) / 2 }
//                   * ( 1 + 1 / ( tan th * mu )^2 )^{ -5/8 },
//  A0 = 2 bz0 r0 / ( 4 - alpha - qT ),
//  bz0 = sqrt( 2 P_mid( r0 ) / beta0 )  ( code units,
//  Heaviside - Lorentz:  p_mag = B^2 / 2 ),
//  alpha = -rho_pow ( 2.25 ),  qT = -t_pow ( 0.5 ) --  both
//  inherited from the disk profile.
//
//  The CT - consistent face construction ( Evans - Hawley ):
//  A_phi lives on the phi - edges;   B_r = + d( A_phi L_phi )
//  / dA_r over theta,  B_th = - d( A_phi L_phi ) / dA_th over
//  r,  with L_phi = r sin( th ) dphi the phi - edge length --
//  discrete div B = 0 to machine precision.   The WBG19 polar
//  clamp  A_phi R >= Phimin  ( Phimin at the ( r_in, th_min )
//  corner )  keeps the near - pole field radial ~ 1 / r^2.

struct disk_field_t
{
    ////////// Types //////////
    using float2_t = type::float2_t;

    ////////// Data //////////
    int      on;
    float2_t beta0;
    float2_t mu;
    float2_t polarity;
    //  The polar - funnel field exclusion ( [field]
    //  psi_min_fac ):  the flux - function floor  Phimin =
    //  psi_min_fac * A0  --  where psi = A_phi R < Phimin
    //  the field is EXCLUDED ( A_phi = Phimin / R ->  B = 0,
    //  EXACTLY divergence - free,  unlike a per - cell B
    //  scaling ).   Motivation:  the polar rim 's Alfven
    //  speed ( the Zanni bending field at small theta )
    //  would otherwise limit the global dt.   0 = off (
    //  Phimin = the domain - corner value,  never fires
    //  inside );  ~ 0.15 de - fields the theta <~ 0.25
    //  funnel.
    float2_t psi_min_fac;
    int      supp_nr, supp_nt;

    ////////// Functions //////////
    __host__ disk_field_t(  )
    {
        on = 0;   beta0 = 1e5;   mu = 1;   polarity = 1;
        return;
    };

    __host__ void read( const input & args )
    {
        on       = args.get< int >
                   ( "field", "on",       0 );
        beta0    = args.get< float2_t >
                   ( "field", "beta0",    1e5 );
        mu       = args.get< float2_t >
                   ( "field", "mu",       1.0 );
        polarity = args.get< float2_t >
                   ( "field", "polarity", 1.0 );
        psi_min_fac = args.get< float2_t >
                      ( "field", "psi_min_fac", 0.0 );
        return;
    };

    //  The vector potential at a phi - edge ( r_f, th_f ),
    //  with the WBG19 polar clamp ( Phimin from the
    //  ( r_in, th_min ) corner,  passed in ).
    __host__ static float2_t aphi_raw
    ( const float2_t R, const float2_t th,
      const float2_t A0, const float2_t r0,
      const float2_t zeta, const float2_t mu )
    {
        //  On the axis ( R = 0 or th = 0 / pi ) the two factors
        //  are  inf  and  0  and their product is NaN.  The
        //  limit is  A_phi -> 0  ( the flux function  psi =
        //  A_phi R ~ th^{7/8} -> 0 ),  which the  Phimin
        //  clamp then lifts -- so return the limit directly.
        if( ! ( R > 0 ) )
            return 0;
        const float2_t tt = tan( th ) * mu;
        if( ! ( tt != 0 ) )
            return 0;
        return A0 * pow( R / r0, zeta )
             * pow( 1 + 1 / ( tt * tt ), -0.625 );
    };

    //  Apply to one block ( host side,  from init_cond ).
    //  bf.at( i )[ a ] is the LOW a - face of cell i.
    template< class bdt_T > __host__
    void apply( bdt_T & b_h, const disk_profile_t & prof )
                const
    {
        if( ! on )
            return;
        const auto & geo( b_h.geo );
        const int n0 = geo.n_ceff[ 0 ], n1 = geo.n_ceff[ 1 ],
                  n2 = geo.n_ceff[ 2 ];
        const float2_t alpha = - prof.rho_pow;
        const float2_t qT    = - prof.t_pow;
        const float2_t zeta  = 1 - 0.5 * ( alpha + qT );
        //  bz0:  p_mag = B^2 / 2 = P_mid( r0 ) / beta0 --
        //  query the profile itself for P_mid( r0 ) so the
        //  normalization is exact regardless of the n0 /
        //  mu conventions inside disk_profile_t.
        float2_t x0[ 3 ] = { prof.r0, 1.5707963267948966,
                             0.0 };
        float2_t rho_m, pre_m, vp_m;
        prof.hydro( x0, rho_m, pre_m, vp_m );
        const float2_t bz0 = sqrt( 2 * pre_m / beta0 );
        const float2_t A0  = 2 * bz0 * prof.r0
                             / ( 4 - alpha - qT );
        type::idx_t zi;   zi[ 0 ] = 0;   zi[ 1 ] = 0;
        zi[ 2 ] = 0;
        const float2_t dphi = geo.dx_f( 2, zi );

        //  Phimin:  the clamp --  the ( r_in, th_min )
        //  corner value by default,  lifted to
        //  psi_min_fac * A0 when the polar - funnel
        //  exclusion is on.
        const float2_t r_in  = geo.x_fc( 0, 0 );
        const float2_t th_in = geo.x_fc( 1, 0 );
        float2_t Phimin = aphi_raw( r_in * sin( th_in ),
            th_in, A0, prof.r0, zeta, mu )
            * r_in * sin( th_in );
        if( psi_min_fac * A0 > Phimin )
            Phimin = psi_min_fac * A0;

        //  A_phi L_phi on the phi - edges:   A3( j, i ) =
        //  A_phi( r_f( i ), th_f( j ) ) * r_f sin( th_f )
        //  dphi.
        const int nA0 = n0 + 1, nA1 = n1 + 1;
        std::vector< float2_t > A3( nA0 * nA1 );
        for( int j = 0; j < nA1; ++ j )
            for( int i = 0; i < nA0; ++ i )
            {
                const float2_t rf  = geo.x_fc( 0, i );
                const float2_t thf = geo.x_fc( 1, j );
                const float2_t Rf  = rf * sin( thf );
                float2_t aR = aphi_raw( Rf, thf, A0, prof.r0,
                                        zeta, mu ) * Rf;
                if( aR < Phimin )
                    aR = Phimin;
                //  A3 holds A_phi * L_phi = aR / Rf * Rf *
                //  dphi...  i.e.  A_phi * r sin( th ) dphi:
                A3[ j * nA0 + i ] = aR * dphi;
            }
        //  ^ NOTE:  A3 = A_phi * ( R dphi );  the R in the
        //  clamp is already folded:   A_phi * R = max( raw,
        //  Phimin ).

        //  The degenerate pole edges ( th_f = 0 / pi ):  the face
        //  area  ~ sin th  vanishes,  so  B = dA / area  is
        //  0 / 0  there.  Take the regular limit instead:  B_r
        //  carried over from the adjacent edge,  B_th = 0 ( the
        //  Phimin - clamped funnel field is radial ).
        auto deg_th = [ & ] ( const int j )
        {
            //  The same criterion the spherical meshgen uses to
            //  snap the polar - axis faces to zero area.
            const auto sn = sin( geo.x_fc( 1, j ) );
            return sn > -1.e-6 && sn < 1.e-6;
        };
        auto br_face = [ & ] ( const int i, const int j,
                               const int k )
        {
            const int je = deg_th( j ) ? ( j > 0 ? j - 1 : 1 )
                                       : j;
            type::idx_t fi;   fi[ 0 ] = i;   fi[ 1 ] = je;
            fi[ 2 ] = k;
            const float2_t Ar = geo.surface( fi, 0 );
            return ( type::float_t ) ( polarity *
                ( A3[ ( je + 1 ) * nA0 + i ]
                  - A3[ je * nA0 + i ] ) / Ar );
        };
        for( int k = 0; k < n2; ++ k )
            for( int j = 0; j < n1; ++ j )
                for( int i = 0; i < n0; ++ i )
        {
            type::idx_t ci;   ci[ 0 ] = i;   ci[ 1 ] = j;
            ci[ 2 ] = k;
            type::idx_t fi;   // r - face at i ( low )
            fi[ 0 ] = i;   fi[ 1 ] = j;   fi[ 2 ] = k;
            auto * bf = b_h.fld.bf.at( ci );
            //  B_r on the low r - face:
            bf[ 0 ] = br_face( i, j, k );
            //  B_th on the low th - face:
            if( deg_th( j ) )
                bf[ 1 ] = 0;
            else
            {
                const float2_t At = geo.surface( fi, 1 );
                bf[ 1 ] = ( type::float_t ) ( - polarity *
                    ( A3[ j * nA0 + i + 1 ]
                      - A3[ j * nA0 + i ] ) / At );
            }
            bf[ 2 ] = 0;
            //  The high faces of the last active cells:
            if( i == n0 - 1 )
            {
                type::idx_t fh;   fh[ 0 ] = i + 1;
                fh[ 1 ] = j;   fh[ 2 ] = k;
                b_h.fld.bf.at( fh )[ 0 ] = br_face( i + 1, j, k );
            }
            if( j == n1 - 1 )
            {
                type::idx_t fh;   fh[ 0 ] = i;
                fh[ 1 ] = j + 1;   fh[ 2 ] = k;
                auto * bh = b_h.fld.bf.at( fh );
                if( deg_th( j + 1 ) )
                    bh[ 1 ] = 0;
                else
                {
                    const float2_t Ath = geo.surface( fh, 1 );
                    bh[ 1 ] = ( type::float_t ) ( - polarity *
                        ( A3[ ( j + 1 ) * nA0 + i + 1 ]
                          - A3[ ( j + 1 ) * nA0 + i ] ) / Ath );
                }
            }
        }

        //  Cell - centered B ( face average ) + the magnetic
        //  energy into the conserved state.
        if( on > 1 )
        {
            //  Debug:  numeric curl of A_phi vs the stored
            //  face fields at probe cells.
            std::cout << "FLDDBG norm:  pre_m = " << pre_m
                      << "  bz0 = " << bz0 << "  A0 = " << A0
                      << "  zeta = " << zeta
                      << "  Phimin = " << Phimin << std::endl;
            auto numB = [ & ] ( const int i, const int j )
            {
                type::idx_t ci;   ci[ 0 ] = i;   ci[ 1 ] = j;
                ci[ 2 ] = 0;
                const float2_t r  = geo.x_cc( 0, i );
                const float2_t th = geo.x_cc( 1, j );
                const float2_t dr = geo.dx_f( 0, ci );
                const float2_t dt = geo.dx_f( 1, ci );
                auto aR = [ & ] ( float2_t rr, float2_t tt )
                {
                    float2_t R = rr * sin( tt );
                    float2_t a = aphi_raw( R, tt, A0, prof.r0,
                                           zeta, mu ) * R;
                    return a < Phimin ? Phimin : a;
                };
                //  bf[ 0 ] lives on the LOW r - face
                //  ( r_f( i ),  th_c( j ) );  bf[ 1 ] on the
                //  low th - face ( r_c( i ),  th_f( j ) ).
                const auto * bl = b_h.fld.bf.at( ci );
                const float2_t rf = geo.x_fc( 0, i );
                const float2_t tf = geo.x_fc( 1, j );
                const float2_t Br
                    = ( aR( rf, th + dt / 2 )
                        - aR( rf, th - dt / 2 ) )
                      / ( rf * sin( th ) * dt );
                const float2_t Bt
                    = - ( aR( r + dr / 2, tf )
                          - aR( r - dr / 2, tf ) ) / ( r * dr );
                std::cout << "FLDDBG ( i " << i << ",  j "
                          << j << " ):  bf = " << bl[ 0 ]
                          << " " << bl[ 1 ]
                          << "   analytic Br Bt = " << Br
                          << " " << Bt << std::endl;
            };
            numB( 8, n1 / 2 );
            numB( 8, 8 );
            numB( 21, n1 / 2 );
        }
        for( int k = 0; k < n2; ++ k )
            for( int j = 0; j < n1; ++ j )
                for( int i = 0; i < n0; ++ i )
        {
            type::idx_t ci;   ci[ 0 ] = i;   ci[ 1 ] = j;
            ci[ 2 ] = k;
            type::idx_t fi = ci;
            const auto * bl = b_h.fld.bf.at( ci );
            ++ fi[ 0 ];
            const auto * br = b_h.fld.bf.at( fi );
            fi = ci;   ++ fi[ 1 ];
            const auto * bt = b_h.fld.bf.at( fi );
            auto * u = b_h.hyd.u.at( ci );
            float2_t bcc[ 3 ];
            bcc[ 0 ] = 0.5 * ( bl[ 0 ] + br[ 0 ] );
            bcc[ 1 ] = 0.5 * ( bl[ 1 ] + bt[ 1 ] );
            bcc[ 2 ] = 0;
            for( int a = 0; a < 3; ++ a )
                u[ idx::bcc + a ] = ( type::float2_t )
                                    bcc[ a ];
            u[ idx::ene ] += 0.5
                * ( bcc[ 0 ] * bcc[ 0 ] + bcc[ 1 ] * bcc[ 1 ]
                    + bcc[ 2 ] * bcc[ 2 ] );
        }
        return;
    };
};

};                              // namespace interm
