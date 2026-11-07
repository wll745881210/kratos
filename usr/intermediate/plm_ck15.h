#pragma once

#include "../../src/modules/dynamics/mhd/reconstruct/plm.h"
#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  CK15 ( Chandrashekar & Klingenberg 2015,  SIAM SISC 37( 3 )
//  B382 ) well - balanced PLM:  the pressure slot is
//  reconstructed through the equilibrium variable
//        w  =  p * exp( - psi ),
//  constant on any state satisfying the discrete hydrostatic
//  law  p_{j+1} = p_j exp( psi_{j+1} - psi_j ).   psi is built
//  LOCALLY,  per interface,  from the trapezoidal integral of
//  lam = d ln p / dx with the effective body force of the
//  CURRENT state ( no profile forcing ):
//      theta:  lam = rho v_phi^2 cot( th ) / p
//      radial:  lam = rho ( ( v_th^2 + v_phi^2 ) / r
//                         - gm / r^2 ) / p
//  The four stencil values are mapped to w with the pairwise
//  ratios exp( -psi_k ) ( gauge psi = 0 at the face's left
//  cell ),  PLM - limited,  and the two face states are
//  recovered with exp( +psi_f ).   At equilibrium w is flat,
//  so p^L = p^R at every face and a contact - preserving
//  Riemann solver returns the pure pressure flux,  which the
//  matched geometric source in stp_t::src_geo cancels to
//  machine precision.   Density stays on plain PLM:  the
//  contact property needs only equal face pressures ( and
//  vanishing normal velocity ).
//
//  The SAME [prob] ck15 bitmask gates this reconstruction and
//  the matched source in interm::stp_t -- the pair is only
//  consistent together;  usr.cpp therefore enrolls this
//  reconstructor whenever ck15 != 0.   omega_0 != 0 is not
//  supported ( the frame shift is not part of lam ).
//
//  Bitmask:  1 = theta,  2 = radial.

template< class base_T >
struct plm_ck15_t : base_T
{
    ////////// Types and data //////////
    using super_t = base_T;
    using super_t::q_ll, super_t::q_l;
    using super_t::q_rr, super_t::q_r;
    using float2_t = type::float2_t;

    int      ck15;
    float2_t gm;

    ////////// Functions //////////
    __host__ virtual void init( const input & args ) override
    {
        phys::unit_t< float2_t > unit;
        unit.init( args );
        const auto mstar = args.get< float2_t >
            ( "prob", "mstar_cgs", 2e33 );
        gm   = phys::cgs::G * mstar / unit.derive( 0, 3, -2 );
        ck15 = args.get< int >( "prob", "ck15", 0 );
        return;
    };

    template< class dat_T, class bdt_T > __device__
    bool apply( dat_T & d, const int & i, const int & n,
                const bdt_T & bdt, const bool is_uni ) const
    {
        //  Only the pressure slot,  only r / th strips,  only
        //  where the bitmask enables this axis.  ( omega_0 is
        //  an integrator - side quantity;  the integrator's
        //  matched source already requires omega_0 = 0. )
        if( n != idx::ene || d.axis > 1 )
            return false;
        const int bit = d.axis == 1 ? 1 : 2;
        if( ! ( ck15 & bit ) )
            return false;

        const auto & geo = bdt.geo;
        type::idx_t j3;
        d.dispatch( j3, bdt );
        const int s0 = i + d.blk_shift;
        float2_t lam[ 4 ], xc[ 4 ];
        for( int k = 0; k < 4; ++ k )
        {
            j3[ d.axis ] = s0 + k - 2;
            const auto * w = bdt.hyd.w.at( j3 );
            const auto rho = w[ idx::rho ];
            const auto pre = w[ idx::ene ];
            const auto v1  = w[ idx::mom + 1 ];
            const auto vf  = w[ idx::mom + 2 ];
            if( d.axis == 1 )
            {
                const auto th = geo.x_cc( 1, j3[ 1 ] );
                lam[ k ] = vf * vf * __cos1( th )
                         / __sin1( th ) * rho / pre;
            }
            else
            {
                const auto r = geo.x_cc( 0, j3[ 0 ] );
                lam[ k ] = ( ( v1 * v1 + vf * vf ) / r
                           - gm / ( r * r ) ) * rho / pre;
            }
            xc[ k ] = geo.x_cc( d.axis, j3[ d.axis ] );
            //  Sanity gate,  part 1:  positive rho / pre on
            //  the whole stencil.
            if( ! ( rho > 0 ) || ! ( pre > 0 ) )
                return false;
        }
        //  Trapezoidal psi differences,  gauge psi_l = 0 (
        //  cell k = 1 ).   xf0 = face between k = 0, 1;
        //  xf1 = the reconstructed face ( k = 1, 2 );
        //  xf2 = face between k = 2, 3.
        const auto xf0 = geo.x_fc( d.axis, s0 - 1 );
        const auto xf1 = geo.x_fc( d.axis, s0     );
        const auto xf2 = geo.x_fc( d.axis, s0 + 1 );
        const auto ps_f  = lam[ 1 ] * ( xf1 - xc[ 1 ] );
        const auto ps_r  = ps_f + lam[ 2 ]
                         * ( xc[ 2 ] - xf1 );
        const auto ps_rr = ps_r + lam[ 2 ] * ( xf2 - xc[ 2 ] )
                                + lam[ 3 ] * ( xc[ 3 ] - xf2 );
        const auto ps_ll = - ( lam[ 1 ] * ( xc[ 1 ] - xf0 )
                             + lam[ 0 ] * ( xf0 - xc[ 0 ] ) );
        //  Sanity gate,  part 2:  the exponential
        //  extrapolation is only meaningful where the state
        //  is close to balance.  Far from balance ( e.g. the
        //  floor - clipped,  super - Keplerian polar rim,
        //  lam * dx ~ 18 there )  it AMPLIFIES instead of
        //  flattening -- plain PLM is the robust path there.
        const float2_t lim = 2;
        if( __abs1( ps_f  ) > lim || __abs1( ps_r  ) > lim
            || __abs1( ps_rr ) > lim || __abs1( ps_ll ) > lim )
            return false;
        q_l  = q_l;
        q_ll = q_ll * __exp1( - ps_ll );
        q_r  = q_r  * __exp1( - ps_r  );
        q_rr = q_rr * __exp1( - ps_rr );
        if( is_uni )
            super_t::uniform( d, i, n, bdt );
        else
            super_t::general( d, i, n, bdt );
        const auto ef = __exp1( ps_f );
        auto * ql = d.ql.at( i );
        auto * qr = d.qr.at( i );
        ql[ n ] *= ef;
        qr[ n ] *= ef;
        const float2_t fl = 1e-30;
        if( ql[ n ] < fl )  ql[ n ] = fl;
        if( qr[ n ] < fl )  qr[ n ] = fl;
        return true;
    };

    template< class dat_T, class bdt_T > __device__
    void uniform( dat_T & d, const int & i, const int & n,
                  const bdt_T & bdt ) const
    {
        if( apply( d, i, n, bdt, true ) )
            return;
        return super_t::uniform( d, i, n, bdt );
    };

    template< class dat_T, class bdt_T > __device__
    void general( dat_T & d, const int & i, const int & n,
                  const bdt_T & bdt ) const
    {
        if( apply( d, i, n, bdt, false ) )
            return;
        return super_t::general( d, i, n, bdt );
    };
};

};                              // namespace interm
