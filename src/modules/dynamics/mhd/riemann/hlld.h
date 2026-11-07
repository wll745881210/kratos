#pragma once

#include "../../prototypes/riemann.h"

namespace dynamics::mhd::riemann
{
////////////////////////////////////////////////////////////
// Derived riemann solver type for HLLD

template< class derived_T = crtp::dummy_t >
struct hlld_t : ::dynamics::riemann::interface_t 
              < crtp::helper< hlld_t,  derived_T > >
{
    __crtp_def_self__( hlld_t, derived_T );

    template< class dat_T, class bdt_T, class eos_T >
    __forceinline__ __device__ void operator(  )
    ( dat_T & d, const bdt_T & b_d,
      const eos_T & f_eos )  const    
    {
        __syncthreads(  );
        for( int i = threadIdx.x ; i <  d. n_feff;
             i    +=  blockDim.x )
            get_self( ).calc( d.flx.at( i ), d.ql.at( i ),
                              d.qr.at( i ), d, b_d, f_eos );
        return;
    };

    template< class   f_T, class dat_T ,
              class bdt_T, class eos_T >
    __forceinline__ __device__ void calc
    ( f_T * flx,  f_T * ql, f_T * qr, const dat_T & d,
      const bdt_T & b_d, const eos_T & f_eos )  const
    {
        const auto & axis( d.axis );
        utils  ::idx_swap( ql  + idx::mom,      axis );
        utils  ::idx_swap( qr  + idx::mom,      axis );
        utils  ::idx_swap( ql  + idx::bcc,      axis );
        utils  ::idx_swap( qr  + idx::bcc,      axis );
        get_self(  ).eval( flx , ql,   qr,  f_eos, d );
        utils  ::idx_swap( flx + idx::mom,  3 - axis );
        const f_T ef[ 2 ] = { flx[ idx::bcc + 2 ] ,
                            - flx[ idx::bcc + 1 ] };
        flx[ idx::bcc + utils::mod3( axis + 1 ) ] = ef[ 0 ];
        flx[ idx::bcc + utils::mod3( axis + 2 ) ] = ef[ 1 ];
        flx[ idx::bcc +              axis       ] =       0;
    };

    template< class f_T > __device__ void f_flux
    ( f_T * flux, const f_T * u , const f_T * w,
      const f_T & p_tot , const f_T & b_x ) const
    {
        const auto & bx_2 = b_x * b_x  ;
        flux[ idx::rho ] = u[ idx::mom ];
        flux[ idx::ene ] = w[ idx::mom ]
              * ( p_tot  + u[ idx::ene ] - bx_2 ) - b_x 
              * ( w[ idx::mom + 1 ] * u[ idx::bcc + 1 ]
                + w[ idx::mom + 2 ] * u[ idx::bcc + 2 ] );
        flux[ idx::mom     ] =  p_tot -  bx_2 
                + u[ idx::mom     ] * w[ idx::mom     ];
        flux[ idx::mom + 1 ] = -b_x * u[ idx::bcc + 1 ]
                + u[ idx::mom + 1 ] * w[ idx::mom     ];
        flux[ idx::mom + 2 ] = -b_x * u[ idx::bcc + 2 ]
                + u[ idx::mom + 2 ] * w[ idx::mom     ];
        flux[ idx::bcc + 1 ] = -b_x * w[ idx::mom + 1 ]
                + u[ idx::bcc + 1 ] * w[ idx::mom     ];
        flux[ idx::bcc + 2 ] = -b_x * w[ idx::mom + 2 ]
                + u[ idx::bcc + 2 ] * w[ idx::mom     ];
    };    
    
    ////////// Operation //////////
    template < class f_T, class eos_T, class dat_T >
    __device__ void eval
    ( f_T * flx, f_T * wli, f_T * wri, const eos_T & f_eos,
      const dat_T & dat ) const;
};

////////////////////////////////////////////////////////////
// Riemann solver function

template< class derived_T >
template< class f_T,  class eos_T, class dat_T >
__device__ void hlld_t< derived_T >::eval
( f_T * flx, f_T * wli, f_T * wri, const eos_T & f_eos,
  const dat_T & dat ) const
{
    flx[ idx::bcc ] = 0;
    
    f_T _cwave        [ 5 ];     // Wave speeds
    f_T *  spd = _cwave + 2;     // -2, -1, 0, +1, +2

    static   const int n_wave  ( 8 );
    f_T u_l[ n_wave ], u_r[ n_wave ];

    const f_T  b_x    = wli [ idx::bcc ];
    const f_T  bx_2   = b_x * b_x;
    const auto f_cons =  [ & ]
    ( auto * u, auto * w, auto & gam1 )
    {
        f_T   p_m( bx_2 / 2 ), cs;
        for( int a = 1; a <= 2; ++ a )
        {
            const auto  n = idx::bcc + a;
            p_m   += w[ n ] * w[ n ] / 2;
            u[ n ] = w[ n ] ;
        }
        gam1 = 1 / f_eos.et_cs( u[ idx::ene ], cs, w );
        for( int n     = idx::mom; n < idx::mom + 3; ++ n )
            u[ n ]     = w[ idx::rho ] * w[ n ];
        u[ idx::rho ]  = w[ idx::rho ];
        u[ idx::ene ] += p_m;
        u[ idx::bcc ]  = b_x;
        return p_m;
    };

    f_T  gam1_l, gam1_r;
    const auto pm_l   = f_cons( u_l, wli, gam1_l );
    const auto pm_r   = f_cons( u_r, wri, gam1_r );
    const auto ptot_l = wli[ idx::ene ] + pm_l;
    const auto ptot_r = wri[ idx::ene ] + pm_r;

    f_T b2_eff_l[ 3 ] = { b_x, wli[ idx::bcc + 1 ]  ,
                               wli[ idx::bcc + 2 ] };
    f_T b2_eff_r[ 3 ] = { b_x, wri[ idx::bcc + 1 ]  ,
                               wri[ idx::bcc + 2 ] };
    for( int a = 0; a < 3; ++ a )
    {
        b2_eff_l[ a ] *= b2_eff_l[ a ];
        b2_eff_r[ a ] *= b2_eff_r[ a ];
    }

    const auto  cfm_l = f_eos.c_fms
                      ( wli, b2_eff_l, gam1_l + 1, 0 );
    const auto  cfm_r = f_eos.c_fms
                      ( wri, b2_eff_r, gam1_r + 1, 0 );

    spd[ -2 ] = utils::min( wli[ idx::mom ] - cfm_l ,
                            wri[ idx::mom ] - cfm_r );
    spd[  2 ] = utils::max( wli[ idx::mom ] + cfm_l ,
                            wri[ idx::mom ] + cfm_r );

    if( spd[ -2 ] >= 0 || spd[ 2 ] <= 0 )
    {
        const bool is_left( spd[ -2 ] >= 0 );
        f_flux( flx, is_left ?    u_l :    u_r ,
                     is_left ?    wli :    wri ,
                     is_left ? ptot_l : ptot_r, b_x );
        return;
    } 

    const auto   sd_l = spd[ -2 ] - wli[ idx::mom ];
    const auto   sd_r = spd[  2 ] - wri[ idx::mom ];
    spd[ 0 ] = ( sd_r * u_r[ idx::mom ] - ptot_r
               - sd_l * u_l[ idx::mom ] + ptot_l )
             / ( sd_r * u_r[ idx::rho ]
               - sd_l * u_l[ idx::rho ] ) ;

    const auto sdm_l  = spd[ -2 ] - spd[ 0 ];
    const auto sdm_r  = spd[  2 ] - spd[ 0 ];
    const auto rhos_l = u_l[ idx::rho ] * sd_l / sdm_l;
    const auto rhos_r = u_r[ idx::rho ] * sd_r / sdm_r;
    const auto rhos_l_sqrt = sqrt( rhos_l );
    const auto rhos_r_sqrt = sqrt( rhos_r );
    spd[ -1 ] = spd[ 0 ] - utils::abs( b_x ) / rhos_l_sqrt;
    spd[  1 ] = spd[ 0 ] + utils::abs( b_x ) / rhos_r_sqrt;


    static const f_T  f_tiny = ( f_T )( 1.00e-4f );
    const  auto  ptot_s = ptot_l + u_l[ idx::rho ] * sd_l
                        * ( sd_l - sdm_l );
    const auto f_star = [ & ]
    ( auto  *     u_star, const auto *  u, const auto *   w,
      const auto &  rhos, const auto & sd, const auto & sdm,
      const auto & p_tot )
    {
        u_star[ idx::rho ] = rhos;
        u_star[ idx::mom ] = rhos * spd[ 0 ];
        u_star[ idx::bcc ] =  b_x;

        const auto  denom = u[ idx::rho ] * sd * sdm - bx_2;
        if( utils::abs( denom ) < f_tiny * ptot_s )
            for( int a = 1; a <= 2; ++ a )
            {
                u_star[ idx::mom + a ] = rhos
                                       * w[ idx::mom + a ];
                u_star[ idx::bcc + a ] = u[ idx::bcc + a ];
            }
        else
        {
            const auto tmp_v = b_x *  ( sd - sdm ) / denom;
            const auto tmp_b
             = ( u[ idx::rho ] * sd * sd  - bx_2 ) / denom;
            for( int a = 1; a <= 2; ++ a )
            {
                u_star[ idx::mom + a ]
                    = rhos * ( w [ idx::mom + a ] -
                               u [ idx::bcc + a ] * tmp_v );
                u_star[ idx::bcc + a ]
                             = u [ idx::bcc + a ] * tmp_b  ;
            }
        }
        f_T u_dot_b( 0 ), w_dot_b( 0 );
        for( int a = 0; a < 3;  ++ a )
        {
            u_dot_b += u_star[ idx::mom + a ]
                     * u_star[ idx::bcc + a ];
            w_dot_b += w     [ idx::mom + a ]
                     * u     [ idx::bcc + a ];
        }
        u_dot_b /= rhos;
        u_star[ idx::ene ]  = b_x * ( w_dot_b - u_dot_b )
            - p_tot  * w[ idx::mom ] + sd * u[ idx::ene ]
            + ptot_s * spd[ 0 ];
        u_star[ idx::ene ] /=   sdm ;
        return u_dot_b;
    };

    f_T us_l [ n_wave ], us_r [ n_wave ];
    const auto u_dot_b_l = f_star
        ( us_l, u_l, wli, rhos_l, sd_l, sdm_l, ptot_l );
    const auto u_dot_b_r = f_star
        ( us_r, u_r, wri, rhos_r, sd_r, sdm_r, ptot_r );
    if     ( spd[ -1 ] >= 0 )
    {
        f_flux ( flx,   u_l, wli, ptot_l, b_x );
        for( int n = 0; n < n_wave; ++ n )
            flx[ n ] +=  spd [ -2 ]
                     * ( us_l[  n ] -  u_l[ n ] );
        return;
    }
    else if( spd[  1 ] <= 0 )
    {
        f_flux( flx,    u_r, wri, ptot_r, b_x );
        for( int n = 0; n < n_wave; ++ n )
            flx[ n ] +=  spd [  2 ]
                     * ( us_r[  n ] -  u_r[ n ] );
        return; 
    }

    f_T uss_l[ n_wave ], uss_r[ n_wave ];
    for( int n = 0; n < n_wave; ++ n )
    {
        uss_l[ n ] = us_l[ n ];
        uss_r[ n ] = us_r[ n ];            
    }
    if( bx_2 > ptot_s * f_tiny )
    {
        const int  bx_sgn( b_x > 0 ? 1 :  -1 );
        const auto rhos_norm_inv = 1.f
              /  ( rhos_l_sqrt + rhos_r_sqrt );
     
        for( int a = 1; a <= 2; ++ a )
        {
            f_T   tmp = rhos_norm_inv *
                ( us_l[ idx:: mom + a ] / rhos_l_sqrt
                + us_r[ idx:: mom + a ] / rhos_r_sqrt
                + bx_sgn * ( us_r[ idx::bcc + a ] 
                           - us_l[ idx::bcc + a ] ) );
            uss_l[ idx::mom + a ] = uss_l[ idx::rho ] * tmp;
            uss_r[ idx::mom + a ] = uss_r[ idx::rho ] * tmp;

            tmp = rhos_norm_inv *
                ( rhos_l_sqrt * us_r[ idx::bcc + a ]
                + rhos_r_sqrt * us_l[ idx::bcc + a ]
                + bx_sgn * rhos_l_sqrt * rhos_r_sqrt
                * ( us_r[ idx::mom + a ] / rhos_r
                  - us_l[ idx::mom + a ] / rhos_l ) );
            uss_l[ idx::bcc + a ] = tmp;
            uss_r[ idx::bcc + a ] = tmp;
        }
        const f_T tmp  = spd [ 0 ] * b_x +
           ( uss_l[ idx::mom + 1 ] * uss_l[ idx::bcc + 1 ] 
           + uss_l[ idx::mom + 2 ] * uss_l[ idx::bcc + 2 ] )
           / uss_l[ idx::rho ];

        uss_l[ idx::ene ]
            -= rhos_l_sqrt * bx_sgn * ( u_dot_b_l - tmp );
        uss_r[ idx::ene ]
            += rhos_r_sqrt * bx_sgn * ( u_dot_b_r - tmp );
    }
    if( spd[ 0 ] >= 0 )
    {
        f_flux ( flx,   u_l, wli, ptot_l, b_x );
        const auto d_v = spd[ -1 ] - spd[ -2 ];
        for( int n = 0; n < n_wave; ++ n )
            flx[ n ] += -spd[ -2 ] * u_l[  n ]
                - d_v * us_l[  n ] + spd[ -1 ] * uss_l[ n ];
    }
    else
    {
        f_flux ( flx,   u_r, wri, ptot_r, b_x );
        const auto d_v = spd[  1 ] - spd[  2 ];
        for( int n = 0; n < n_wave; ++ n )
            flx[ n ] += -spd[  2 ] * u_r[  n ]
                - d_v * us_r[  n ] + spd[  1 ] * uss_r[ n ];
    }
    return;
}

};                      // namespace dynamics::mhd::riemann
