#pragma once

#include "../../indices.h"
#include "../../prototypes/riemann.h"
#include <type_traits>

namespace dynamics::hydro::riemann
{
////////////////////////////////////////////////////////////
// Derived riemann solver type for HLLC

template< class derived_T = crtp::dummy_t >
struct hllc_t : crtp::helper_t
< hllc_t, derived_T, ::dynamics::riemann::interface_t >
{
    __crtp_def_self__( hllc_t, derived_T );

    ////////// Driver for riemann solver //////////
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
        utils  ::idx_swap( ql  + idx::mom,     d.axis );
        utils  ::idx_swap( qr  + idx::mom,     d.axis );
        get_self(  ).eval( flx , ql,   qr, f_eos,   d );
        utils  ::idx_swap( flx + idx::mom, 3 - d.axis );
        return;
    };

    ////////// Aux Utilities //////////
    template< class f_T > __device__ __forceinline__
    f_T  v_2( const f_T * w ) const
    {
        return w[ idx::mom     ] * w[ idx::mom     ]
             + w[ idx::mom + 1 ] * w[ idx::mom + 1 ]
             + w[ idx::mom + 2 ] * w[ idx::mom + 2 ];
    };
    template< class F, class F2 >  __device__ void f_lr_hyd
    ( F * f, const F * w, const F2 & v, const F & e ) const
    {
        const auto rho = w[ idx::rho ];
        const auto pre = w[ idx::ene ];
        f[ idx::rho ]  = rho * v;
        f[ idx::ene ]  = pre * w[ idx::mom ] + e * v;
        for( int  a    = idx::mom; a < idx::mom + 3; ++ a )
             f [  a ]  = rho  * w[ a ] * v;
        f[ idx::mom ] += pre;
        return;
    };

    ////////// Operation //////////
    template< class f_T, class eos_T, class dat_T >
    __device__ void eval
    ( f_T * flx, f_T * wli, f_T * wri, const eos_T & f_eos,
      const dat_T & dat ) const;
};

////////////////////////////////////////////////////////////
// Riemann solver function

template< class derived_T >
template< class f_T, class eos_T, class dat_T > __device__
void hllc_t< derived_T >::eval
( f_T * flx, f_T * wli, f_T * wri, const eos_T & f_eos,
  const dat_T & d ) const
{
    const auto & self = get_self(  );
    f_T * fl   = wli;
    f_T * fr   = wri;
    f_T * wroe = flx;  // Temp variables for Roe averages

    //////////////////////////////
    // Step 0. Compute equivalent values for L/R
    f_T  el, er, cl, cr;
    const f_T gm1i_l = f_eos.et_cs( el, cl, wli );
    const f_T gm1i_r = f_eos.et_cs( er, cr, wri );

    //////////////////////////////
    // Step 1. Compute the Roe-averaged state
    const f_T sqrtdl  = __sqrt1     ( wli[ idx::rho ] );
    const f_T sqrtdr  = __sqrt1     ( wri[ idx::rho ] );
    const f_T isdlpdr = ( f_T ) 1 / ( sqrtdl + sqrtdr );
    wroe[ idx::rho ]  =               sqrtdl * sqrtdr  ;
    for( int  n   = idx::mom; n < idx::mom + 3; ++ n )
        wroe[ n ] = ( sqrtdl * wli[ n ] +
                      sqrtdr * wri[ n ] ) * isdlpdr;
    // hroe is the specific total enthalpy
    // (enthalpy + kinetic energy) in Roe's average
    const f_T hroe = isdlpdr *
            ( ( el + wli [ idx::ene ] ) / sqrtdl +
              ( er + wri [ idx::ene ] ) / sqrtdr );
    const f_T gam1_roe = ( f_T  ) 1 /
        ( isdlpdr * ( sqrtdl * gm1i_l + sqrtdr * gm1i_r ) );

    //////////////////////////////
    // Step 2. Compute sound speed in Roe-averaged states
    const f_T q = hroe - 0.5f * v_2( wroe );
    const f_T a = ( q <= 0 ) ? 0 : __sqrt1( gam1_roe * q );

    //////////////////////////////
    // Step 3. Compute the max/min wave speeds based on
    // L/R and Roe-averaged values
    const f_T al = __min1( wroe[ idx::mom ] -  a  ,
                           wli [ idx::mom ] - cl );
    const f_T ar = __max1( wroe[ idx::mom ] +  a  ,
                           wri [ idx::mom ] + cr );
    const f_T bp = ar > 0 ? ar : 0;
    const f_T bm = al < 0 ? al : 0;

    //////////////////////////////
    // Step 4. Compute the contact wave speed and
    // pressure
    const f_T vxl =  wli[ idx::mom ] - al;
    const f_T vxr =  wri[ idx::mom ] - ar;
    const f_T tl  =  wli[ idx::ene ] + vxl
                  *  wli[ idx::rho ] * wli[ idx::mom ];
    const f_T tr  =  wri[ idx::ene ] + vxr
                  *  wri[ idx::rho ] * wri[ idx::mom ];
    const f_T ml  =  wli[ idx::rho ] * vxl;
    const f_T mr  = -wri[ idx::rho ] * vxr;

    // am for the contact discontinuity (CD) speed
    // cp for the pressure at CD
    f_T cp, am;
    constexpr auto f_max = FLT_MAX / 1e1f;
    constexpr auto f_min = FLT_MIN * 1e1f;
    if( __abs1( ml + mr ) < f_min )
    {
        cp = 0;
        am = 0;
    }
    else if( ( tr != 0 &&
               __abs1( ml ) > f_max / __abs1( tr ) ) ||
             ( tl != 0 &&
               __abs1( mr ) > f_max / __abs1( tl ) ) )
    {
        const f_T mli = utils:: sgn ( ml ) /
                      __max1( __abs1( ml ) , f_min );
        const f_T mri = utils:: sgn ( mr ) /
                      __max1( __abs1( mr ) , f_min );
        am  = ( tl - tr ) / ( ml + mr ) ;
        cp  = __max1( ( mri * tr  + mli * tl )
                    / ( mli + mri ), ( f_T ) 0 );
    }
    else
    {
        am = ( tl - tr ) / ( ml + mr );
        cp = __max1( ( ml * tr  + mr * tl )
                   / ( ml + mr ), ( f_T ) 0 );
    }

    //////////////////////////////
    // Step 6. Compute flux weights
    const f_T d_am_bm = __max1( __abs1( am -  bm ), f_min )
                      *       ( am > bm ? 1 : -1 );
    const f_T d_am_bp = __max1( __abs1( am -  bp ), f_min )
                      *       ( am > bp ? 1 : -1 );
    const f_T sl = ( am >= 0 ?  am / d_am_bm : 0 );
    const f_T sr = ( am <  0 ?  am / d_am_bp : 0 );
    const f_T sm = ( am >  0 ? -bm / d_am_bm :
                               -bp / d_am_bp );


    //////////////////////////////
    // Step 5. Compute L/R fluxes along the line bm, bp
    f_lr_hyd( fl, wli, wli[ idx::mom ] - bm, el );
    f_lr_hyd( fr, wri, wri[ idx::mom ] - bp, er );

    //////////////////////////////
    // Step 7. Compute the HLLC fluxes at the interface,
    // including the weighted contribution of the flux
    // along the contact
    for( int n = idx::rho ; n < idx::mom + 3; ++ n )
        flx[ n ] = sl * fl[ n ]   +   sr * fr  [ n ];
    flx[ idx::mom ] += sm * cp;
    flx[ idx::ene ] += sm * cp * am;
    return ;
}

};  // namespace dynamics::hydro::riemann
