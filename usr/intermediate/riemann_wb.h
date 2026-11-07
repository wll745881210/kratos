#pragma once

#include "../../src/modules/dynamics/mhd/riemann/hlld.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  Well - balanced HLLD variant:  the "pressure - central"
//  rule.
//
//  The two reconstructed states handed to the HLLD are given a
//  COMMON pressure ( their arithmetic mean ) before the solve.
//  With unequal states the HLLD star pressure is the
//  density - weighted mean ( ptot_s = ( rho_l ptot_r +
//  rho_r ptot_l ) / ( rho_l + rho_r ) for a static interface ),
//  i.e. it is biased by the density jump;  with two EQUAL
//  pressures that bias vanishes identically and the momentum
//  flux carries exactly the reconstructed face pressure.
//  That matters here because the disk equilibrium needs
//  d p / d r to cancel the gravity + centrifugal sources to
//  ~3e-3,  so a bias of O( density jump ) in the pressure is
//  amplified ~300x ( measured:  the disk first-step radial
//  residual is ~3e4 times the uniform-flow one ).
//
//  This is the same idea as Cargo - LeRoux / the "deviation"
//  well - balancing,  applied at the Riemann stage instead of
//  the reconstruction stage:  the solver is not allowed to
//  invent a pressure imbalance that the source terms do not
//  know about.
//
//  LIMITATION:  it deliberately removes the pressure jump the
//  upwinding would otherwise keep,  so it is NOT shock - robust
//  in the way the stock HLLD is;  it is meant for smooth,
//  near - equilibrium stratified flows.  It is OPTIONAL:  the
//  pgen selects it by filling position 5 of mhd_t::enroll
//  ( [prob] riemann = 1 ).  Nothing under src/ is modified.

struct hlld_wb_t
    : dynamics::mhd::riemann::hlld_t< hlld_wb_t >
{
    using super_t = dynamics::mhd::riemann::hlld_t< hlld_wb_t >;

    template< class f_T, class dat_T, class bdt_T, class eos_T >
    __forceinline__ __device__ void calc
    ( f_T * flx, f_T *  ql, f_T * qr, const dat_T & d,
      const bdt_T & b_d, const eos_T & f_eos )  const
    {
        const auto & axis( d.axis );
        utils::idx_swap( ql + idx::mom, axis );
        utils::idx_swap( qr + idx::mom, axis );
        utils::idx_swap( ql + idx::bcc, axis );
        utils::idx_swap( qr + idx::bcc, axis );
        //  slot idx::ene holds the PRESSURE and is not swapped.
        const f_T p_face = ( f_T )( 0.5 )
                         * ( ql[ idx::ene ] + qr[ idx::ene ] );
        ql[ idx::ene ] = p_face;
        qr[ idx::ene ] = p_face;
        this->eval( flx, ql, qr, f_eos, d );
        utils::idx_swap( flx + idx::mom, 3 - axis );
        const f_T ef[ 2 ] = { flx[ idx::bcc + 2 ],
                            - flx[ idx::bcc + 1 ] };
        flx[ idx::bcc + utils::mod3( axis + 1 ) ] = ef[ 0 ];
        flx[ idx::bcc + utils::mod3( axis + 2 ) ] = ef[ 1 ];
        flx[ idx::bcc +              axis       ] =       0;
        return;
    };
};

};                              // namespace interm