#pragma once

#include "../../dynamics/mhd/riemann/hlld.h"
#include "../../../utilities/phys/constants.h"
#include "../chem_hydro/base_gen.h"

namespace chemistry::mhd
{
//////////////////////////////////////////////////////////
// Riemann solver with chemistry (MHD)

template< typename T >
using rie_mh_t = dynamics::mhd::riemann::hlld_t< T >;

// HLLD core for the hydro-magnetic fields, plus passive
// upwind advection of the species.
//
// The species fluxes are NORMALIZED so that their mass content
// sums to the mass flux EXACTLY,
//     sum_n flx_n * mcgs_n  =  m_A * flx[ rho ].
// Without the normalization the reconstructed w can make the
// species - carried mass differ from the mass flux.
template< class derived_T = crtp::dummy_t >
struct rie_t : crtp::helper_t< rie_t, derived_T, rie_mh_t >
{
    ////////// Types //////////
    __crtp_def_self__( rie_t, derived_T );
    using super_t
        = crtp::helper_t< rie_t, derived_T, rie_mh_t >;

    ////////// Device-side interface //////////
    template< class f_T, class eos_T, class dat_T >
    __device__ void eval
    ( f_T * flx, f_T * wli, f_T * wri, const eos_T & f_eos,
      const dat_T & d ) const
    {
        super_t::eval  ( flx, wli, wri,  f_eos, d );
        const f_T * w ( flx[ idx::rho ] > 0 ? wli : wri );
        f_T  norm( 0 );
        for( int n = 0; n < f_eos.n_sps; ++ n )
            norm += w[ n + f_eos.n_hva ] * f_eos.mcgs( n );
        if( norm <= 0 || __isinf( norm )
                      || __isnan( norm ) )
            norm = 0;
        else
            norm = flx[ idx::rho ] / norm * phys::cgs::ma;
        for( int n = 0; n < f_eos.n_sps; ++ n )
            flx[ n + f_eos.n_hva ] = norm * w[ n + f_eos.n_hva ];
    };
};
};                         // namespace chemistry::mhd
