#pragma once

#include "../../dynamics/hydro/riemann/hllc.h"
#include "../../../utilities/phys/constants.h"
#include "base_gen.h"

namespace chemistry::hydro
{
////////////////////////////////////////////////////////////
// Riemann solver with chemistry

template< typename T >
using rie_hy_t = dynamics::hydro::riemann::hllc_t< T >;

template< class derived_T = crtp::dummy_t >
struct rie_t :
       crtp::helper_t< rie_t, derived_T, rie_hy_t >
{
    ////////// Types //////////
    __crtp_def_self__( rie_t, derived_T );
    using super_t
        = crtp::helper_t< rie_t, derived_T, rie_hy_t >;

    template< class f_T, class eos_T, class dat_T >
    __device__ void eval
    ( f_T * flx, f_T * wli, f_T * wri, const eos_T & eos,
      const dat_T & d ) const
    {
        const f_T rhl = wli[ idx::rho ] ;
        const f_T rhr = wri[ idx::rho ] ;
        super_t::eval ( flx, wli, wri, eos, d );
        const f_T * w ( flx[ idx::rho ] > 0 ? wli : wri );
        
        f_T  norm( 0 );
        for( int n = 0; n < eos.n_sps; ++ n )
            norm += w[ n + eos.n_hva ] * eos.mcgs( n );
        if( norm <= 0 || __isinf( norm )
                      || __isnan( norm ) )
            norm = 0;
        else
            norm = flx[ idx::rho ] / norm * phys::cgs::ma;
        for( int n = 0; n < eos.n_sps; ++ n )
            flx[ n + eos.n_hva ]
                 = norm * w[ n +  eos.n_hva ];
        return;
    };
};
};                         // namespace chemistry::hydro
