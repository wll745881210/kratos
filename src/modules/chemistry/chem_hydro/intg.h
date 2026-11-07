#pragma once

#include "../../dynamics/hydro/hydro.h"

namespace chemistry::hydro
{
////////////////////////////////////////////////////////////
// Integrator: negative density fence w/ chemistry awareness

template< typename T >
using rk2_hy_t = dynamics::hydro::_t< T >;

template< template< class T > class base_T = rk2_hy_t >
struct itg_t : base_gen_t, base_T< itg_t< base_T > >
{
    ////////// Types //////////
    // __crtp_def_self__( rec_t, derived_T );
    using super_t = base_T< itg_t< base_T > >;

    ////////// Functions //////////
    
};

};                            // namespace chemistry::hydro



