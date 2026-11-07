#pragma once

#include "../../../io/args/input.h"
#include "../../../utilities/types/crtp.h"
#include "../../../utilities/mapping/mapping.h"

namespace dynamics::riemann
{
////////////////////////////////////////////////////////////
// Base class for the Riemann solver

struct base_t
{
    ////////// Initialization //////////
    __host__ virtual void init( const input & args ) {  };
};

template< class derived_T = crtp::dummy_t >
struct interface_t : public base_t     {  };
};                           // namespace dynamics::riemann
