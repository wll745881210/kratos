#include "../../chem_therm/chemistry.h"
#include "../radiation.h"
#include "integrator.h"

namespace extension::radiation
{
////////////////////////////////////////////////////////////
// Integrator

__host__ void itg_t::init
( const  input & args, particle::base_t & mod ) 
{
    auto & rad  = dynamic_cast
        < ::extension::radiation_t & >( mod );
    n_stride = rad.hv_eV.size(  )
             + rad.p_che->species.size(  );
    return super_t::init( args, mod );
}

};                     // namespace extension::radiation
