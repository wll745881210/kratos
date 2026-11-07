#include "../../chem_therm/chemistry.h"
#include "../../hydro/hydro.h"
#include "../radiation.h"
#include "proxy.h"

#include <sstream>

namespace extension::radiation
{
////////////////////////////////////////////////////////////
//

__host__ void prx_t::setup
( const mesh::block::dual_t   &      d ,
  const std ::map< int, int > & m_rank ,
  particle  ::base_t          &    mod )
{
    auto & b_d( ref( d.d (  ) ) );
    rad  = b_d. rad;

    this->hyd.u = d_hyd( b_d.reg ).hyd.u;    
    auto &  rad = dynamic_cast< radiation_t & > ( mod );
    this->hyd.n_hva = rad.p_hyd->p_eos->n_hvar(  );
    
    return particle::proxy_base_t::setup( d, m_rank, mod );
};  // This is for particle modules only, not generic!

};                     // namespace extension::radiation
