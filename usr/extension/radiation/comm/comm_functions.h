#pragma once

namespace extension::radiation
{
////////////////////////////////////////////////////////////
//

template< class pol_T >
__host__ void comm_t< pol_T >::update
( const mesh::mod_base_t::v_reg_t & news,
  particle::               base_t & mod )
{
    auto & rad  = dynamic_cast
        < ::extension::radiation_t & >( mod );
    n_hv = rad.hv_eV.size(  );
    n_sp = rad.p_che->species.size(  );
    return super_t::update( news, mod );
}

};

