#pragma once

#include "../../../src/mesh/module_base/module_base.h"
#include "../../../src/modules/multigrid/multigrid.h"
#include "../../../src/modules/particle/cic/cic.h"
#include "../radiation/block/block_data.h"

namespace extension
{
////////////////////////////////////////////////////////////
// Base of all proxies

using mesh::    mod_base_t;
using mesh::region_logic_t;

struct prx_base_t
{
    static std::weak_ptr< mod_base_t >  p_mgd; // Dyn
    static std::weak_ptr< mod_base_t >  p_hyd; // Hydro
    static std::weak_ptr< mod_base_t >  p_par; // Particle
    
    static auto & d_mg
    ( const region_logic_t & reg,
      const std::weak_ptr< mod_base_t > & p_mg )
    {
        return dynamic_cast
             < multigrid::data::block_data_t  & >
             ( p_mg.lock(  )->data( reg ).d(  ) );
    };
    static auto & d_mgd( const region_logic_t & reg )
    {
        return d_mg( reg, p_mgd );
    };

    static auto & d_par( const region_logic_t & reg )
    {
        return dynamic_cast
             < radiation::block_data_t  &  >
             ( p_par.lock(  )->data( reg ).d(  ) );
    };

    static auto & d_hyd( const region_logic_t & reg )
    {
        return dynamic_cast
             < ::dynamics::hydro::block_data_t & >
             ( p_hyd.lock(  )->data( reg ).d(  ) );
    };            
};

};                              // namespace extension
