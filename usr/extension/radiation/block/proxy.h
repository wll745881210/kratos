#pragma once

#include "../../proxy/proxy_base.h"

namespace extension::radiation
{
////////////////////////////////////////////////////////////
// 
struct prx_t : particle::radiation::proxy_t< block_data_t >,
               prx_base_t
{
    ////////// Type //////////
    using bdt_t = typename prx_t::bdt_t;

    ////////// Data //////////
    rad_t      rad;
    struct
    {
        int      n_hva;
        mesh::dat_3d_t< type::float2_t > u;
    }   hyd;
    
    ////////// Function //////////
    prx_t(  ) = default;
    
    template< class bdt_T > __forceinline__
    static auto  d( const bdt_T & bdt )
    {
        const auto & b_d
            = dynamic_cast< const bdt_t & > ( bdt );
        prx_t     res;
        res.rad = b_d.rad;
        return    res;
    };

    ////////// Specific for particle module proxy //////////
    __host__ virtual void setup
    ( const mesh::block::dual_t   &      d , 
      const std ::map< int, int > & m_rank ,
      particle  ::base_t          &    mod ) override;

    static __host__
    void set_module( mesh::mod_base_t & mod )
    {
        p_par = mod.weak_from_this(  );
        return;
    };        
};                      // struct prx_t
};                      // namespace extension::radiation
