#pragma once

#include "../particle.h"
#include "boundary/keeper.h"
#include "block_data.h"
#include "integrator.h" 
#include "particle.h"
#include "proxy.h"

namespace particle::cic
{
////////////////////////////////////////////////////////////
// Base type for radiation

class base_t : public particle::base_t
{
protected:                      // Data
    std::shared_ptr< boundary::keeper_t > q_bdk; // bdk acc
    int n_fld;                  // # of fields
    std::function< void( mesh::mesh_t & ) > par_gen;
public:                         // Function
    virtual void read( const input & args )
    {
        n_fld = 1;
        if( ! f_bdt_yield )
            enroll_dat   < block_data_t > (  );
        return particle::base_t::read ( args );
    };
    virtual void init( const input & args, mesh_t & mesh )
    {
        q_bdk->init( args, mesh );        
        return particle::base_t::init( args, mesh );
    };

    virtual void finalize( mesh_t & mesh )
    {
        if( q_bdk )
            q_bdk->finalize(  );
        return particle::base_t::finalize( mesh );
    };

    virtual void step  ( mesh::mesh_t & mesh )
    {
        q_bdk->act_bnd ( mesh, 0 );
        return particle::base_t::step ( mesh );
    };

    using   particle   ::base_t::update;    
    virtual void update( mesh_t & mesh )
    {
        particle::base_t::update( mesh );
        par_gen                 ( mesh );
        return    q_bdk ->update( mesh );
    };

    virtual comm_mode_t  comm_mode(  )
    {
        return keep_pool;
    };

    template < class prx_T = proxy_t           <    >,
               class pol_T = pool_t< particle_t<  > >,
               class gen_T = generate  ::base_t<  >  ,
               class itg_T = intg_t            <  >  ,
               class map_T = block_map_t < prx_T  >  ,
               class com_T = comm::comm_t< pol_T  >  >
    void enroll(  )
    {
        if( ! block_yield )
            block_yield = [ & ]
        ( mesh::block_t & b, mesh::mesh_t & mesh )
        {
            auto p = block_yield_base ( b , mesh );
            auto & b_h   =  prx_T::ref( p->h(  ) );
            auto & b_d   =  prx_T::ref( p->d(  ) );
            b_h.cic.n_fld(  ) = n_fld;
            b_d.cic.n_fld(  ) = n_fld;
            b_h.cic.order(  ) = space_order(  );
            b_d.cic.order(  ) = space_order(  );
            return p ;
        };
        q_bdk = enroll_bdk< boundary::keeper_t > (   );
        enroll_bdk        < boundary::keeper_t > (   );
        p_bdk ->enroll_comm
              < boundary::den::comm_t< prx_T > > (   );
        q_bdk ->enroll_comm
              < boundary::acc::comm_t< prx_T > > (   );
        if( ! par_gen )
            par_gen = [ & ] ( mesh::mesh_t & mesh )
        {
            auto & pol = dynamic_cast< pol_T & >( * p_pol );
            auto & gen = dynamic_cast< gen_T & >( * p_gen );
            auto & map = dynamic_cast< map_T & >( * p_map );
            gen.generate ( pol, map, * this ) ;
        };
        return particle::base_t::enroll< prx_T , pol_T,
                   gen_T, itg_T,  map_T, com_T > (   );
    }
};
};                       // namespace particle::cic
