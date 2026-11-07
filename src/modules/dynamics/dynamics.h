#pragma once

#include "../../types.h"
#include "../../io/args/input.h"
#include "../../device/device.h"
#include "../../mesh/mesh.h"
#include "hydro/types/block_data.h"
#include "prototypes/prototypes.h"
#include <memory>

namespace dynamics
{
////////////////////////////////////////////////////////////
// Base class for (magneto-)hydrodynamics

class base_t : public mesh::mod_base_t
{
    ////////// Initializer //////////
public:                         // Functions
    base_t  (  ) {  };
    virtual void read( const input & );
    virtual void init( const input & , mesh::mesh_t & );
    virtual void finalize            ( mesh::mesh_t & );
    
    ////////// Modules //////////
public:                         // Modules
    std::shared_ptr< eos        ::base_t > p_eos;
    std::shared_ptr< riemann    ::base_t > p_rie;
    std::shared_ptr< integrator ::base_t > p_itg;
    std::shared_ptr< reconstruct::base_t > p_rec;
protected:                      // Functors
    std::function
    < void( mesh::mesh_t &, const int & ) > integrate;
protected:                      // Functions
    virtual void enroll_default(  )  {  };
public:                         // Functions
    virtual void step  ( mesh::mesh_t  & mesh );
public:                         // Interfaces
    virtual int intra_order(  );
    virtual int space_order(  );
    virtual int  time_order(  );
    
    template< class prx_T, class itg_T, class eos_T ,
              class rec_T, class rie_T, class com_T >    
    void enroll(  ) ;
};

////////////////////////////////////////////////////////////
// Enrolling integration modules via template

template< class prx_T, class itg_T, class eos_T ,
          class rec_T, class rie_T, class com_T >
void base_t::enroll (  )
{
    p_eos = std::make_shared< eos_T > (  );
    p_rec = std::make_shared< rec_T > (  );
    p_rie = std::make_shared< rie_T > (  );
    p_itg = std::make_shared< itg_T > (  );

    integrate  = [ & ] ( mesh ::mesh_t & mesh ,
                         const     int &  stp )
    {
        auto & eos = dynamic_cast< eos_T & > ( * p_eos );
        auto & rec = dynamic_cast< rec_T & > ( * p_rec );
        auto & rie = dynamic_cast< rie_T & > ( * p_rie );
        auto & itg = dynamic_cast< itg_T & > ( * p_itg );
        auto & cyc = ( * mesh. p_cyc );
        auto & dev = ( * this->p_dev );

        for( auto & d : ( * this ) )
        {
            const auto & b_d ( prx_T::d( d.d(  )  ) );
            eos.cons2prim    ( b_d,     d, stp, dev );
            dev.event_record ( d  . event, d.stream );
            dev.event_wait   ( cyc.stream, d. event );
        }
        cyc.redc_dt_start ( mesh ) ;
        p_bdk ->act_bnd   ( mesh, stp );
        cyc.redc_dt_finish( mesh ) ;
        for( auto & d : ( * this ) )
        {   // v-- reference if necessary
            auto & b_d0 = d.d(  );
            b_d0.   t = cyc .t;
            b_d0.dt_0 = cyc .dt (   );
            b_d0.dt   = b_d0.dt_0 * itg.step_size ( stp );
            itg.calc_flx( prx_T::d( b_d0 ), d, eos, rec,
                          rie, stp,  dev );
            dev.event_record( d.event, d.stream  );
        }
        p_bdk->act_flx( mesh,  stp );
        for( auto & d : ( * this ) )
        {
            decltype( prx_T::d( d.d(  ) ) )
                 b_d( prx_T::d( d.d(  ) ) );
            b_d. t =  cyc.t;
            b_d.dt =  cyc.dt(  ) * itg.step_size( stp );
            itg.intg_flx( b_d, d, eos, stp, dev );
        }
        p_dev->sync_all_streams (  );
    };
    p_bdk->enroll_comm< com_T > (    );
    return prx_T::set_module( * this );
}

};                              // namespace dynamic
