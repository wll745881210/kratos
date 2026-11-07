#pragma once

#include "../dynamics.h"
#include "boundary/keeper.h"
#include "integrator/rk2.h"
#include "eos/adiabatic.h"
#include "reconstruct/plm.h"
#include "riemann/hllc.h"
#include "types/block_data.h"

namespace dynamics
{
////////////////////////////////////////////////////////////
// Hydrodynamics

class hydro_t : public base_t
{
protected:                      // Initials
    type ::float2_t             rho0;
    type ::float2_t             pre0;
    type ::coord2_t             vel0;
    void read_chem( const input &  );
public:                         // Functions
    virtual void    read             ( const input & );
    virtual void    enroll_default   (               );
    virtual void    init_cond( mesh::block::dual_t & );

    template
    < class prx_T = hydro::proxy_t,
      class itg_T = hydro::integrator :: rk2_t<       > ,
      class eos_T = hydro::eos   ::adiabatic_t<       > ,
      class rec_T = hydro::reconstruct:: plm_t<       > ,
      class rie_T = hydro::riemann    ::hllc_t<       > ,
      class com_T = hydro::boundary   ::comm_t< prx_T > >
    void enroll(  )
    {
        if( ! f_bdt_yield )
            enroll_dat  < typename  prx_T::bdt_t > (  );
        block_yield = [ & ]
        ( mesh::block_t & b, mesh::mesh_t & mesh )
        {
            auto p = block_yield_base ( b , mesh );
            auto & b_h   =  prx_T::ref( p->h(  ) );
            auto & b_d   =  prx_T::ref( p->d(  ) );
            b_d.geo.n_gh       = space_order(    );
            b_h.geo.n_gh       = space_order(    );
            b_d.hyd.order(  )  = space_order(    );
            b_h.hyd.order(  )  = space_order(    );
            b_d.hyd.n_fld(  )  = intra_order(    );
            b_h.hyd.n_fld(  )  = intra_order(    );
            b_d.hyd.n_stp(  )  =  time_order(    );
            b_h.hyd.n_stp(  )  =  time_order(    );
            return p ;
        };
        return base_t::enroll< prx_T, itg_T, eos_T ,
                               rec_T, rie_T, com_T > (  );
    };
};
};                              // namespace dynamic
