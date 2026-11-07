#pragma once

#include "../../../../src/modules/particle/boundary/comm.h"
#include "../particle/particle_pool.h"

namespace extension::radiation
{
////////////////////////////////////////////////////////////
//

template< class pol_T >
struct comm_t
    : particle::comm::comm_t< pol_T, comm_t< pol_T > >
{
    ////////// Type and Data //////////
    using super_t
    = particle::comm::comm_t< pol_T, comm_t< pol_T > >;
    int n_hv;
    int n_sp;

    ////////// Functions //////////
    __host__ virtual void update
    ( const mesh::mod_base_t::v_reg_t & news,
      particle::               base_t & mod );

    __host__ virtual void set_mem( device::base_t & f_dev )
    {
        super_t::set_mem( f_dev );
        for( int n = 0; n < super_t::n_rank; ++ n )
            super_t::send_h[ n ].set_n_int( n_hv, n_sp );
        return;
    };
        
};
};

