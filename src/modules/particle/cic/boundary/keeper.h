#pragma once

#include "../../../dynamics/hydro/boundary/keeper.h"
#include "comm_acc.h"
#include "comm_den.h"

namespace particle::cic::boundary
{
////////////////////////////////////////////////////////////
// Boundary keeper specific for CIC

class keeper_t : public dynamics::hydro::boundary::keeper_t
{
public:                         // Functions
    keeper_t(  ) = default;

    virtual void init( const input & args, mesh_t & mesh )
    {
        n_th_lim = args.get< int >
            ( "dynamics", "bnd_thread_limit", 64 );
        return mesh::boundary::keeper_t::init( args, mesh );
    };
    virtual bool desired_mode( const neighbor_t & nb )
    {
        return true;            // All nb modes needed
        // return nb.mode < 2;
    };
    virtual void act_bnd( mesh_t & mesh, const int & step )
    {
        p_com->barrier(  );  // Otherwise comm inconsistent
        exec_queue( q_ghst, false, step, [ & ] (  )
        {
            for( auto & h_w : q_same       )
                h_w.get(  ) . f_pack( step );
        }   );
        exec_queue( q_plgt, false, step, [ & ] (  ) {  } );
        return;
    };
};                    // class     keeper_t
};                    // namespace particle::cic::boundary
