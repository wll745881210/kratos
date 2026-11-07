#pragma once

#include "../../../src/modules/dynamics/hydro/boundary/keeper.h"

#include <functional>

namespace extension::immersed
{
////////////////////////////////////////////////////////////
// Boundary keeper with a global immersed-boundary pass
//
// The regular communication / physical boundaries are applied
// first (so the primitive ghost cells are current); then a single
// pass over every block conditions the interface cells of the
// immersed body.  Running the pass here -- after the global c2p
// and before any flux computation -- guarantees that every block
// is conditioned before a neighbour can read one of its interface
// cells as a ghost cell during the flux reconstruction.

struct keeper_t : dynamics::hydro::boundary::keeper_t
{
    ////////// Types //////////
    using super_t = dynamics::hydro::boundary::keeper_t;

    ////////// Callback //////////
    std::function
    < void( mesh::mesh_t &, const int & ) > act_imm;

    ////////// Actions //////////
    virtual void act_bnd
    ( mesh::mesh_t & mesh, const int & step ) override
    {
        // 1) The original boundary run: communication, physical
        //    boundaries and prolongation, exactly as in the base.
        super_t::act_bnd( mesh, step );
        if( act_imm )
        {
            // 2) The immersed boundary, strictly AFTER the original
            //    run: one pass over every block conditions the
            //    interface cells with current fluid ghosts on hand.
            act_imm( mesh, step );
            // 3) Communication-only refresh of the ghost exchange so
            //    the ghosts carry the conditioned interface state
            //    when the fluxes are evaluated: same-level ghosts
            //    ( q_same ), the fine-to-coarse restriction ( q_ghst )
            //    and the coarse-to-fine prolongation ( q_plgt ) are
            //    all re-applied.  The physical boundaries are NOT
            //    re-run ( they write ghost cells only, which the
            //    conditioning -- an interior-cell overwrite --
            //    never touches ).
            exec_queue( q_ghst, false, step, [ & ](  )
            {
                for( auto & h_w : q_same )
                    h_w.get(  ).f_pack( step );
            }   );
            exec_queue( q_plgt, false, step, [ & ](  ) {  } );
        }
        return;
    };
};

};                              // namespace extension::immersed
