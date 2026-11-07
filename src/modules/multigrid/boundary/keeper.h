#pragma once

#include "../../../mesh/boundary/keeper.h"
#include "../types/enum.h"

namespace multigrid::boundary
{
////////////////////////////////////////////////////////////
// Boundary condition keeper

using namespace             mesh;
using mesh::boundary::holder_c_t;

class keeper_t : public mesh::boundary::keeper_t
{
    ////////// Initialization/finalization //////////
public:                         // Functions
    virtual void init       ( const input &, mesh_t & );
    virtual void assign_comm( block_t &,     mesh_t & );

    ////////// Actions //////////
protected:                      // Functions
    std::function
    < void( holder_c_t &, const dlvl_t & ) >   set_action;
    virtual bool desired_mode( const mesh::neighbor_t & );
public:                         // Functions
    virtual void act_bnd_l( mesh::mesh_t & , const  int & ,
                            const dlvl_t & );
    template< class com_T >
    void enroll_comm(   )
    {
        set_action = [ & ]
        ( holder_c_t & h, const dlvl_t & dl )
        {
            auto & bcom = dynamic_cast< com_T & >( * h.p );
            bcom.just_restrict = ( dl == dlvl_t::coarser );
        };
        mesh::boundary::keeper_t::enroll_comm< com_T > (  );
        return;
    };

};

};                         // namespace multigrid::boundary
