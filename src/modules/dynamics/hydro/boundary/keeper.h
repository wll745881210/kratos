#pragma once

#include "../../../../mesh/boundary/keeper.h"
#include "../../prototypes/integrator.h"
// #include "../hydro.h"
#include "phys.h"
#include "phys_functors.h"
#include "comm_functions.h"

namespace dynamics::hydro::boundary
{
////////////////////////////////////////////////////////////
// Boundary condition keeper

using  namespace    mesh;
using  holder_c_t = mesh::boundary::holder_c_t;
using  holder_c_w = std ::reference_wrapper< holder_c_t >;

struct queue_t
{
    std::vector< holder_c_w >  send;
    std::vector< holder_c_w >  recv;        
    std::vector< holder_c_w >  pack;
    std::vector< holder_c_w >  unpk;
};

class keeper_t : public mesh::boundary::keeper_t
{
    ////////// Initialization //////////
public:                         // Functions
    keeper_t(  );
    virtual void init( const input &, mesh_t & );
protected:                      // Data
    //  Dirichlet ( "fixd" ) boundary functor; its ambient
    //  state is filled from [init] in init.
    std::shared_ptr< fixed_t<  > > p_fixd;

    ////////// Actions //////////
protected:                      // Data
    std::vector< holder_c_w > q_same;
    queue_t                   q_ghst;
    queue_t                   q_plgt;
    queue_t                   q_flux;
protected:                      // Functions
    virtual bool desired_mode( const neighbor_t & nb );
    virtual void   exec_queue
    ( queue_t & q, const bool & comm, const int & step,
      const std::function< void(  ) > & f );
public:                         // Functions
    virtual void update ( mesh_t & );
    virtual void act_bnd( mesh_t & , const int & );
    virtual void act_flx( mesh_t & , const int & );
};

};  // namespace dynamics::hydro::boundary
