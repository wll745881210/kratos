#include "../operation/operation.h"
#include "comm.h"
#include "keeper.h"

namespace multigrid::boundary
{
////////////////////////////////////////////////////////////
// Initializer

void keeper_t::init( const input & args, mesh::mesh_t & m )
{
    n_th_lim = args.get< int >
           ( "multigrid", "thread_limit",  64 );
    return mesh::boundary::keeper_t::init( args,   m );
}

void keeper_t::assign_comm( block_t & b, mesh_t & mesh )
{
    mesh::boundary::keeper_t::assign_comm( b, mesh );
    for( auto & [ nb, h ] : bnd_comm[ b.reg ] )
    {
        h.tag_s *= 100;
        h.tag_r *= 100;
    }   // Avoid double-send while keep async
    return;
}

////////////////////////////////////////////////////////////
// Action

bool keeper_t::desired_mode( const mesh::neighbor_t & nb )
{
    return ( nb.mode < 1 );
}

void keeper_t::act_bnd_l( mesh::mesh_t &  m, const int & l,
                          const dlvl_t & dl )
{
    for( auto & [ reg, v_nb ] : bnd_comm )
        for( auto  & p : v_nb )
            if( p.second.f_pack )
            {
                set_action( p.second, dl );
                p.second.f_pack      ( l );
            }
    p_com->wait_all_d(  );
    p_mod->sync_streams( m );
    // for( auto & d : ( * p_mod ) )
    //     p_dev->sync_stream( d.stream );
    for( auto & [ reg, v_nb ] : bnd_comm )
        for( auto  & p : v_nb )
            if( p.second.f_unpk )
            {
                set_action( p.second, dl );
                p.second.f_unpk      ( l );
            }
    return;
}

};                         // namespace multigrid::boundary
