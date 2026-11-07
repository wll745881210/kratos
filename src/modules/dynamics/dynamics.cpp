#include "dynamics.h"

namespace dynamics
{
////////////////////////////////////////////////////////////
// Initializer

void base_t::read( const input & args )
{
    if( ! integrate )
        enroll_default(   );
    return;
}

void base_t::init( const input & args, mesh::mesh_t & mesh )
{
    p_dev      = mesh. p_dev ;
    p_bdk->init( args, mesh );
    p_rec->init( args );
    p_rie->init( args );
    p_itg->init( args, * this );
    p_eos->read( args );
    return update   (  mesh );
}

void base_t::finalize( mesh::mesh_t & mesh )
{
    p_itg->finalize         ( * this );
    p_eos->free       ( * mesh.p_dev );
    mesh::mod_base_t::finalize( mesh );
    return;
}

////////////////////////////////////////////////////////////
// Interface

void base_t::step( mesh::mesh_t & mesh )
{
    for( int  step =  0; step < time_order(  ); ++ step )
        this->integrate( mesh , step );
    return;
}

int base_t::intra_order(  )
{
    return p_eos ? p_eos->n_hvar(  ) : 2;
}

int base_t::space_order(  )
{
    return  p_rec ?
        ( ( p_rec->space_order(  ) - 1 ) / 2 + 1 ) * 2 : 0;
}   // Calibrate to even: 1-->2, 2-->2, 3-->4, 4-->4, ...

int base_t:: time_order(  )
{
    return p_itg ? p_itg-> time_order(  ) : 1;
}

};                              // namespace dynamic
