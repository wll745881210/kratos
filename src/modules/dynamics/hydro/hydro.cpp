#include "../../../utilities/mapping/loop.h"
#include "types/block_data.h"
#include "hydro.h"

namespace dynamics
{
////////////////////////////////////////////////////////////
// Default initialization for a typical hydrodynamic solver

void hydro_t::read( const input & args )
{
    rho0 = args.get< type::float2_t >( "init", "rho0", 0 );
    pre0 = args.get< type::float2_t >( "init", "pre0", 0 );
    vel0 = args.get< type::coord2_t >( "init", "vel0", 0 );
    return dynamics::base_t::read( args );
}

void hydro_t::enroll_default(  )
{
    if( ! f_bdt_yield )
        enroll_dat< hydro::      block_data_t > (  );
    if( ! p_bdk )
        enroll_bdk< hydro::boundary::keeper_t > (  );
    return;
}

void hydro_t::init_cond( mesh::block::dual_t & d )
{
    auto & b_h = dynamic_cast
               < hydro::block_data_t & > ( d.h(  ) );

    const  auto & geo ( d.b_w.get(  ).geo ) ;
    for( auto i : utils::loop( geo.n_ceff ) )
    {
        auto * u = b_h.hyd.u.at( i );
        u[ idx::rho ] = rho0;
        u[ idx::ene ] = pre0 / ( p_eos->gamma - 1 );
        for( int n = 0; n < 3; ++ n )
        {
            u[ idx::ene ] += pow( vel0[ n ], 2 ) * rho0 / 2;
            u[ idx::mom + n  ]  = vel0[ n ]      * rho0 ;
        }
    }
    return;
}
};                              // namespace dynamics
