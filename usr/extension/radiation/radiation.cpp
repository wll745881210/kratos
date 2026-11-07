#include "../chem_therm/chemistry.h"
#include "integrate/generate.h"
#include "integrate/integrator.h"
#include "comm/comm_functions.h"
#include "block/proxy.h"
#include "particle/particle_pool.h"
#include "radiation.h"

namespace extension
{
////////////////////////////////////////////////////////////
//

void radiation_t::init_cond( mesh::block::dual_t & d )
{
    auto & bdt  = radiation::prx_t::ref ( d.h(  ) );
    const  auto & geo( bdt.geo );
    
    if( n_hv != bdt.rad.n_fld(  ) )
        throw std::runtime_error( "Inconsistent n_hv\n" );
    for( auto i : utils::loop( geo.n_ceff ) )
    {
        for( int n = 0; n < n_hv; ++ n )
        {
            bdt.rad. mfps_i.at( i )[ n ] = 0;
            bdt.rad. mfpa_i.at( i )[ n ] = 0;
        }
        for( int n = 0; n < n_sh; ++ n )
            bdt.rad. shield.at( i )[ n ] = 0;
    }
    return d.copy_h2d (  );
}

void radiation_t::read ( const input & args )
{
    is_first_run = true;
    particle::radiation::base_t::read( args ); 
    args( "radiation", "hv_eV", hv_eV );
    n_hv = hv_eV.size(  );

    using namespace extension::radiation;
    return enroll_extension
        < prx_t, pol_t, gen_t<  >, itg_t >(  );
}

void radiation_t::init
( const input & args, mesh::mesh_t & mesh )
{
    // mesh.p_dev->prep_rng(  );
    
    n_sp = p_che->species.size(  );
    n_sh = p_che->n_shield;
    return particle::radiation::base_t::init( args, mesh );
};

void radiation_t::save( mesh     ::mesh_t & mesh,
                        binary_io::base_t & bio )
{
    return particle::radiation::base_t::save( mesh, bio );
};

void radiation_t::step( mesh::mesh_t & mesh )
{
    mesh.p_cyc->redc_dt_start ( mesh );
    mesh.p_cyc->redc_dt_finish( mesh );    
    if( is_first_run )
    {
        f_reset( mesh );
        is_first_run = false;
        if( mesh.p_com->is_root(  ) )
            std::cout << "Conducting first ray-tracing for "
                      << "correct N_col...\n";
        particle::radiation::base_t::step( mesh );
    }
    p_che->prep_rad( mesh );
    f_reset        ( mesh );
    particle::radiation::base_t::step( mesh );
    return this->sync_streams( mesh );
};

};                          // namespace extension


