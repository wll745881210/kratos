#include "../../../utilities/mapping/loop.h"
#include "../chemistry.h"
#include "mhd.h"

namespace chemistry::mhd
{
//////////////////////////////////////////////////////////
// MHD for chemistry

void base_t::parasite
( std::weak_ptr< mesh::mod_base_t > q_che )
{
    this->q_che = std::dynamic_pointer_cast
        < ::chemistry::base_t > ( q_che.lock(  ) );
    return;
}

void base_t::read( const input & args )
{
    dynamics::mhd_t::read ( args );
    if( q_che.expired(  ) )
        throw std::runtime_error( "q_che unbound: mhd" );
    if( ! p_eos )
        throw std::runtime_error( "EoS is not enrolled." );
    q_che.lock(  )->p_eos = p_eos ;
    return p_eos->read( args ); // Not called in chemistry
}

void base_t::init( const input & args, mesh::mesh_t & mesh )
{
    q_che.lock(  )->init( args, mesh );
    this          -> read_chem( args );
    dynamics::mhd_t::init( args, mesh  );

    const auto & species = q_che.lock(  )->species;
    auto & rec = dynamic_cast
        < ::chemistry::hydro::base_gen_t & > ( * p_rec );
    rec. n_sps = species. size(  );
    rec. n_hva = p_eos->n_hvar(  );
    return rec. set_chem( args, species, * p_dev );
}

int base_t:: intra_order(  )
{
    return p_eos->n_hvar(  )
         + q_che.   lock(  )->species.size(  );
}

void base_t::enroll_default(  )
{
    if( ! f_bdt_yield )
        enroll_dat< block_data_t > (  );
    if( ! p_bdk )
        enroll_bdk
            < dynamics::mhd::boundary::keeper_t > (  );
    // No default enroll_chem_mhd: a chemistry EoS ( e.g.
    // nuclear::eos_t< mhd adiabatic > ) must be enrolled
    // explicitly by the problem generator.
    return dynamics::mhd_t::enroll_default (  );
}

//////////////////////////////////////////////////////////
// Initialization under chemistry

void base_t::read_chem( const input & args )
{
    if( x_spe.size(  ) > 0 )
        return;                 // Already initialized

    const auto & species( q_che.lock(  )->species );
    for ( const  auto & s : species )
        x_spe.push_back( args.get< type::float2_t >
                       ( "species_init", s.name, 1e-20 ) );
    // From number fraction to mass fraction
    type::float2_t norm( 0 );
    for( size_t i = 0; i < x_spe.size(  ); ++ i )
         norm +=  x_spe[ i ];

    mu_cgs_ave = 0;
    for( size_t i = 0; i < x_spe.size(  ); ++ i )
    {
        x_spe[ i ] /= norm;
        mu_cgs_ave += x_spe[ i ] * species[ i ].mass_cgs;
    }
    if( args .found( "init", "rho0_cgs" ) )
        rho0 = args.get< type::float2_t >
             ( "init", "rho0_cgs" ) / p_eos->unit.rho0;
    if( args .found( "init", "T0"       ) )
    {
        const auto T = args.get< float > ( "init", "T0" );
        pre0 = phys::cgs::kb * T * rho0 * p_eos->unit.rho0
             / mu_cgs_ave / p_eos->unit.ene0;
    }
    if( args .found( "init", "vel0_cgs" ) )
    {
        vel0 = args.get
             < type::coord2_t >( "init", "vel0_cgs" );
        for( auto & v : vel0.x )
            v /= p_eos->unit.vel0;
    }
    return;
}

void base_t::init_cond( mesh::block::dual_t & d )
{
    if( q_che.expired(  ) || ! q_che.lock(    )->p_eos )
        throw std::runtime_error( "q_che is unbound\n" );

    // rho, mom, ene = pre0 / ( gamma - 1 ) + KE + |B0|^2 / 2,
    // face-centered B0; the conserved cell-centered B0 is
    // set below so that e_g_cgs sees the same |B|^2.
    dynamics::mhd_t::init_cond( d );

    auto & b_h  = dynamic_cast
         < ::dynamics::mhd::block_data_t & >( d.h(  ) );
    const  auto & geo  ( d.b_w.get(  ).geo ) ;
    for( auto i : utils::loop( geo. n_ceff ) )
    {
        auto * u = b_h.hyd.u.at( i );
        u[ idx::rho ] = rho0;
        for( int n = 0; n <  x_spe.size(  ); ++ n )
            u[ n + p_eos->n_hvar(  ) ] = u [ idx::rho ]
                 * phys::cgs::ma / mu_cgs_ave * x_spe [ n ];
        for( int a = 0; a < 3; ++ a )
            u[ idx::bcc + a ] = b0[ a ];
    }
    return;
}

};                             // namespace chemistry::mhd
