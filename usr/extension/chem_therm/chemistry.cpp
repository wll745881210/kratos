#include "../../../src/utilities/mapping/mapping.h"
#include "../radiation/radiation.h"
#include "../hydro/proxy.h"
#include "solver/solver.h"
#include "chemistry.h"
#include "reaction/reaction_list.h"
#include "reaction/photoreac_prep.h"

namespace extension
{
////////////////////////////////////////////////////////////
//
using prx_t    =   hydro    ::       prx_t<  >;
using sol_t    =   chemistry::    solver_t<  >;
using eos_t    = ::chemistry::hydro::eos_t<  >;
using rls_t    =   chemistry::          list_t;
using rad_t    =                   radiation_t;

void chem_t::read( const input & args )
{
    // Read dust grain parameters into members before
    // species parsing
    r_dust_cgs   = args.get< type::float2_t >
        ( "chemistry", "r_dust_cgs",   1e-7 );
    rho_dust_cgs = args.get< type::float2_t >
        ( "chemistry", "rho_dust_cgs", 3e0  );

    // Read per-population overrides: dust_pop_C = 1e-5 3.0
    // Syntax: dust_pop_BASEELEM = radius_cm density_gcm3
    // Species without a specific entry (incl. "Gr") use
    // global defaults.
    for( const auto & kv : args.get_item_map(  ) )
        if( kv.first.find( "chemistry::dust_pop_" ) == 0 )
        {
            std::stringstream ss( kv.second );
            dust_pop_t dp{ r_dust_cgs, rho_dust_cgs };
            ss >> dp.r_cgs >> dp.rho_cgs;
            std::string base = kv.first.substr
                ( strlen( "chemistry::dust_pop_" ) );
            dust_pops[ base ] = dp;
        }

    parse_species( args );
    type::float2_t norm( 0 ), norm_m( 0 );
    for ( const  auto & s : species )
    {
        const auto x = args.get< type::float2_t >
                   ( "species_init", s.name, 1e-20 );
        norm   += x;
        norm_m += x * s.mass_cgs;
    }
    mu_ave_cgs = norm_m / norm;

    if( ! intg )
        enroll< prx_t, rls_t, eos_t,  sol_t > (  );
    h_cool_cgs = args.get< float_t >
               ( "chemistry", "h_cool_cgs", 1e9 );
    output_reac_rates = args.get< bool >
               ( "chemistry", "output_reac_rates", false );
    return;
};

void chem_t::init( const input & args, mesh::mesh_t & mesh )
{
    ::chemistry::base_t::init( args, mesh );
    auto & rls = dynamic_cast
         < chemistry::list_t & >( * p_rls );
    n_shield = rls.l_ph_shield;
    return;
}

void chem_t::step( mesh::mesh_t & mesh )
{
    ::chemistry::base_t::step ( mesh );
    return  this->sync_streams( mesh );
};

void chem_t::prep_rad( mesh::mesh_t & mesh )
{
    const auto & rls = dynamic_cast< rls_t & > ( * p_rls );
    const auto & eos = dynamic_cast< eos_t & > ( * p_eos );
    const auto & rad = dynamic_cast< rad_t & > ( * p_rad );
    const auto f_rsc = [ & ] ( const auto  & bdt )
    {
        dim3 n_bl;
        n_bl.x = bdt.geo.n_ceff[ 0 ];
        n_bl.y = bdt.geo.n_ceff[ 1 ];
        n_bl.z = bdt.geo.n_ceff[ 2 ];
        const int n_th = 32;
        const int n_sh = sizeof ( type::float_t )
                       * n_th * bdt.rad.n_fld(  );
        return std::make_tuple( n_bl, dim3( n_th ), n_sh );
    };
    for( auto & d : ( * this ) )
    {
        const auto & b_d( prx_t::d( d.d(  ) ) );
        p_dev->launch( ker_prep_rad, f_rsc( b_d ),
                       d.stream, b_d,  eos, rls );
    }
    return sync_streams( mesh );
}

//////////////////////////////////////////////////////////
// Output: optionally capture per-reaction rates per cell.

void chem_t::save
( mesh::mesh_t & mesh, binary_io::base_t & bio )
{
    if( output_reac_rates )
        compute_rates( mesh, bio );
    mod_base_t::save( mesh, bio );
    return;
}

void chem_t::finalize( mesh::mesh_t & mesh )
{
    if( output_reac_rates )
        free_reac_rates( mesh );
    return ::chemistry::base_t::finalize( mesh );
}

////////////////////////////////////////////////////////////
// Species parser that allow special cases


void chem_t::parse_species_single
( const input & args, const std::string & sp_str )
{
    if( sp_str.find( "Gr" ) != std::string::npos )
        return parse_species_gr( args, sp_str );
    return base_t::parse_species_single( args, sp_str );
}

void chem_t::parse_species_gr
( const input & args, const std::string & sp_str )
{
    // Backward-compatible "Gr" grain species parser.
    // Treats each "Gr" particle as one whole grain.
    if( ! elem_dat.contain( "Gr" ) )
    {
        const ::chemistry::elem_t
            gr( { "Gr", -1, 0, 0, 1, 4.5 } );
        elem_dat.enroll_elem( gr );
    }
    auto s( sp_str );
    auto & sp = species.emplace_back(  );
    sp.mass_cgs = 4 * 3.1416 / 3
        * pow( r_dust_cgs, 3 ) * rho_dust_cgs;
    sp.elem[ -1 ] = 1;
    // Element "-1": Dust grains "Gr".
    sp.n_atom     = 1;
    sp.name       = s;
    // Charged grains ( Gr+ / Gr- ):  the sign suffix carries
    // the net charge,  same convention as the base parser.
    int chg = utils::string::replace_all( s, "+", "" );
    chg    -= utils::string::replace_all( s, "-", "" );
    if( chg != 0 )
        sp.elem[ elem_dat.elem( "e" ).z ] = chg;
    return;
}

bool chem_t::is_grain_species
( const std::string & name ) const
{
    if( name.find( "Gr" ) != std::string::npos )
        return true;
    const auto reg_d
        = std::regex( ".*\\([dsm]\\)[+-]?[0-9]*" );
    if( std::regex_match( name, reg_d ) )
        return true;
    return false;
}

type::float_t chem_t::get_sigma_dust_cgs
( const std::string & name ) const
{
    auto params = get_dust_params( name );
    if( name.find( "Gr" ) != std::string::npos )
        return 3.1415927f * pow( params.r_cgs, 2 );
    // For parenthetical species: per-atom cross-section
    // = 3 * m_particle / (4 * r_dust * rho_dust).
    // This is derived from: sigma_grain = pi * r^2,
    // N_atom = m_grain / m_atom,
    // m_grain = (4*pi/3) * r^3 * rho =>
    // sigma_atom = sigma_grain / N_atom
    //   = pi * r^2 / ( (4*pi/3) * r^3 * rho / m_atom )
    //   = 3 * m_atom / (4 * r * rho).    
    auto idx = species.at( name );
    return 3.0f * species.at( idx ).mass_cgs
       / ( 4.0f * params.r_cgs * params.rho_cgs );
}

dust_pop_t chem_t::get_dust_params
( const std::string & name ) const
{
    dust_pop_t res{ r_dust_cgs, rho_dust_cgs };
    // Exact species name first ( enables multiple size
    // bins per base element, e.g. dust_pop_C(d)2 = r rho ),
    // then the base-element fallback: "C(d)+" -> "C".
    auto it = dust_pops.find( name );
    if( it != dust_pops.end(  ) )
        return it->second;
    auto base = std::regex_replace
        ( name, std::regex( "\\([dsm]\\)" ),  "" );
    base = std::regex_replace
        ( base, std::regex( "[+-]?[0-9]*$" ), "" );
    it = dust_pops.find( base );
    if( it != dust_pops.end(  ) )
        res = it->second;
    return res;
}

};                              // namespace chemistry
