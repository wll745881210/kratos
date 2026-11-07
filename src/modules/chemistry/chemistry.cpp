#include "../../utilities/mapping/loop.h"
#include "chem_hydro/base_gen.h"
#include "chemistry.h"
#include "reaction/reaction_list.h"

#include <regex>

namespace chemistry
{
////////////////////////////////////////////////////////////
// Chemistry module

void base_t::read( const input & args )
{
    if( ! intg )
        enroll<  >(  );
    return parse_species( args );
}

void base_t::init( const input & args, mesh::mesh_t & mesh )
{
    if( initialized )
        return;
    initialized = true;
    p_dev = mesh.p_dev;
    p_rls-> init( args, * this );
    p_sol-> init( args, * this );
    return  setup_mods  ( args );
}

void base_t::step( mesh::mesh_t & mesh )
{
    //  Solver kernels read what earlier modules prepared
    //  on other streams;  chemistry runs once per cycle.
    mesh.p_dev->sync_all_streams(  );
    for( auto  & d : ( * this ) )
         intg( d ) ;
    return sync_streams( mesh ) ;
}

void base_t::update( mesh::mesh_t & mesh )
{
    return; // By default, chemistry is parasitic to hydro
}

void base_t::finalize( mesh::mesh_t & mesh )
{
    p_sol       ->free( * p_dev );
    p_eos       ->free( * p_dev );
    return p_rls->free( * p_dev );
}

void base_t::parasite( std::weak_ptr< mod_base_t > q )
{
    q.lock(  )->parasite( weak_from_this(  ) );
    return mesh::mod_base_t::parasite   (  q );
}

////////////////////////////////////////////////////////////
// Chemical species parser

void base_t::parse_species( const input & args )
{
    std::vector < std  ::string > sps  ;
    args( "chemistry", "species", sps );
    for( const auto & s : sps )
        parse_species_single( args, s );
    return species.setup_idx(  );
}

void base_t::parse_species_single
( const input & args, const std::string & sp_str )
{
    auto s( sp_str );
    auto &  sp = species.emplace_back(  );
    sp.name = s ;

    utils          ::string::replace_all( s, "*", "" );
    int chg = utils::string::replace_all( s, "+", "" );
    chg    -= utils::string::replace_all( s, "-", "" );
    s       = std::regex_replace
                ( s, std::regex( "\\([^)]*\\)" ), "" );
    if( chg != 0 )
        sp.elem [ elem_dat.elem( "e" ).z ] = chg;
    if( s.compare( "e" ) == 0 )
    {
        sp.mass_cgs = phys::cgs::me;
        return;
    }

    std::regex pat_e( "[A-Z][a-z]*(?=\\d*)" );
    std::regex pat_n( "^\\d+" );
    std::smatch    m;
    while( ! s.empty (  ) )
    {
        if( ! std::regex_search( s, m, pat_e ) )
            break;
        const std::string e = m[ 0 ];
        s = m.suffix(  ).str(  );

        const auto z = elem_dat.elem( e ).z;
        if( sp.elem.find( z ) == sp.elem.end(  ) )
            sp.elem [ z ] = 0 ;
        if( std::regex_search ( s, m, pat_n ) )
        {
            sp.elem[ z ] += std::stoi( m[ 0 ] );
            s = m.suffix (  )  . str (  ) ;
        }
        else
            ++ sp.elem[ z ];
    }
    sp.mass_cgs = -chg * phys::cgs::me;
    sp.n_atom   = 0 ;
    for( const auto & [ z, n ] : sp.elem )
    {
        sp.mass_cgs += phys::cgs ::ma  * n
            * elem_dat.elem( z ).m;
        sp.n_atom   += ( z > 0 ) * n;
    }
    return;
}

};                              // namespace chemistry
