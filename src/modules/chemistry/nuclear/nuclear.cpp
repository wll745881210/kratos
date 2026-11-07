#include "../utilities/periodic.h"
#include "nuclear.h"
#include "../reaction/reaction_list.h"
#include "corrector.h"
#include "solver.h"
#include "eos.h"
#include <regex>

namespace chemistry::nuclear
{
////////////////////////////////////////////////////////////
// Chemistry module

void base_t::read( const input & args )
{
    if( ! intg )
    {
        using rls_t = chemistry::reaction::list::std_t
                    < reaction ::std_t<  > > ;
        using sol_t = solver::default_t
            < corrector::base_t, dat_t< type::float_t > >;
        using prx_t = chemistry::hydro::proxy_t;

        enroll< prx_t, rls_t, eos_t<  >, sol_t > (  );
        // ::chemistry::hydro::eos_t<  >, sol_t >(  );
    }
    return parse_species ( args );
}

void base_t::parse_species( const input & args )
{
    std::vector < std  ::string > sps  ;
    args( "chemistry", "species", sps );

    std::regex pat_e( "^[A-Z][a-z]*" );
    std::regex pat_n( "^\\d+"        );
    std::smatch    m;

    for( auto s : sps )
    {
        auto & sp = species.emplace_back(  );
        sp.name = s ;

        std::regex_search ( s, m,  pat_n );
        sp.elem[ 0 ] = std::stoi( m[ 0 ] );
        s = m.suffix(  ).str(  );

        std::regex_search ( s, m,  pat_e );
        const std::string & e( m[ 0 ] );
        s = m.suffix(  ).str(  );

// #warning "Should be re-enabled when Z is necessary?"
        // Now n_atom is actually z
        // sp.n_atom = ( std::regex_search( s, m, pat_n ) ?
        //               std::stoi( m[ 0 ] ) :
        //               periodic::elem( e ).z );

        sp.n_atom   = 1;
        sp.mass_cgs = phys::cgs::ma * sp.elem[ 0 ];
    }
    species.setup_idx(  );
    return;
}
};                       // namespace chemistry::nuclear
