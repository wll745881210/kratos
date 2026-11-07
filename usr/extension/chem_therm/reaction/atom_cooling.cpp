#include "../chemistry.h"
#include "atom_cooling.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Enum of cooling kinds

__host__ bool atom_cool_t::parse
( const std::string & label, const input & rarg,
  const ::chemistry::base_t & chem )
{
    const auto type = rarg.get< std::string >
        ( label, "rate_type", "" );
    if( type.find( "atom_cool"  ) == std::string::npos )
        return false;
    return super_t::parse( label, rarg , chem );
}

__host__ void atom_cool_t::print
( const ::chemistry::sp_vec_t & sps )
{
    std::cout << sps[ i_chem[ 0 ] ].name << " : pars = [ ";
    for( int i = 0; i < n_pars; ++ i )
        std::cout << pars[ i ]
                  << ( i < n_pars - 1 ? ", " : " ]\n" );
    return;
}
};                       // namespace extension::chemistry
