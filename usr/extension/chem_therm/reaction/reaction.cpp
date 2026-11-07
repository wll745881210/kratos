#include "../../algo/interp.h"
#include "../chemistry.h"
#include "reaction.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Derived reaction type

__host__ bool reac_t::parse
( const std::string & label, const input & rarg,
  const ::chemistry::base_t & chem )
{
    if( label.find( "reaction" ) == std::string::npos )
        return false;    
    auto kind = rarg.get< std::string >
        ( label, "rate_type", "standard" );
    for( auto & s_s : kind )
        s_s = std::tolower( s_s );    
    const auto f_kind = [ & ] ( const auto & sub )
    {
        return kind.find( sub ) != std::string::npos;
    }   ;
    if( f_kind( "arr" ) ||
        f_kind( "std" ) || f_kind( "stand" ) )
        this->kind = standard;
    else if( f_kind( "intp" ) || f_kind( "inter" ) )
        this->kind =   interp;
    else if( f_kind( "troe" ) )
        this->kind =     troe;
    else
        return false;
    const auto res = super_t::parse( label, rarg, chem );
    if( res && this->kind == interp )
        n_intp = n_pars / 2;
    return res;
}

};                      // namespace extension::chemistry
