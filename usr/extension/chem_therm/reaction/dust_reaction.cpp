#include "../chemistry.h"
#include "reaction_list.h"
#include "dust_reaction.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Enum of cooling kinds

__host__ bool dust_reac_t::parse_gr
( const list_t &  rls, const std::string         & label,
  const input  & rarg, const ::chemistry::base_t & chem )
{
    if( ! super_t::parse( label, rarg, chem ) )
        return false;

    // Per-species cross-section for (d) grain species.
    auto & chem_ext
         = dynamic_cast< const chem_t & >( chem );
    this->pars[ 0 ] *= chem_ext.get_sigma_dust_cgs
        ( chem.species.at( this->i_chem[ 0 ] ).name );
    kind = surface;
    return    true;
}

__host__ bool dust_reac_t::parse_thermal
( const list_t &  rls, const ::chemistry::species_t & sp,
  const ::chemistry::base_t & chem )
{
    auto & chem_ext
        = dynamic_cast< const chem_t & >( chem );
    if( ! chem_ext.is_grain_species( sp.name ) )
        return false;
    kind   = thermal;    
    std::vector< int > i_chem;
    i_chem.push_back( sp.idx );
    i_chem.push_back( sp.idx ); // Same product as reac.
    
    std::vector< f_t > pars, enes;
    // Compute cross-section per particle.
    // For "Gr": sigma = pi * r^2 (whole-grain
    // cross-section).
    // For C(d): sigma = 3 * m_atom / (4 * r * rho)
    // (per-atom cross-section).
    pars.push_back
        ( chem_ext.get_sigma_dust_cgs( sp.name ) );
    pars.push_back( 0.5 );      // \propto sqrt( T )
    // Energy array enes left blank intentionally
    std::vector< f_t > tmp;     // Also blank intentionally
    super_t::setup_pars( chem, i_chem, 1, pars, enes, tmp );
    return true;
}

__host__ bool dust_reac_t::parse_sublimation
( const list_t &  rls, const std::string         & label,
  const input  & sarg, const ::chemistry::base_t & chem )
{
    const auto rate_type = sarg.get< std::string >
        ( label, "rate_type", "" );
    if ( rate_type != "sublimation" )
        return false;

    if ( ! super_t::parse( label, sarg, chem ) )
        return false;

    auto & chem_ext
         = dynamic_cast< const chem_t & >( chem );
    const auto & sp_name = chem.species.at
        ( this->i_chem[ 0 ] ).name;
    const auto dp = chem_ext.get_dust_params( sp_name );
    this->pars[ 0 ] *= 3.0
        / ( 4.0 * dp.r_cgs * dp.rho_cgs );

    kind = sublimation;
    return true;
}

};                      // namespace extension::chemistry
