#include "../chemistry.h"
#include "mol_cooling.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Molecular cooling

__host__ bool mol_cool_t::parse
( const std::string & label_, const input & rarg,
  const ::chemistry::base_t & chem )
{
    auto label( label_ );
    if( label.find( "rotational" ) == std::string::npos ||
        ! super_t::parse( label, rarg, chem ) )
        return false;

    std::vector < float_t > ln_T, ln_ntilde;
    rarg( label, "ln_T",           ln_T );
    rarg( label, "ln_ntilde", ln_ntilde );
    std::vector < std::vector< float_t > > ln_coord;
    ln_coord = { ln_ntilde, ln_T };
    size_t dimr[ 2 ] = { ln_ntilde.size( ), ln_T.size( ) };

    const auto set_intp = [ & ]
    ( auto & intp, const  auto & key ,
      const auto & coord, const auto & dim )
    {
        std::vector       < float_t >   tmp  ;
        rarg              ( label, key, tmp );
        intp.setup_non_uni( dim, coord, tmp );
        intp.set_nearest  (                 );
        intp.to_const     ( *    chem.p_dev );
    };
    set_intp( f_ro_ln_l0,  "ln_l0",    ln_T,  dimr[ 1 ] );
    set_intp( f_ro_alpha,  "alpha",    ln_coord,   dimr );
    set_intp( f_ro_ln_lte, "ln_l_lte", ln_coord,   dimr );
    set_intp( f_ro_ln_n12, "ln_n_1_2", ln_coord,   dimr );
    utils::string::replace_all
            ( label,       "rotational",  "vibrational" );
    rarg    ( label,       "ln_T",                 ln_T );
    rarg    ( label,       "ln_ntilde",       ln_ntilde );
    size_t dimv[ 2 ] = { ln_ntilde.size( ), ln_T.size( ) };
    ln_coord = { ln_ntilde, ln_T };
    set_intp( f_vi_ln_l0,  "ln_l0",    ln_T,  dimv[ 1 ] );
    set_intp( f_vi_ln_lte, "ln_l_lte", ln_coord,   dimv );
    return true;
}

__host__ mol_cool_t & mol_cool_t::operator =
( mol_cool_t & src )
{
    super_t::operator  = ( src );
    i_cool             =   src.i_cool;
    f_vi_ln_l0 .move_from( src.f_vi_ln_l0  );
    f_vi_ln_lte.move_from( src.f_vi_ln_lte );
    f_ro_ln_l0 .move_from( src.f_ro_ln_l0  );
    f_ro_alpha .move_from( src.f_ro_alpha  );
    f_ro_ln_lte.move_from( src.f_ro_ln_lte );
    f_ro_ln_n12.move_from( src.f_ro_ln_n12 );
    return ( * this );
}

};                      // namespace extension::chemistry
