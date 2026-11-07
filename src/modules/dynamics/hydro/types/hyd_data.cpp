#include "block_data.h"

namespace dynamics::hydro
{
////////////////////////////////////////////////////////////
// Data chunk for hydrodynamics

__host__ int & hyd_t::n_stp(  ) const
{
    return const_cast< hyd_t * >( this )->u.n_step;
}

__host__ std::string hyd_t::prefix(  ) const
{
    return "hydro_";
}

__host__ void hyd_t::setup
( const f_new_t & f_new, const geo_t & geo )
{
    idx_t n_gh( idx_t::null(  ) );
    const auto order = this->order(  );
    for( int a = 0; a < n_gh.size (  ); ++ a )
        n_gh[ a ] = ( geo.n_ceff[ a ] > 1 ? order : 0 );

    type::idx_t zero( { 0, 0, 0 } );
    type::idx_t ones( { 1, 1, 1 } );
    w.init( f_new, geo.n_ceff, n_gh, n_fld(  ),         1 );
    u.init( f_new, geo.n_ceff, zero, n_fld(  ), n_stp(  ) );

    for( int n = 0; n < geo.n_dim; ++ n )
    {
        auto n_ceff   = geo.n_ceff;
        n_ceff[ n ]  += 1;
        f[ n ].init( f_new, n_ceff, ones, n_fld(  ), 1 );
    }
    return;
}

__host__ void hyd_t::free( const f_free_t & f_free )
{
    u.free      ( f_free );
    w.free      ( f_free );
    for( auto & f_s :  f )
        f_s.free( f_free );
    return;
}

__host__ void hyd_t::read ( const f_read_t  & f_r )
{
    return u.read ( f_r, prefix(  ) + "cons_" );
}

__host__ void hyd_t::write( const f_write_t & f_w )
{
    return u.write( f_w, prefix(  ) + "cons_" );
};

__host__ void hyd_t::copy_to
( const f_cp_t & f_cp, hyd_t & tgt ) const
{
    return u.cp_to( tgt.u, f_cp );
}

};                              // namespace dynamics::hydro
