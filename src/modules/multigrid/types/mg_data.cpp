#include "../../../utilities/mapping/loop.h"
#include "dual_data.h"
#include "mg_data.h"

namespace multigrid::data
{
////////////////////////////////////////////////////////////
// Defult multi-grid data type

__host__ void base_t::setup
( const f_new_t & f_new, const geo_t & geo )
{
    idx_t n_gh( idx_t::null(  ) );
    const auto order = this->order(  );
    for( int a = 0; a < n_gh.size(  ); ++ a )
        n_gh[ a ] = ( geo.n_ceff[ a ] > 1 ? order : 0 );
    src_dat   .init( f_new, geo.n_ceff, n_gh, n_fld(  ) );
    return fld.init( f_new, geo.n_ceff, n_gh, n_fld(  ) );
}

__host__ void base_t::free( const f_free_t & f_free )
{
    src_dat   .free( f_free );
    return fld.free( f_free );
}

__host__ void base_t:: read
( const  f_read_t & f_r, const std::string & prefix )
{
    src_dat   .read( f_r, prefix + "src_" );
    return fld.read( f_r, prefix );
}

__host__ void base_t::write
( const f_write_t & f_w, const std::string & prefix )
{
    src_dat.write( f_w, prefix + "src_" );    
    return fld.write( f_w, prefix );
}

__host__ void base_t::null( const f_null_t & f_null )
{
    // f_null( src_dat.dat, src_dat .n_size );
    return  f_null( fld. dat, fld.n_size );
}

__host__ void base_t::copy_to
( const f_cp_t & f_cp, base_t & tgt ) const
{
    src_dat   .cp_to( tgt.src_dat, f_cp );
    return fld.cp_to( tgt.    fld, f_cp );
}
};                           // namespace multigrid::data
