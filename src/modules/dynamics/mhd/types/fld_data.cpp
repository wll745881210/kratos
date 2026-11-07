#include "fld_data.h"

namespace dynamics::mhd
{
////////////////////////////////////////////////////////////
// Data chunk for MHD

__host__ int & fld_t::n_stp(  ) const
{
    return const_cast< fld_t * >( this )->bf.n_step;
}

__host__ std::string fld_t::prefix(  ) const
{
    return "field_";
}

__host__ void fld_t::setup
( const f_new_t &  f_new, const geo_t & geo )
{
    auto n_ceff = geo  .  n_ceff;
    for( auto   & n_s  :  n_ceff.x )
        n_s +=  ( n_s  >  1 );
    idx_t n_gh  ( order(  ) );
    idx_t ones         (  1 );
    bf.init( f_new, n_ceff, n_gh, 3, n_stp(  ) );
    ec.init( f_new, n_ceff, n_gh, 3,         1 );
    ee.init( f_new, n_ceff, ones, 3,         1 );
    if( need_ni )
    {
        e_dif  .init( f_new, n_ceff, ones, 3, 1 );
        j_edge .init( f_new, n_ceff, ones, 3, 1 );
        e_diag .init( f_new, n_ceff, ones, 3, 1 );
        nod_w  .init( f_new, n_ceff, ones, 1, 1 );
        nod_p  .init( f_new, n_ceff, ones, 1, 1 );
        heat   .init( f_new, geo.n_ceff, n_gh, 2, 1 );
        bf_prev.init( f_new, n_ceff, n_gh, 3, 1 );
    }
    if( need_ni_ee )
    {
        a_edge .init( f_new, n_ceff, ones, 3, 1 );
        eta_cc .init( f_new, n_ceff, ones, 6, 1 );
    }
    if( need_ni_imp )
        cg_x .init( f_new, n_ceff, n_gh, 3, 1 );
    if( need_ni_proj )
    {
        //  No ghost ring:  the projection's kernels index the
        //  potential by the real cells only ( see mg_cc_lam ).
        idx_t none( idx_t::null(  ) );
        phi  .init( f_new, geo.n_ceff, none, 1, 1 );
        prhs .init( f_new, geo.n_ceff, none, 1, 1 );
    }
    return;
}

__host__ void fld_t::free( const f_free_t & f_free )
{
    bf.free( f_free );
    ec.free( f_free );
    ee.free( f_free );
    if( need_ni )
    {
        e_dif  .free( f_free );
        j_edge .free( f_free );
        e_diag .free( f_free );
        nod_w  .free( f_free );
        nod_p  .free( f_free );
        heat   .free( f_free );
        bf_prev.free( f_free );
    }
    if( need_ni_ee )
    {
        a_edge .free( f_free );
        eta_cc .free( f_free );
    }
    if( need_ni_imp )
        cg_x .free( f_free );
    if( need_ni_proj )
    {
        phi  .free( f_free );
        prhs .free( f_free );
    }
    return;
}

__host__ void fld_t::read ( const  f_read_t & f_r )
{
    return bf.read ( f_r, prefix(  ) + "bf_" );    
}

__host__ void fld_t::write( const f_write_t & f_w )
{
    return bf.write( f_w, prefix(  ) + "bf_" );
};

__host__ void fld_t::copy_to
( const f_cp_t & f_cp, fld_t & tgt ) const
{
    return bf.cp_to( tgt.bf, f_cp );
}

};                              // namespace dynamics::mhd

