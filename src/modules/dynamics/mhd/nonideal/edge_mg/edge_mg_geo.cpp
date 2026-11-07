#include "edge_mg_geo.h"

#include <vector>

////////////////////////////////////////////////////////////
//  The host half of the geometry policy switch.
//
//  `pole_flags` is the ONE place that decides whether the
//  theta axis reaches a pole; `geo_*_from` validate an
//  arbitrary `mesh::geo_t` and fill the Cartesian or the
//  spherical policy object.  The policy objects themselves
//  and their `map` / `node_x` stay in the header:  they are
//  `__host__ __device__`,  and the kernels need them.

namespace mhd_ee::ni::edge_mg
{

__host__ bool geo_sph_from
( const geo_t & geo, geo_sph_t & g )
{
    //  the TYPE check first: without it a cartesian block
    //  is happily read
    //  as a spherical one ( the uniform-grid test passes ),
    //  and the "wrong"
    //  answer looks like a device bug rather than a
    //  dispatch bug -- which is
    //  exactly what happened on the first A4.7 run (
    //  2026-10-29 ).
    if( geo.geo_type != geo_t::spherical )
    {
        return false;
    }
    for( int a = 0; a < 2; ++ a )
    {
        if( ! geo.is_uniform[ a ] )
        {
            return false;
        }
    }
    g.r0  = geo.x_fc( 0, 0 );
    g.dr  = geo.dx_f( 0, idx_t( 0 ) );
    g.th0 = geo.x_fc( 1, 0 );
    g.dth = geo.dx_f( 1, idx_t( 0 ) );
    g.ph0 = geo.x_fc( 2, 0 );
    g.dph = geo.dx_f( 2, idx_t( 0 ) );
    //  ---- the poles, decided WITH A TOLERANCE, by the ONE
    //  test above
    //  ( pole_flags -- so the policy and every cplx_t of
    //  this block agree ).
    pole_flags( geo, geo.n_ceff, g.pole_lo, g.pole_hi );
    return true;
};
__host__ bool geo_cart_from
( const geo_t & geo, geo_cart_t & g )
{
    for( int a = 0; a < 3; ++ a )
    {
        if( ! geo.is_uniform[ a ] )
        {
            return false;
        }
        g.x0[ a ] = geo.x_fc( a, 0 );
        g.dx[ a ] = geo.dx_f( a, idx_t( 0 ) );
    }
    return true;
};
__host__ void pole_flags
( const mesh::geo_t & geo, const idx_t & n_ceff, bool & lo, bool & hi )
{
    lo = false;
    hi = false;
    if( geo.geo_type != geo_t::spherical )
    {
        return;
    }
    const float2_t th0 = geo.x_fc( 1, 0 );
    const float2_t dth = geo.dx_f( 1, idx_t( 0 ) );
    lo = sph_is_axis( th0 );
    hi = sph_is_axis( th0 + float2_t( int( n_ceff[ 1 ] ) ) * dth );
    return;
};
__host__ bool sph_is_axis( const float2_t & th )
{
    return std::fabs( std::sin( th ) ) < float_t( 1.e-8 );
};

};    // namespace mhd_ee::ni::edge_mg
