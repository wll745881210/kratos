#pragma once

#include "../../src/mesh/meshgen/spherical.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  Spherical meshgen with a midplane - compressed theta
//  grid ( the WBG19 recipe ):  geometric cell - width
//  progression away from the midplane,  w_pole / w_mid
//  = [mesh] th_ratio  ( default 1 = uniform ).   The theta
//  domain must be symmetric about pi / 2.

class sph_disk_t : public mesh::meshgen::sph_t
{
public:                         // Data
    type::float2_t th_ratio;
    type::float2_t r_log;    //  [mesh] r_log:  log - radial

    ////////// Functions //////////
    __host__ sph_disk_t(  ) {  th_ratio = 1;  r_log = 0;  };

    //  The optional log - radial map  (  the Kuiper
    // benchmark  spans  1 - 1000 AU  with  the  rim
    // resolved;  the linear default wastes the grid ).
    //  Off by default --  the ppd main body's setups
    // are untouched.
    virtual void read( const input & args ) override
    {
        mesh::meshgen::sph_t::read( args );
        r_log = args.get< type::float2_t >
            ( "mesh", "r_log", 0 );
        th_ratio = args.get< type::float2_t >
                       ( "mesh", "th_ratio", 1 );
        if( fabs( th_ratio - 1 ) < 1e-12 )
            return;
        // The compressed map assumes a midplane -
        // symmetric theta domain.
        const auto t0 = x_lim_global[ 0 ][ 1 ];
        const auto t1 = x_lim_global[ 1 ][ 1 ];
        const auto pi = 3.1415926535897932385;
        if( fabs( t0 + t1 - pi ) > 1e-9 )
            throw std::runtime_error
                ( "sph_disk_t: th domain not symmetric" );
        if( n_ceff_global[ 1 ] % 2 )
            throw std::runtime_error
                ( "sph_disk_t: n_cell theta must be even" );
        return;
    };

    virtual void set_uniformity ( mesh::block_t & b ) override
    {
        mesh::meshgen::sph_t::set_uniformity( b );
        if( fabs( th_ratio - 1 ) > 1e-12 )
            b.geo.is_uniform[ 1 ] = false;
        return;
    };

    // The base's theta - center formula is not the cell
    // center ( returns ~ cos^2 - weighted values,  never
    // exercised before because theta was always uniform ).
    // Rewrite xv[ 1 ] with the sin - weighted centers.
    virtual void set_coord_axes( mesh::block_t & b ) override
    {
        mesh::meshgen::sph_t::set_coord_axes( b );
        if( fabs( th_ratio - 1 ) < 1e-12 )
            return;
        auto & geo( b.geo );
        auto * xf = geo.xf[ 1 ] + geo.n_gh;
        auto * xv = geo.xv[ 1 ] + geo.n_gh;
        for( int i = -geo.n_gh;
             i <= geo.n_ceff[ 1 ] + geo.n_gh + 1; ++ i )
        {
            const auto tl = xf[ i     ];
            const auto tr = xf[ i + 1 ];
            xv[ i ] = ( ( sin( tr ) - tr * cos( tr ) )
                      - ( sin( tl ) - tl * cos( tl ) ) )
                    / ( cos( tl ) - cos( tr ) );
        }
        return;
    };

    //  The optional log - radial map:   r( xi ) =
    // x_min ( x_max / x_min )^xi  (  [mesh] r_log = 1;
    // the Kuiper benchmark  spans  1 - 1000 AU  with  the
    // rim resolved ).
    virtual type::float2_t location
    ( const type::float2_t & x_logic,
      const int & ax ) const override
    {
        if( ax == 0 && r_log > 0 )
            return x_lim_global[ 0 ][ 0 ] * pow
                ( x_lim_global[ 1 ][ 0 ]
                  / x_lim_global[ 0 ][ 0 ], x_logic );
        if( ax != 1 || fabs( th_ratio - 1 ) < 1e-12 )
            return mesh::meshgen::sph_t::location
                ( x_logic, ax );
        const auto pi  = 3.1415926535897932385;
        const auto t0  = x_lim_global[ 0 ][ 1 ];
        const auto m   = n_ceff_global[ 1 ] / 2;
        const auto q   = pow( th_ratio, -1. / ( m - 1 ) );
        const auto wp  = ( pi / 2 - t0 ) * ( q - 1 )
                       / ( pow( q, m ) - 1 );
        // Lower half:  geometric widths shrinking toward
        // the midplane;  upper half mirrored.
        const auto xi  = x_logic < 0.5 ? x_logic
                                       : 1 - x_logic;
        const auto th  = t0 + wp * ( pow( q, 2 * m * xi )
                                     - 1 ) / ( q - 1 );
        return x_logic < 0.5 ? th : pi - th;
    };
};

};                              // namespace interm
