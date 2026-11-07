#include "edge_mg_grad.h"

#include <vector>

////////////////////////////////////////////////////////////
//  The host half of the subcell eta gradient.
//
//  `eta_grad_lat` turns the per-cell tensor lattice into a
//  per-cell gradient contribution on the device ( it is a
//  host driver around `eta_grad_k` ),  and `_d1_line` is the
//  one-line reference the gates compare against.  The
//  kernels and the `__device__` helpers stay in the header.

namespace mhd_ee::ni::edge_mg
{

__host__ void eta_grad_lat
( const geo_sph_t & g, const cplx_t & c,
  const std::vector< float_t > & lat, std::vector< float_t >
      & gr )
{
    const int nr  = int( c.n[ 0 ] );
    const int nt  = int( c.n[ 1 ] );
    const int nph = int( c.n[ 2 ] );
    gr.assign( std::size_t( c.n_cell ) * 18, 0. );
    std::vector< float_t > cr, ct, v;
    cr.resize( std::size_t( nr ) );
    ct.resize( std::size_t( nt ) );
    v .resize( std::size_t( ( nt > nr ) ? nt : nr ) );
    for( int i = 0; i < nr; ++ i )
    {
        cr[ i ] = float_t( eta_r_centre( g, i ) );
    }
    for( int j = 0; j < nt; ++ j )
    {
        ct[ j ] = float_t( eta_theta_centre( g, j ) );
    }
    //  ---- d/dr and d/dtheta: gather each line, then the
    //  stencil
    for( int k2 = 0; k2 < nph; ++ k2 )
    {
        for( int j = 0; j < nt; ++ j )
        {
            for( int i = 0; i < nr; ++ i )
            {
                const std::size_t ic = ( std::size_t( k2 ) * nt + j )
                                     * nr + i;
                for( int q = 0; q < 6; ++ q )
                {
                    for( int u = 0; u < nr; ++ u )
                    {
                        v[ u ] = lat[ ( ( std::size_t( k2 ) * nt + j )
                                      * nr + u ) * 6 + q ];
                    }
                    gr[ ic * 18 + q ] = _d1_line( cr, v, i );
                    for( int u = 0; u < nt; ++ u )
                    {
                        v[ u ] = lat[ ( ( std::size_t( k2 ) * nt + u )
                                      * nr + i ) * 6 + q ];
                    }
                    gr[ ic * 18 + 6 + q ] = _d1_line( ct, v, j );
                }
            }
        }
    }
    //  ---- d/dphi: uniform and PERIODIC, so centred and no
    //  wall
    for( int j = 0; j < nt; ++ j )
    {
        for( int i = 0; i < nr; ++ i )
        {
            for( int k2 = 0; k2 < nph; ++ k2 )
            {
                const std::size_t ic = ( std::size_t( k2 ) * nt + j )
                                     * nr + i;
                const int km = ( k2 + nph - 1 ) % nph;
                const int kp = ( k2 + 1 ) % nph;
                for( int q = 0; q < 6; ++ q )
                {
                    const std::size_t jm = ( std::size_t( km ) * nt + j )
                                         * nr + i;
                    const std::size_t jp = ( std::size_t( kp ) * nt + j )
                                         * nr + i;
                    gr[ ic * 18 + 12 + q ]
                        = ( lat[ jp * 6 + q ] - lat[ jm * 6 + q ] )
                        / ( 2. * float_t( g.dph ) );
                }
            }
        }
    }
    //  ---- rotate to CARTESIAN at the cell centre
    for( int k2 = 0; k2 < nph; ++ k2 )
    {
        for( int j = 0; j < nt; ++ j )
        {
            const float_t th = float_t( eta_theta_centre( g,
                j ) );
            const float_t ph = float_t( eta_ph_centre( g, k2
                ) );
            const float_t st = std::sin( th ), ctt =
                std::cos( th );
            const float_t sp = std::sin( ph ), cp =
                std::cos( ph );
            const float_t er[ 3 ] = { st * cp, st * sp, ctt
                };
            const float_t et[ 3 ] = { ctt * cp, ctt * sp,
                -st };
            const float_t ep[ 3 ] = { -sp, cp, 0. };
            for( int i = 0; i < nr; ++ i )
            {
                const float_t r  = float_t( eta_r_centre( g,
                    i ) );
                const float_t cs[ 3 ] = { 1., r, r * st };
                const std::size_t ic = ( std::size_t( k2 ) * nt + j )
                                     * nr + i;
                float_t g0[ 18 ];
                for( int t = 0; t < 18; ++ t )
                {
                    g0[ t ] = gr[ ic * 18 + t ];
                }
                for( int a = 0; a < 3; ++ a )
                {
                    for( int q = 0; q < 6; ++ q )
                    {
                        gr[ ic * 18 + a * 6 + q ]
                            = er[ a ] * g0[      q ] / cs[ 0 ]
                            + et[ a ] * g0[  6 + q ] / cs[ 1 ]
                            + ep[ a ] * g0[ 12 + q ] / cs[ 2 ];
                    }
                }
            }
        }
    }
    return;
};
__host__ float_t _d1_line
( const std::vector< float_t > & cc, const std::vector<
    float_t > & v,
  const int & i )
{
    const int n = int( cc.size(  ) );
    float_t c[ 3 ];
    int   id[ 3 ];
    eta_d1_stencil( i, n,
                    int( i ) > 1 ? cc[ i - 2 ] : cc[ i ],
                    int( i ) > 0 ? cc[ i - 1 ] : cc[ i ], cc[ i ],
                    int( i ) + 1 < n ? cc[ i + 1 ] : cc[ i ],
                    int( i ) + 2 < n ? cc[ i + 2 ] : cc[ i ], id, c );
    return c[ 0 ] * v[ id[ 0 ] ] + c[ 1 ] * v[ id[ 1 ] ]
         + c[ 2 ] * v[ id[ 2 ] ];
};

};    // namespace mhd_ee::ni::edge_mg
