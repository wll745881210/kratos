#include "edge_mg_global.h"

#include <vector>

////////////////////////////////////////////////////////////
//  The host assemblers.
//
//  `assemble_m1k_a` builds the mass,  the stiffness and the
//  summed operator of one level on the host ( the reference
//  side;  production uses the device path in the kernels/
//  subdirectory ),  and `reduce_rhs` maps a raw edge vector
//  onto the compact space.  Both are pure host loops over the
//  complex;  nothing here is device code.

namespace mhd_ee::ni::edge_mg
{

__host__ void reduce_rhs
( const int & n_edge, const std::vector< int > & keep,
  const std::vector< int > & comp, const drop_t & mode,
  const std::vector< float_t > & b_raw, std::vector< float_t
      > & b )
{
    if( mode == drop_t::compact )
    {
        int nc( 0 );
        for( int e = 0; e < n_edge; ++ e )
        {
            if( comp[ e ] + 1 > nc )
            {
                nc = comp[ e ] + 1;
            }
        }
        b.assign( std::size_t( nc ), 0. );
        for( int e = 0; e < n_edge; ++ e )
        {
            if( comp[ e ] >= 0 )
            {
                b[ comp[ e ] ] = b_raw[ e ];
            }
        }
        return;
    }
    b = b_raw;
    for( int e = 0; e < n_edge; ++ e )
    {
        if( keep[ e ] == 0 )
        {
            b[ e ] = 0.;
        }
    }
    return;
};
__host__ void assemble_m1k_a
( const cplx_t & c, const cell_blk_t * p_blk,
  const std::vector< int > & keep, const std::vector< int > & comp,
  const int & n_compact, const drop_t & mode, const float_t
      & dt,
  csr_t & M1, csr_t & K, csr_t & A )
{
    const int ne = c.n_edge;
    const bool cm = ( mode == drop_t::compact );
    const bool raw = ( mode == drop_t::raw );
    std::vector< int >    re, ce;
    std::vector< float_t > v1, v2, va;
    const std::size_t cap = std::size_t( c.n_cell ) * 24 + std::size_t( ne );
    re.reserve( cap );
    ce.reserve( cap );
    v1.reserve( cap ); v2.reserve( cap ); va.reserve( cap );
    int ge[ 12 ];
    for( int ic = 0; ic < c.n_cell; ++ ic )
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) ) % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        c.cell_edges( i, ge );
        for( int e = 0; e < 12; ++ e )
        {
            const int ge_ = ge[ e ];
            if( ! raw && keep[ ge_ ] == 0 )
            {
                continue;
            }
            const int row = cm ? comp[ ge_ ] : ge_;
            for( int e2 = 0; e2 < 12; ++ e2 )
            {
                const int ge2 = ge[ e2 ];
                if( ! raw && keep[ ge2 ] == 0 )
                {
                    continue;
                }
                const float_t a1 = float_t( p_blk[ ic ].m1[
                    e * 12 + e2 ] );
                const float_t a2 = float_t( p_blk[ ic ]. k[
                    e * 12 + e2 ] );
                if( a1 == 0. && a2 == 0. )
                {
                    continue;
                }
                re.push_back( row );
                ce.push_back( cm ? comp[ ge2 ] : ge2 );
                v1.push_back( a1 );
                v2.push_back( a2 );
                va.push_back( a1 + dt * a2 );
            }
        }
    }
    //  the identity policy: the dropped rows are entries of
    //  the matrix
    //  ( and the caller must zero their right-hand side,
    //  see reduce_rhs )
    if( mode == drop_t::identity )
    {
        for( int e = 0; e < ne; ++ e )
        {
            if( keep[ e ] != 0 )
            {
                continue;
            }
            re.push_back( e );
            ce.push_back( e );
            v1.push_back( 1. );
            v2.push_back( 1. );
            va.push_back( 1. + dt );
        }
    }
    const int n = cm ? n_compact : ne;      // raw and identity both:  ne
    M1.build( n, re, ce, v1 );
    K .build( n, re, ce, v2 );
    A .build( n, re, ce, va );
    return;
};

};    // namespace mhd_ee::ni::edge_mg
