#pragma once

#include "edge_mg_cplx.h"
#include "edge_mg_data.h"
#include "edge_mg_geo.h"

////////////////////////////////////////////////////////////
//  Per-cell local-block assembly,  weak form,  one
//  implementation shared by the host gates and the device
//  kernels.
//
//  The local algebra is IDENTICAL to the certified Python
//  reference ( proto/p8_sph2d.build_geo,  which is itself
//  p8_assemble.build_local with an analytic Jacobian ):
//
//      phi_e   = J^{-T} h_e             ( covariant Piola )
//      curl_e  = J h_c / det J
//      phi_f   = J e_a h_f / det J      ( contravariant )
//      M1  = int phi_e . phi_e'   dV
//      M2  = int phi_f . phi_f'   dV
//      M12 = int phi_e . phi_f    dV
//      K   = int ( eta curl_e ) . curl_e'  dV
//
//  with 3-point Gauss per direction ( exact for the
//  polynomial degrees involved when the metric is affine )
//  and the local order
//      ED = [ (a, sb, sl) for a in 0..2, sb, sl in 0..1 ]
//      FD = [ (a, s)      for a in 0..2,  s in 0..1 ]
//  so that a dump can be compared element by element with
//  the Python side.
//
//  Accumulation is in float2_t ( docs/mhd/nonideal/
//  edge-mg-foundations.md 4.1 );  the stored block is
//  float_t.

namespace mhd_ee::ni::edge_mg
{
using type::float_t;


////////////////////////////////////////////////////////////
//  3-point Gauss-Legendre on [0,1]  ( leggauss(3) mapped,
//  i.e. exactly what p8_assemble._gauss does ).
//
//  Written as functions,  not as a namespace-scope const
//  array:  a host `static const` array is not readable in
//  device code ( nvcc:  "undefined in device code" ),  and
//  __constant__ storage would need a host copy.  Two
//  functions are cheaper than either.

__host__ __device__ inline float2_t gq_x( const int & q )
{
    const float2_t a( 0.7745966692414834 );
    const float2_t h( 0.5 );
    return ( q == 0 ) ? h * ( float2_t( 1 ) - a )
         : ( q == 1 ) ? h
                      : h * ( float2_t( 1 ) + a );
};

__host__ __device__ inline float2_t gq_w( const int & q )
{
    return ( q == 1 ) ? float2_t( 8 ) / float2_t( 18 )
                      : float2_t( 5 ) / float2_t( 18 );
};

////////////////////////////////////////////////////////////
//  3x3 inverse ( closed form,  no branches beyond det )

__host__ __device__ inline float2_t _det3
( const float2_t ( & A )[ 3 ][ 3 ] )
{
    return A[ 0 ][ 0 ] * ( A[ 1 ][ 1 ] * A[ 2 ][ 2 ]
                         - A[ 1 ][ 2 ] * A[ 2 ][ 1 ] )
         - A[ 0 ][ 1 ] * ( A[ 1 ][ 0 ] * A[ 2 ][ 2 ]
                         - A[ 1 ][ 2 ] * A[ 2 ][ 0 ] )
         + A[ 0 ][ 2 ] * ( A[ 1 ][ 0 ] * A[ 2 ][ 1 ]
                         - A[ 1 ][ 1 ] * A[ 2 ][ 0 ] );
};

__host__ __device__ inline void _inv3
( const float2_t ( & A )[ 3 ][ 3 ], const float2_t & d,
  float2_t ( & B )[ 3 ][ 3 ] )
{
    const float2_t id = float2_t( 1 ) / d;
    B[ 0 ][ 0 ] =  ( A[ 1 ][ 1 ] * A[ 2 ][ 2 ]
                   - A[ 1 ][ 2 ] * A[ 2 ][ 1 ] ) * id;
    B[ 0 ][ 1 ] = -( A[ 0 ][ 1 ] * A[ 2 ][ 2 ]
                   - A[ 0 ][ 2 ] * A[ 2 ][ 1 ] ) * id;
    B[ 0 ][ 2 ] =  ( A[ 0 ][ 1 ] * A[ 1 ][ 2 ]
                   - A[ 0 ][ 2 ] * A[ 1 ][ 1 ] ) * id;
    B[ 1 ][ 0 ] = -( A[ 1 ][ 0 ] * A[ 2 ][ 2 ]
                   - A[ 1 ][ 2 ] * A[ 2 ][ 0 ] ) * id;
    B[ 1 ][ 1 ] =  ( A[ 0 ][ 0 ] * A[ 2 ][ 2 ]
                   - A[ 0 ][ 2 ] * A[ 2 ][ 0 ] ) * id;
    B[ 1 ][ 2 ] = -( A[ 0 ][ 0 ] * A[ 1 ][ 2 ]
                   - A[ 0 ][ 2 ] * A[ 1 ][ 0 ] ) * id;
    B[ 2 ][ 0 ] =  ( A[ 1 ][ 0 ] * A[ 2 ][ 1 ]
                   - A[ 1 ][ 1 ] * A[ 2 ][ 0 ] ) * id;
    B[ 2 ][ 1 ] = -( A[ 0 ][ 0 ] * A[ 2 ][ 1 ]
                   - A[ 0 ][ 1 ] * A[ 2 ][ 0 ] ) * id;
    B[ 2 ][ 2 ] =  ( A[ 0 ][ 0 ] * A[ 1 ][ 1 ]
                   - A[ 0 ][ 1 ] * A[ 1 ][ 0 ] ) * id;
    return;
};

////////////////////////////////////////////////////////////
//  The basis at ONE quadrature point -- shared by the
//  volume
//  assembly,  the surface load and the volume load.
//
//  ONE implementation on purpose. All three integrals must
//  use the
//  same quadrature, the same analytic Jacobian and the same
//  basis
//  functions; a second copy is exactly where the M43 sign
//  flip and
//  the M12 x12 lived. Extracted from assemble_cell in
//  2026-10-30
//  (when the volume load landed), with every existing gate
//  required
//  to be BIT-IDENTICAL afterwards.
//
//      x     the physical point
//      J     dx/dxi  ( already span-scaled by the policy )
//      det   |det J|
//  JiT J^{-1} ( phE uses it in the J^{-T} sense below )
//  phF 2-form basis psi_f = J e_a h_f / det ( 6 )
//  phE 1-form basis phi_e = J^{-T} h_e ( 12 )
//  cuE the curl curl phi_e = J h_c / det ( 12 )

template< class geo_T >
__host__ __device__ inline void basis_at
( const geo_T & g, const idx_t & i, const float2_t ( & xi )[ 3 ],
  float2_t ( & x )[ 3 ], float2_t ( & J )[ 3 ][ 3 ], float2_t & det,
  float2_t ( & JiT )[ 3 ][ 3 ],
  float2_t ( & phF )[ 6 ][ 3 ], float2_t ( & phE )[ 12 ][ 3 ],
  float2_t ( & cuE )[ 12 ][ 3 ] )
{
    g.map( i, xi, x, J, det );
    _inv3( J, det, JiT );
    //  --- face ( 2-form ) basis
    {
        int t( 0 );
        for( int a = 0; a < 3; ++ a )
        {
            for( int s = 0; s < 2; ++ s )
            {
                const float2_t val
                    = s ? xi[ a ]
                        : ( float2_t( 1 ) - xi[ a ] );
                for( int q = 0; q < 3; ++ q )
                {
                    phF[ t ][ q ] = J[ q ][ a ] * ( val / det );
                }
                ++ t;
            }
        }
    }
    //  --- edge ( 1-form ) basis and its curl
    {
        int t( 0 );
        for( int a = 0; a < 3; ++ a )
        {
            const int b = ( a + 1 ) % 3;
            const int l = ( a + 2 ) % 3;
            for( int sb = 0; sb < 2; ++ sb )
            {
                for( int sl = 0; sl < 2; ++ sl )
                {
                    const float2_t fb
                        = sb ? xi[ b ] : ( float2_t( 1 ) - xi[ b ] );
                    const float2_t fl
                        = sl ? xi[ l ] : ( float2_t( 1 ) - xi[ l ] );
                    const float2_t dwb
                        = sb ? float2_t( 1 ) : float2_t( -1 );
                    const float2_t dwl
                        = sl ? float2_t( 1 ) : float2_t( -1 );
                    float2_t he[ 3 ] = { 0, 0, 0 };
                    he[ a ] = fb * fl;
                    for( int q = 0; q < 3; ++ q )
                    {
                        float2_t s( 0 );
                        for( int p = 0; p < 3; ++ p )
                        {
                            s += JiT[ p ][ q ] * he[ p ];
                        }
                        phE[ t ][ q ] = s;
                    }
                    float2_t hc[ 3 ] = { 0, 0, 0 };
                    hc[ b ] =  dwl * fb;
                    hc[ l ] = -dwb * fl;
                    for( int q = 0; q < 3; ++ q )
                    {
                        float2_t s( 0 );
                        for( int p = 0; p < 3; ++ p )
                        {
                            s += J[ q ][ p ] * hc[ p ];
                        }
                        cuE[ t ][ q ] = s / det;
                    }
                    ++ t;
                }
            }
        }
    }
    return;
};

//////////////////////////////////////////////////////////
//  One cell

//  eta may also be given as a FIRST-ORDER TAYLOR expansion
//  about the cell
//  centre: eta(x) = eta_c + (d eta / d x_a) (x - x_c)_a.
//  That is what makes a
//  spatially varying eta second order rather than first (
//  see
//  docs/mhd/nonideal/HANDOVER-ETA-SUBCELL.md ): with ONE
//  constant tensor per
//  cell the K error is O(h) because curl phi_e flips sign
//  across the six face
//  patches of a cell, so the linear part of (eta - eta_h)
//  does NOT integrate
//  away. The gradient is NOT stored -- the caller derives
//  it from a neighbour
//  stencil of the 6-component lattice, so the storage stays
//  6 numbers a cell.
//
//  p_deta is 3 x 6: [a][q] = d eta_q / d x_a, with q in the
//  module's Voigt
//  order (xx, yy, zz, xy, xz, yz) and a the CARTESIAN
//  index. p_deta == nullptr
//  reproduces exactly the constant-tensor path ( and every
//  existing gate is
//  required to stay BIT-IDENTICAL ).

template< class geo_T >
__host__ __device__ void assemble_cell
( const geo_T & g, const cplx_t & c, const idx_t & i,
  const float2_t ( & eta )[ 6 ], cell_blk_t & blk,
  const float2_t * p_deta = nullptr, const float2_t * p_xc = nullptr )
{
    float2_t m1 [ 12 ][ 12 ];
    float2_t kk [ 12 ][ 12 ];
    float2_t m2 [  6 ][  6 ];
    float2_t m12[ 12 ][  6 ];
    for( int e = 0; e < 12; ++ e )
    {
        for( int e2 = 0; e2 < 12; ++ e2 )
        {
            m1 [ e ][ e2 ] = float2_t( 0 );
            kk [ e ][ e2 ] = float2_t( 0 );
        }
        for( int f = 0; f < 6; ++ f )
        {
            m12[ e ][ f ] = float2_t( 0 );
        }
    }
    for( int f = 0; f < 6; ++ f )
    {
        for( int f2 = 0; f2 < 6; ++ f2 )
        {
            m2[ f ][ f2 ] = float2_t( 0 );
        }
    }
    float2_t det_c( 0 );
    for( int qa = 0; qa < 3; ++ qa )
    {
        for( int qb = 0; qb < 3; ++ qb )
        {
            for( int qc = 0; qc < 3; ++ qc )
            {
                float2_t xi[ 3 ] = { gq_x( qa ), gq_x( qb ),
                                     gq_x( qc ) };
                float2_t x[ 3 ], J[ 3 ][ 3 ], det, JiT[ 3 ][ 3 ];
                float2_t phF[ 6 ][ 3 ], phE[ 12 ][ 3 ], cuE[ 12 ][ 3 ];
                basis_at< geo_T >( g, i, xi, x, J, det, JiT,
                                   phF, phE, cuE );
                const float2_t wgt
                    = gq_w( qa ) * gq_w( qb ) * gq_w( qc ) * det;
                //  --- the tensor AT THIS QUADRATURE POINT
                float2_t eq[ 6 ];
                for( int q = 0; q < 6; ++ q )
                {
                    eq[ q ] = eta[ q ];
                }
                if( p_deta != nullptr )
                {
                    for( int a = 0; a < 3; ++ a )
                    {
                        const float2_t d = x[ a ] - p_xc[ a ];
                        for( int q = 0; q < 6; ++ q )
                        {
                            eq[ q ] += p_deta[ a * 6 + q ] * d;
                        }
                    }
                }
                //  --- eta . curl,  once per local edge
                float2_t ec[ 12 ][ 3 ];
                for( int e = 0; e < 12; ++ e )
                {
                    const float2_t v0 = cuE[ e ][ 0 ];
                    const float2_t v1 = cuE[ e ][ 1 ];
                    const float2_t v2 = cuE[ e ][ 2 ];
                    ec[ e ][ 0 ] = eq[ 0 ] * v0 + eq[ 3 ] * v1
                                 + eq[ 4 ] * v2;
                    ec[ e ][ 1 ] = eq[ 3 ] * v0 + eq[ 1 ] * v1
                                 + eq[ 5 ] * v2;
                    ec[ e ][ 2 ] = eq[ 4 ] * v0 + eq[ 5 ] * v1
                                 + eq[ 2 ] * v2;
                }
                //  --- accumulate
                for( int e = 0; e < 12; ++ e )
                {
                    for( int e2 = 0; e2 < 12; ++ e2 )
                    {
                        float2_t s1( 0 ), sk( 0 );
                        for( int q = 0; q < 3; ++ q )
                        {
                            s1 += phE[ e ][ q ] * phE[ e2 ][ q ];
                            sk += ec [ e ][ q ] * cuE[ e2 ][ q ];
                        }
                        m1[ e ][ e2 ] += wgt * s1;
                        kk[ e ][ e2 ] += wgt * sk;
                    }
                    //  M12 is NOT symmetric: one
                    //  accumulation per
                    //  ( e, f ) pair, so this loop must sit
                    //  OUTSIDE
                    //  the e2 loop ( inside it, every entry
                    //  was added
                    //  12 times -- a x12 that A1/A4 could
                    //  not see,
                    //  because they never touch M12 ).
                    for( int f = 0; f < 6; ++ f )
                    {
                        float2_t sm( 0 );
                        for( int q = 0; q < 3; ++ q )
                        {
                            sm += phE[ e ][ q ] * phF[ f ][ q ];
                        }
                        m12[ e ][ f ] += wgt * sm;
                    }
                }
                for( int f = 0; f < 6; ++ f )
                {
                    for( int f2 = 0; f2 < 6; ++ f2 )
                    {
                        float2_t s( 0 );
                        for( int q = 0; q < 3; ++ q )
                        {
                            s += phF[ f ][ q ] * phF[ f2 ][ q ];
                        }
                        m2[ f ][ f2 ] += wgt * s;
                    }
                }
                det_c += wgt;
            }
        }
    }
    //  --- store
    blk.detj = float_t( det_c );
    for( int e = 0; e < 12; ++ e )
    {
        for( int e2 = 0; e2 < 12; ++ e2 )
        {
            blk. m1[ e * 12 + e2 ] = float_t( m1[ e ][ e2 ] );
            blk.  k[ e * 12 + e2 ] = float_t( kk[ e ][ e2 ] );
        }
        for( int f = 0; f < 6; ++ f )
        {
            blk.m12[ e * 6 + f ] = float_t( m12[ e ][ f ] );
        }
    }
    for( int f = 0; f < 6; ++ f )
    {
        for( int f2 = 0; f2 < 6; ++ f2 )
        {
            blk.m2[ f * 6 + f2 ] = float_t( m2[ f ][ f2 ] );
        }
    }
    return;
};

};  // namespace edge_mg
