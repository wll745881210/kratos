#pragma once

#ifdef DBG_CT_ID_DBG

#include <cstring>
#include "../../hydro/integrator/integrator.h"

namespace dynamics::mhd::integrator
{
////////////////////////////////////////////////////////////
//  Forward CT - curl telescoping probe  ( DBG_CT_ID_DBG ).
//
//  The ideal CT advances a face by  ( C E )_face
//      [ ( E_a1 l_a1 )( i ) - ( E_a2 l_a2 )( i )
//      + ( E_a2 l_a2 )( i + e_a1 )
//      - ( E_a1 l_a1 )( i + e_a2 ) ] / sfc( i, a )
//  --  the four - corner stencil of base_t::intg_field
//  built from the ideal - side helpers  edg_f / surface /
//  sfc_thin.  div B is preserved iff every cell 's six
//  face updates telescope,
//      sum_a [ ( C E )( i + e_a, a ) - ( C E )( i, a ) ]
//      = 0,
//  resp. iff the same sum of the RAW four - edge
//  circulations vanishes -- the exact dual of the
//  nonideal  ct_op_dbg_cell  criterion  ( implicit_mg.h ).
//
//  E is SYNTHETIC  ( a deterministic function of the edge
//  indices ):  no state,  no boundary condition,  so the
//  probe isolates the geometry tables of the forward
//  stencil.  Two switches,  both reported:
//    flat = 1 : E does not vary along i[ 2 ]  --  a field
//               a 2.5D ( thin - axis ) mesh can represent.
//    flat = 0 : E varies along i[ 2 ],  so it exercises
//               the thin - axis pair drops of intg_field.
//    ee = 1   : read the run 's own fld.ee instead of the
//               synthetic E  ( the honest forward operator
//               of that state ).
//  Filter: reported maxima cover only cells with
//  max_a | circ | >= ct_id_thr  ( 1e-4 );  the unfiltered
//  max and the dead - cell count are printed separately.

constexpr double ct_id_thr = 1.e-4;

//  Synthetic edge EMF.
template< class idx_T >
__device__ __forceinline__ type::float_t ct_id_e
( const idx_T & i, const int & q, const bool & flat )
{
    const auto x0 = type::float_t( i[ 0 ] );
    const auto x1 = type::float_t( i[ 1 ] );
    const auto x2 = flat ? type::float_t( 0 )
                         : type::float_t( i[ 2 ] );
    const auto xq = type::float_t( q );
    return sinf( 0.7f * x0 + 1.3f * x1 + 2.1f * x2
                 + 0.4f * xq )
         + 0.3f * cosf( 1.1f * x0 - 0.9f * x1
                      + 0.5f * x2 + 1.7f * xq );
}

//  One face 's four - edge circulation,  the stencil of
//  intg_field verbatim:  the pair { term1, term4 } reads
//  E_a1 at i and i + e_a2 and is dropped when axis a2 is
//  thin;  { term2, term3 } reads E_a2 at i and i + e_a1
//  and is dropped when axis a1 is thin.
template< class bdt_T, class itg_T, class idx_T >
__device__ __forceinline__ type::float2_t ct_id_curl
( const bdt_T & b_d, const itg_T & itg, const idx_T & i,
  const int & a, const bool & use_ee, const bool & flat )
{
    using f2_t = type::float2_t;
    const auto & geo( b_d.geo );
    const int a1( utils::mod3( a + 1 ) );
    const int a2( utils::mod3( a + 2 ) );
    const bool th_1( a1 >= geo.n_dim );
    const bool th_2( a2 >= geo.n_dim );
    auto i1( i );   ++ i1[ a1 ];
    auto i2( i );   ++ i2[ a2 ];
    auto e = [ & ]( const idx_T & j, const int & q )
    {
        return use_ee
             ? f2_t( b_d.fld.ee.at( j )[ q ] )
             : f2_t( ct_id_e( j, q, flat ) );
    };
    f2_t db( 0 );
    if( ! th_2 )
        db  = e( i,  a1 ) * f2_t( itg.edg_f( a1, i,  geo ) );
    if( ! th_1 )
        db -= e( i,  a2 ) * f2_t( itg.edg_f( a2, i,  geo ) );
    if( ! th_1 )
        db += e( i1, a2 ) * f2_t( itg.edg_f( a2, i1, geo ) );
    if( ! th_2 )
        db -= e( i2, a1 ) * f2_t( itg.edg_f( a1, i2, geo ) );
    return db;
}

//  Face area exactly as intg_field selects it.
template< class bdt_T, class itg_T, class idx_T >
__device__ __forceinline__ type::float2_t ct_id_sfc
( const bdt_T & b_d, const itg_T & itg, const idx_T & i,
  const int & a )
{
    const auto & geo( b_d.geo );
    return a >= geo.n_dim
         ? type::float2_t( itg.sfc_thin( a, i, geo ) )
         : type::float2_t( geo.surface( i, a ) );
}

//  Pack  ( value, linear index )  for an atomic argmax:  a
//  non - negative float orders like its bit pattern.
__device__ __forceinline__ unsigned long long ct_id_key
( const type::float2_t & v, const unsigned int & l )
{
    return ( ( unsigned long long )
             __float_as_uint( float( v ) ) << 32 ) | l;
}

////////////////////////////////////////////////////////////
//  One cell:  its six faces,  the two telescoping sums.
template< class bdt_T, class itg_T, class idx_T >
__device__ __forceinline__ void ct_id_dbg_cell
( const bdt_T & b_d, const itg_T & itg, const idx_T & i,
  const bool & use_ee, const bool & flat, const idx_T & tgt,
  unsigned long long * out )
{
    using f2_t = type::float2_t;
    const auto & geo( b_d.geo );
    f2_t c[ 6 ], ce[ 6 ], cf( 0 ), ff( 0 ), s_c( 0 ), s_f( 0 );
    bool ok( true );
    for( int a = 0; a < 3; ++ a )
    {
        auto k( i );   ++ k[ a ];
        const auto sa( ct_id_sfc( b_d, itg, i, a ) );
        const auto sk( ct_id_sfc( b_d, itg, k, a ) );
        c [ a     ] = ct_id_curl( b_d, itg, i, a, use_ee, flat );
        c [ 3 + a ] = ct_id_curl( b_d, itg, k, a, use_ee, flat );
        //  intg_field zeroes the flux of a non - positive
        //  face area  ( the polar gauge faces ).
        ce[ a     ] = sa > 0 ? c[ a     ] / sa : f2_t( 0 );
        ce[ 3 + a ] = sk > 0 ? c[ 3 + a ] / sk : f2_t( 0 );
        cf = fmax( cf, fmax( fabs( c [ a     ] ),
                             fabs( c [ 3 + a ] ) ) );
        ff = fmax( ff, fmax( fabs( ce[ a     ] ),
                             fabs( ce[ 3 + a ] ) ) );
        ok &= isfinite( double( c[ a ] ) )
           && isfinite( double( c[ 3 + a ] ) );
        s_c += c [ 3 + a ] - c [ a ];
        s_f += ce[ 3 + a ] - ce[ a ];
    }
    const f2_t rel_c( cf > 0 ? fabs( s_c ) / cf : f2_t( 0 ) );
    const f2_t rel_f( ff > 0 ? fabs( s_f ) / ff : f2_t( 0 ) );
    const int n0( geo.n_ceff[ 0 ] );
    const int n1( geo.n_ceff[ 1 ] );
    const unsigned int lin( ( unsigned int )
        ( i[ 0 ] + n0 * ( i[ 1 ] + n1 * i[ 2 ] ) ) );
    if( out != nullptr )
    {
        atomicAdd( out + 10, 1ull );
        if( ! ok )
            atomicAdd( out + 8, 1ull );
        atomicMax( out + 1, ct_id_key( cf, lin ) );
        atomicMax( out + 3, ct_id_key( ff, lin ) );
        atomicMax( out + 5, ct_id_key( rel_c, lin ) );
        if( cf > 0 )
        {
            atomicAdd( out + 9, 1ull );
            atomicMax( out + 4, ct_id_key( rel_c, lin ) );
        }
        else
            atomicAdd( out + 7, 1ull );
        if( cf >= type::float2_t( ct_id_thr ) )
        {
            atomicAdd( out + 6, 1ull );
            atomicMax( out + 0, ct_id_key( rel_c, lin ) );
            atomicMax( out + 2, ct_id_key( rel_f, lin ) );
        }
    }
    const int n2( geo.n_ceff[ 2 ] );
    const bool s0( i[ 0 ] == 0 || i[ 0 ] == n0 / 2
                || i[ 0 ] == n0 - 1 );
    const bool s1( i[ 1 ] == 0 || i[ 1 ] == n1 / 2
                || i[ 1 ] == n1 - 1 );
    const bool samp( s0 && s1 && i[ 2 ] == 0 );
    const bool hit( tgt[ 0 ] >= 0 && i == tgt );
    if( hit || ( out != nullptr && samp ) )
        printf( "DBGCTID ee=%d flat=%d hit=%d ndim=%d"
                " nce=%d,%d,%d i=%d %d %d"
                " rel=%.3e S=%+.3e cfmax=%.3e"
                " relf=%.3e Sf=%+.3e ffmax=%.3e"
                " c=%+.3e %+.3e %+.3e %+.3e %+.3e %+.3e\n",
                int( use_ee ), int( flat ), int( hit ),
                geo.n_dim, n0, n1, n2,
                i[ 0 ], i[ 1 ], i[ 2 ],
                double( rel_c ), double( s_c ), double( cf ),
                double( rel_f ), double( s_f ), double( ff ),
                double( c[ 0 ] ), double( c[ 1 ] ),
                double( c[ 2 ] ), double( c[ 3 ] ),
                double( c[ 4 ] ), double( c[ 5 ] ) );
    return;
}

template< class bdt_T, class itg_T >
__global__ void ct_id_dbg_k
( const bdt_T b_d, const itg_T f_itg, const int use_ee,
  const int flat, const type::idx_t tgt,
  unsigned long long * out )
{
    const auto & geo( b_d.geo );
    const int n0( geo.n_ceff[ 0 ] );
    const int n1( geo.n_ceff[ 1 ] );
    const int t( blockIdx.x * blockDim.x + threadIdx.x );
    if( t >= n0 * n1 * geo.n_ceff[ 2 ] )
        return;
    type::idx_t i;
    i[ 0 ] =   t % n0;
    i[ 1 ] = ( t / n0 ) % n1;
    i[ 2 ] =   t / ( n0 * n1 );
    ct_id_dbg_cell( b_d, f_itg, i, use_ee != 0, flat != 0,
                    tgt, out );
    return;
}

////////////////////////////////////////////////////////////
//  Host side:  four passes  ( flat x ee ),  one printout
//  each,  then a detail line for the worst gated cell.
inline void ct_id_dbg_print
( const unsigned long long * h, const int use_ee,
  const int flat, const int nd, const int * nce )
{
    auto un = [ ]( const unsigned long long & k, int & j )
    {
        j = int( k & 0xffffffffull );
        const unsigned u = unsigned( k >> 32 );
        float v;
        std::memcpy( & v, & u, sizeof( v ) );
        return double( v );
    };
    int j0( 0 ), j1( 0 ), j2( 0 ), j3( 0 ), j4( 0 );
    const double r0( un( h[ 0 ], j0 ) );
    const double c1( un( h[ 1 ], j1 ) );
    const double r2( un( h[ 2 ], j2 ) );
    const double f3( un( h[ 3 ], j3 ) );
    const double r4( un( h[ 4 ], j4 ) );
    auto cell = [ ]( const int & j, const int * n, int * c )
    {
        c[ 0 ] = j % n[ 0 ];
        c[ 1 ] = ( j / n[ 0 ] ) % n[ 1 ];
        c[ 2 ] = j / ( n[ 0 ] * n[ 1 ] );
        return;
    };
    int t0[ 3 ], t1[ 3 ], t2[ 3 ], t3[ 3 ], t4[ 3 ];
    cell( j0, nce, t0 );
    cell( j1, nce, t1 );
    cell( j2, nce, t2 );
    cell( j3, nce, t3 );
    cell( j4, nce, t4 );
    printf( "DBGCTID MAX ee=%d flat=%d ndim=%d"
            " nce=%d,%d,%d ncell=%llu gate=%llu live=%llu"
            " dead=%llu nonfin=%llu\n"
            "DBGCTID   rel=%.3e at=%d,%d,%d"
            "  cfmax=%.3e at=%d,%d,%d"
            "  relf=%.3e at=%d,%d,%d"
            "  ffmax=%.3e at=%d,%d,%d  rel_all=%.3e"
            " at=%d,%d,%d\n",
            use_ee, flat, nd, nce[ 0 ], nce[ 1 ], nce[ 2 ],
            h[ 10 ], h[ 6 ], h[ 9 ], h[ 7 ], h[ 8 ],
            r0, t0[ 0 ], t0[ 1 ], t0[ 2 ],
            c1, t1[ 0 ], t1[ 1 ], t1[ 2 ],
            r2, t2[ 0 ], t2[ 1 ], t2[ 2 ],
            f3, t3[ 0 ], t3[ 1 ], t3[ 2 ],
            r4, t4[ 0 ], t4[ 1 ], t4[ 2 ] );
    return;
}

template< class bdt_T, class itg_T >
inline void ct_id_dbg_run
( const bdt_T & b_d, const itg_T & f_itg,
  device::base_t & dev, const device::stream_t & str,
  const int & step )
{
    if( step != 0 )
        return;
    static unsigned long long * buf = nullptr;
    if( buf == nullptr )
        buf = dev.malloc_device< unsigned long long >( 16 );
    unsigned long long z[ 16 ], h[ 16 ];
    for( int k = 0; k < 16; ++ k )
        z[ k ] = 0;
    const auto & geo( b_d.geo );
    const int nce[ 3 ] = { geo.n_ceff[ 0 ], geo.n_ceff[ 1 ],
                           geo.n_ceff[ 2 ] };
    const int n_c( nce[ 0 ] * nce[ 1 ] * nce[ 2 ] );
    const int n_th( 128 );
    const dim3 n_bl( ( n_c + n_th - 1 ) / n_th );
    auto empty = [ ](  )
    {
        type::idx_t t;
        t = -1;
        return t;
    };
    unsigned long long * nul = nullptr;
    //  Three passes:  synthetic 2.5D - flat,  synthetic 3D,
    //  then the run 's own ee  ( where flat is moot ).
    const int us[ 3 ] = { 0, 0, 1 };
    const int fl[ 3 ] = { 1, 0, 0 };
    for( int p = 0; p < 3; ++ p )
    {
        const int use_ee( us[ p ] ), flat( fl[ p ] );
        dev.cp( buf, z, 16 );
        dev.launch( ct_id_dbg_k< bdt_T, itg_T >, n_bl,
                    dim3( n_th ), 0, str, b_d, f_itg, use_ee,
                    flat, empty(  ), buf );
        dev.sync_stream( str );
        dev.cp( h, buf, 16 );
        ct_id_dbg_print( h, use_ee, flat, geo.n_dim, nce );
        //  Worst gated cell,  full breakdown  ( out = null
        //  so only that one cell prints ).
        const int j = int( h[ 0 ] & 0xffffffffull );
        type::idx_t t;
        t[ 0 ] = j % nce[ 0 ];
        t[ 1 ] = ( j / nce[ 0 ] ) % nce[ 1 ];
        t[ 2 ] = j / ( nce[ 0 ] * nce[ 1 ] );
        dev.launch( ct_id_dbg_k< bdt_T, itg_T >, n_bl,
                    dim3( n_th ), 0, str, b_d, f_itg, use_ee,
                    flat, t, nul );
        dev.sync_stream( str );
    }
    return;
}

};                      // namespace dynamics::mhd::integrator

#endif                  // DBG_CT_ID_DBG