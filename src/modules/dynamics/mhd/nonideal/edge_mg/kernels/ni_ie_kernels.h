//  ni_ie_kernels.h -- the __global__ layer of the implicit-step kernels.
//  Only kernels live here ( the project rule:  `__global__` appears nowhere else ).
//  Each kernel takes its op as a by-value policy object, so the policy type is a
//  compile-time argument and the kernel body never branches on a runtime tag.
#pragma once

#include "../edge_mg_cplx.h"
#include "../edge_mg_data.h"
#include "../edge_mg_geo.h"
#include "../../../../../../utilities/mapping/reduction.h"

namespace mhd_ee::ni::edge_mg
{
using type::  float_t;
using type:: float2_t;

////////////////////////////////////////////////////////////
//  the geometry helpers the kernels below differentiate on.
//  They are host+device inline,  and they live here so that the
//  kernels do not depend on include order inside edge_mg_grad.h.
////////////////////////////////////////////////////////////

__host__ __device__ inline float2_t eta_theta_centre
( const geo_sph_t & g, const int & j )
{
    const float2_t ta = g.th0 + float2_t( j ) * g.dth;
    const float2_t ma = cos( ta );
    const float2_t mb = cos( ta + g.dth );
    return acos( float2_t( 0.5 ) * ( ma + mb ) );
};
__host__ __device__ inline float2_t eta_r_centre
( const geo_sph_t & g, const int & i )
{
    return g.r0 + ( float2_t( i ) + float2_t( 0.5 ) ) * g.dr;
};
__host__ __device__ inline float2_t eta_ph_centre
( const geo_sph_t & g, const int & k )
{
    return g.ph0 + ( float2_t( k ) + float2_t( 0.5 ) ) * g.dph;
};
__host__ __device__ inline void eta_d1_stencil
( const int & i, const int & n, const float2_t & cm2,
  const float2_t & cm, const float2_t & c0, const float2_t & cp,
  const float2_t & cp2, int ( & id )[ 3 ], float_t ( & w )[
      3 ] )
{
    if( i >= 2 && i <= n - 3 )
    {
        //  CENTRED on a non-uniform grid: nodes at -a, 0,
        //  +b
        const float_t a = float_t( c0 ) - float_t( cm );
        const float_t b = float_t( cp ) - float_t( c0 );
        w[ 0 ] = -b / ( a * ( a + b ) );
        w[ 1 ] = ( b - a ) / ( a * b );
        w[ 2 ] =  a / ( b * ( a + b ) );
        id[ 0 ] = i - 1;
        id[ 1 ] = i;
        id[ 2 ] = i + 1;
        return;
    }
    if( i < 2 )
    {
        //  FORWARD,  anchored at i:  0, a, a+b
        const float_t a = float_t( cp ) - float_t( c0 );
        const float_t b = float_t( cp2 ) - float_t( cp );
        w[ 0 ] = -( 2. * a + b ) / ( a * ( a + b ) );
        w[ 1 ] =  ( a + b ) / ( a * b );
        w[ 2 ] = -a / ( ( a + b ) * b );
        id[ 0 ] = i;
        id[ 1 ] = i + 1;
        id[ 2 ] = i + 2;
        return;
    }
    //  BACKWARD, anchored at i: the mirror image, and its
    //  weights are
    //  the NEGATED forward ones ( g( t ) = f( -t ) => g'( 0
    //  ) = -f'( 0 ) )
    const float_t a = float_t( c0 ) - float_t( cm );
    const float_t b = float_t( cm ) - float_t( cm2 );
    w[ 0 ] =  ( 2. * a + b ) / ( a * ( a + b ) );
    w[ 1 ] = -( a + b ) / ( a * b );
    w[ 2 ] =  a / ( ( a + b ) * b );
    id[ 0 ] = i;
    id[ 1 ] = i - 1;
    id[ 2 ] = i - 2;
    return;
};
__host__ __device__ inline void j_from_curl
( const float2_t ( & curl_f )[ 6 ], const float2_t ( & area )[ 6 ],
  float2_t ( & jc )[ 3 ] )
{
    for( int a = 0; a < 3; ++ a )
    {
        const int lo = 2 * a;
        const int hi = 2 * a + 1;
        const bool vl = area[ lo ] > float2_t( 0 );
        const bool vh = area[ hi ] > float2_t( 0 );
        if( vl && vh )
        {
            jc[ a ] = float2_t( 0.5 )
                * ( curl_f[ hi ] / area[ hi ]
                  + curl_f[ lo ] / area[ lo ] );
        }
        else if( vl || vh )
        {
            jc[ a ] = vh ? curl_f[ hi ] / area[ hi ]
                         : curl_f[ lo ] / area[ lo ];
        }
        else
        {
            jc[ a ] = float2_t( 0 );
        }
    }
    return;
};

//============================================================
//  from edge_mg_eta_dev.h
//============================================================

template< class op_T >
__global__ void eta_xc_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T >
__global__ void eta_lat_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

//============================================================
//  from edge_mg_grad.h
//============================================================

static __global__ void eta_grad_k
( const geo_sph_t g, const cplx_t c, const float2_t * p_lat,
  float2_t * p_gr )
{
    const int nr  = int( c.n[ 0 ] );
    const int nt  = int( c.n[ 1 ] );
    const int nph = int( c.n[ 2 ] );
    const int n_c = int( c.n_cell );
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_c; ic += stride )
    {
        const int i  = ic % nr;
        const int j  = ( ic / nr ) % nt;
        const int k2 = ic / ( nr * nt );
        const int km = ( k2 + nph - 1 ) % nph;
        const int kp = ( k2 + 1 ) % nph;
        const float2_t crm2 = eta_r_centre    ( g, i > 1 ? i - 2 : i );
        const float2_t crm  = eta_r_centre    ( g, i > 0 ? i - 1 : i );
        const float2_t cr   = eta_r_centre    ( g, i );
        const float2_t crp  = eta_r_centre    ( g, i + 1 < nr ? i + 1 : i );
        const float2_t crp2 = eta_r_centre    ( g, i + 2 < nr ? i + 2 : i );
        const float2_t ctm2 = eta_theta_centre( g, j > 1 ? j - 2 : j );
        const float2_t ctm  = eta_theta_centre( g, j > 0 ? j - 1 : j );
        const float2_t ct   = eta_theta_centre( g, j );
        const float2_t ctp  = eta_theta_centre( g, j + 1 < nt ? j + 1 : j );
        const float2_t ctp2 = eta_theta_centre( g, j + 2 < nt ? j + 2 : j );
        int    idr[ 3 ], idt[ 3 ];
        float_t  wr[ 3 ],  wt[ 3 ];
        eta_d1_stencil( i, nr, crm2, crm, cr, crp, crp2, idr, wr );
        eta_d1_stencil( j, nt, ctm2, ctm, ct, ctp, ctp2, idt, wt );
        float_t g0[ 18 ];
        //  the strides are in CELLS, and a cell carries SIX
        //  numbers
        const int s_j  = nr * 6;
        const int s_k  = nr * nt * 6;
        for( int q = 0; q < 6; ++ q )
        {
            const int base = ic * 6 + q;
            //  EVERY offset is a difference of CELL indices
            //  times the
            //  stride of that direction, and the r stride
            //  is ONE CELL =
            //  SIX numbers. Writing `base + idr[u]` instead
            //  of
            //  `base + ( idr[u] - i ) * 6` reads the middle
            //  of another
            //  cell's tensor: with a CONSTANT eta the
            //  gradient then comes out as garbage rather than
            //  zero, which is why every offset is written as
            //  a difference of cell indices.
            const float_t vr0 = float_t( p_lat[ base + (
                idr[ 0 ] - i ) * 6 ] );
            const float_t vr1 = float_t( p_lat[ base + (
                idr[ 1 ] - i ) * 6 ] );
            const float_t vr2 = float_t( p_lat[ base + (
                idr[ 2 ] - i ) * 6 ] );
            const float_t vt0 = float_t( p_lat[ base + (
                idt[ 0 ] - j ) * s_j ] );
            const float_t vt1 = float_t( p_lat[ base + (
                idt[ 1 ] - j ) * s_j ] );
            const float_t vt2 = float_t( p_lat[ base + (
                idt[ 2 ] - j ) * s_j ] );
            const float_t vp0 = float_t( p_lat[ base + ( km
                - k2 ) * s_k ] );
            const float_t vp1 = float_t( p_lat[ base + ( kp
                - k2 ) * s_k ] );
            g0[      q ] = wr[ 0 ] * vr0 + wr[ 1 ] * vr1 + wr[ 2 ] * vr2;
            g0[  6 + q ] = wt[ 0 ] * vt0 + wt[ 1 ] * vt1 + wt[ 2 ] * vt2;
            g0[ 12 + q ] = ( vp1 - vp0 ) / ( 2. * float_t(
                g.dph ) );
        }
        const float_t th = float_t( eta_theta_centre( g, j )
            );
        const float_t ph = float_t( eta_ph_centre( g, k2 )
            );
        const float_t st = sin( th ), ctt = cos( th );
        const float_t sp = sin( ph ), cp = cos( ph );
        const float_t er[ 3 ] = { st * cp, st * sp, ctt };
        const float_t et[ 3 ] = { ctt * cp, ctt * sp, -st };
        const float_t ep[ 3 ] = { -sp, cp, 0. };
        const float_t r  = float_t( eta_r_centre( g, i ) );
        const float_t cs[ 3 ] = { 1., r, r * st };
        for( int a = 0; a < 3; ++ a )
        {
            for( int q = 0; q < 6; ++ q )
            {
                p_gr[ ic * 18 + a * 6 + q ]
                    = float2_t(  er[ a ] * g0[      q ] / cs[ 0 ]
                               + et[ a ] * g0[  6 + q ] / cs[ 1 ]
                               + ep[ a ] * g0[ 12 + q ] / cs[ 2 ] );
            }
        }
    }
    return;
};

//============================================================
//  from edge_mg_rep_dev.h
//============================================================

template< class op_T >
__global__ void rep_rhs_raw_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T >
__global__ void rep_cell_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

//============================================================
//  from edge_mg_wb_dev.h
//============================================================

static __global__ void wb0_face_k
( const int n_face, const wb_face_t * p_wf, const int * p_fed,
  const signed char * p_fsg, const signed char * p_fnz,
  const float_t * p_ee, const float_t dt, float2_t * p_bf )
{
    const int f = int( blockIdx.x * blockDim.x + threadIdx.x );
    if( f >= n_face )
    {
        return;
    }
    const wb_face_t & r = p_wf[ f ];
    if( r.deg != 0 )
    {
        return;
    }
    const float_t ar = r.ar;
    if( ! ( ar > float_t( 0 ) ) )
    {
        return;
    }
    const int nz = int( p_fnz[ f ] );
    float_t sum( 0 );
    for( int u = 0; u < nz; ++ u )
    {
        sum += float_t( p_fsg[ f * 4 + u ] )
             * p_ee[ p_fed[ f * 4 + u ] ];
    }
    const float_t inc = - dt * sum;
    p_bf[ r.slot ] += float2_t( inc / ar );
    return;
};

static __global__ void wb0_edge_k
( const int n_edge, const int * p_eptr, const int * p_ecell,
  const float_t * p_edl, const signed char * p_ewall,
  const float_t * p_ej, const int bc_edge, float_t * p_ee )
{
    const int e = int( blockIdx.x * blockDim.x + threadIdx.x );
    if( e >= n_edge )
    {
        return;
    }
    const int b = p_eptr[ e ];
    const int n = p_eptr[ e + 1 ] - b;
    float_t acc( 0 );
    for( int q = 0; q < n; ++ q )
    {
        const int cc = p_ecell[ b + q ];
        acc += p_ej[ cc * 3     ] * p_edl[ e * 3     ]
             + p_ej[ cc * 3 + 1 ] * p_edl[ e * 3 + 1 ]
             + p_ej[ cc * 3 + 2 ] * p_edl[ e * 3 + 2 ];
    }
    float_t ee( 0 );
    if( n > 0 )
    {
        ee = acc / float_t( n );
    }
    if( bc_edge == 1 && p_ewall[ e ] != 0 )
    {
        ee = float_t( 0 );
    }
    p_ee[ e ] = ee;
    return;
};

static __global__ void wb0_cell_k
( const int n_cell, const int * p_cf, const float_t * p_ar6,
  const signed char * p_fdg, const signed char * p_fsg,
  const signed char * p_fnz, const int * p_fed,
  const int * p_aslot, const float_t * p_a,
  const float2_t * p_lat, float_t * p_ej )
{
    const int ic = int( blockIdx.x * blockDim.x + threadIdx.x );
    if( ic >= n_cell )
    {
        return;
    }
    float2_t cf[ 6 ], ar[ 6 ];
    for( int s = 0; s < 6; ++ s )
    {
        const int gf = p_cf[ ic * 6 + s ];
        ar[ s ] = float2_t( p_ar6[ ic * 6 + s ] );
        if( p_fdg[ gf ] != 0 )
        {
            cf[ s ] = float2_t( 0 );
            continue;
        }
        const int nz = int( p_fnz[ gf ] );
        float2_t t( 0 );
        for( int u = 0; u < nz; ++ u )
        {
            t += float2_t( p_fsg[ gf * 4 + u ] )
               * float2_t( p_a[ p_aslot[ p_fed[ gf * 4 + u ] ] ] );
        }
        cf[ s ] = t;
    }
    float2_t j3[ 3 ];
    j_from_curl( cf, ar, j3 );
    const float2_t * e = p_lat + ic * 6;
    p_ej[ ic * 3     ] = float_t( e[ 0 ] * j3[ 0 ]
                                + e[ 3 ] * j3[ 1 ]
                                + e[ 4 ] * j3[ 2 ] );
    p_ej[ ic * 3 + 1 ] = float_t( e[ 3 ] * j3[ 0 ]
                                + e[ 1 ] * j3[ 1 ]
                                + e[ 5 ] * j3[ 2 ] );
    p_ej[ ic * 3 + 2 ] = float_t( e[ 4 ] * j3[ 0 ]
                                + e[ 5 ] * j3[ 1 ]
                                + e[ 2 ] * j3[ 2 ] );
    return;
};

static __global__ void wb_dev_k
( const wb_face_t * p_f, const int n_f, float2_t * p_bf,
  const float_t * p_old, const float_t * p_new )
{
    const int f = int( blockIdx.x * blockDim.x + threadIdx.x );
    if( f >= n_f )
    {
        return;
    }
    const wb_face_t & r = p_f[ f ];
    if( r.deg != 0 )
    {
        return;
    }
    const float_t ar = r.ar;
    if( ! ( ar > float_t( 0 ) ) )
    {
        return;
    }
    const float_t bh   = float_t( p_bf[ r.slot ] );
    const float_t flux = bh * ar + p_new[ f ] - p_old[ f ];
    p_bf[ r.slot ] = float2_t( flux / ar );
    return;
};

//  |div b| of the CT state,  one thread per cell along r.
//  Naive on purpose:  it reads the very face fluxes the
//  write-back will have to preserve,  so a stuck value here
//  means the view is mis-wired, not that the physics is
//  off.

static __global__ void ee_div_k
( const view_t v, float_t * p_dmax )
{
    const auto & geo = v.geo;
    const idx_t  n   = geo.n_ceff;
    idx_t i( 0, int( blockIdx.x ), int( blockIdx.y ) );
    for( int j = threadIdx.x; j < n[ 0 ]; j += blockDim.x )
    {
        i[ 0 ] = j;
        float2_t dv( 0 );
        for( int a = 0; a < 3; ++ a )
        {
            if( n[ a ] == 1 )
            {
                continue;      // thin / periodic wrap
            }
            idx_t k_lo( i );
            idx_t k_hi( i );
            ++ k_hi[ a ];
            dv += float2_t( v.bf.at( k_hi )[ a ] )
                * float2_t( geo.surface( k_hi, a ) )
                - float2_t( v.bf.at( k_lo )[ a ] )
                * float2_t( geo.surface( k_lo, a ) );
        }
        const float_t vol = geo.volume( i );
        if( vol <= 0 )
        {
            continue;
        }
        utils::atomic_max
            ( p_dmax, float_t( __abs1( dv ) ) / vol );
    }
    return;
};

}  // namespace mhd_ee::ni::edge_mg
