#pragma once

#include "fld_op.h"
#include "fld_m1.h"

namespace rad::fld
{

////////////////////////////////////////////////////////////
//  FLD kernel functors:  the operations live in the
// functor's  __device__ operator( )  ( the dynamics /
// prototypes pattern -- the  __global__  kernels below
// only forward );  the per - launch parameters are the
// functor members.   Design notes:  fld - method.md.

//  Seed  src_dat  with  E^n + dt ( S + a T^4 )  and fill
// the phybnd ghost slots with the zero - gradient
// continuation  ( the MG transfers read them;  a stale 0
// compounds a boundary error per step ).

struct src_fun_t
{
    type::float_t src_emit;
    type::float_t t_emit;
    int           s_on;
    int           s_snap;   //  also snapshot  s_base
    int           steady;   //  drop E^n  ( equilibrium )

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const auto & n_gm = b_d.mgd.fld.n_gh;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    N0  = n0 + 2 * n_gm[ 0 ];
        const int    N1  = n1 + 2 * n_gm[ 1 ];
        const int    N2  = n2 + 2 * n_gm[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= N0 * N1 * N2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % N0 - n_gm[ 0 ];
        i[ 1 ] = ( t / N0 ) % N1 - n_gm[ 1 ];
        i[ 2 ] =   t / ( N0 * N1 ) - n_gm[ 2 ];
        if( ! in_real( i, n0, n1, n2 ) )
        {
            ghost_fill( b_d, i );
            if( s_snap )
                b_d.s_base.at( i )[ 0 ]
                    = b_d.mgd.src_dat.at( i )[ 0 ];
            return;
        };
        type::float_t s_ext( src_emit );
        if( s_on )
        {
            const auto * s_i( b_d.s_ir.at( i ) );
            s_ext += s_i[ 0 ] + s_i[ 1 ];
        }
        const auto s_v( ( steady
                          ? type::float_t( 0 )
                          : b_d.mgd.fld.at( i )[ 0 ] )
                        + s_ext + t_emit );
        b_d.mgd.src_dat.at( i )[ 0 ] = s_v;
        if( s_snap )
            b_d.s_base.at( i )[ 0 ] = s_v;
        return;
    };

    private:
    template< class idx_T >
    __device__ __forceinline__ static bool in_real
    ( const idx_T & i, const int & n0,
      const int & n1, const int & n2 )
    {
        return    i[ 0 ] >= 0 && i[ 0 ] < n0
               && i[ 1 ] >= 0 && i[ 1 ] < n1
               && i[ 2 ] >= 0 && i[ 2 ] < n2;
    };

    //  Zero - gradient continuation into the phybnd ghost
    // slots  ( i  clamps to the boundary cell ).
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ static void ghost_fill
    ( const bdt_T & b_d, const idx_T & i )
    {
        bool fill( false );
        for( int a = 0; a < 3; ++ a )
        {
            if( i[ a ] < 0
                && b_d.geo.phybnd_flag[ a ][ 0 ] )
                fill = true;
            if( i[ a ] >= b_d.geo.n_ceff[ a ]
                && b_d.geo.phybnd_flag[ a ][ 1 ] )
                fill = true;
        }
        if( ! fill )
            return;
        type::idx_t j( i );
        for( int a = 0; a < 3; ++ a )
        {
            if( j[ a ] < 0 )
                j[ a ] = 0;
            if( j[ a ] >= b_d.geo.n_ceff[ a ] )
                j[ a ] = b_d.geo.n_ceff[ a ] - 1;
        }
        b_d.mgd.fld.at( i )[ 0 ]
            = b_d.mgd.fld.at( j )[ 0 ];
        return;
    };
};

//  The Robin inhomogeneous source:  add  F dt / dx  to
// the boundary row's  src_dat  ( derivation:  fld -
// method.md § 3 ).   Only the boundary cell of each
// enabled  ( a, k )  is touched.

struct robin_p_t
{
    int           robin[ 3 ][ 2 ];
    type::float_t rsrc [ 3 ][ 2 ];

    __host__ robin_p_t(  ) = default;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        for( int a = 0; a < geo.n_dim; ++ a )
            for( int k = 0; k < 2; ++ k )
            {
                if( robin[ a ][ k ] <= 0 )
                    continue;
                if( ! geo.phybnd_flag[ a ][ k ] )
                    continue;
                const bool on( k == 0
                    ? ( i[ a ] == 0 )
                    : ( i[ a ] == geo.n_ceff[ a ] - 1 ) );
                if( ! on )
                    continue;
                b_d.mgd.src_dat.at( i )[ 0 ]
                    += rsrc[ a ][ k ];
            }
        return;
    };
};

//  The Dirichlet anchor,  as the incoming half of the
// Marshak pair:   the operator's  vac_bnd  outflow
// ( c E / 4 )  is completed by  + c E_dir / 4  on the RHS
// at the flagged physical faces,  so the boundary  E
// relaxes to  E_dir  ( the  T_min  environment --  the
// spherical - grid analog of the periodic  mean
// subtraction;   seeliger - literature.md § 2 ).   The
// pair shares  c4  and the same cell  dx  convention as
// the operator's  w_side.

struct anchor_fun_t
{
    int           dir[ 3 ][ 2 ];
    type::float_t c4;
    type::float_t e_dir;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        for( int a = 0; a < geo.n_dim; ++ a )
            for( int k = 0; k < 2; ++ k )
            {
                if( dir[ a ][ k ] <= 0 )
                    continue;
                if( ! geo.phybnd_flag[ a ][ k ] )
                    continue;
                const bool on( k == 0
                    ? ( i[ a ] == 0 )
                    : ( i[ a ] == geo.n_ceff[ a ] - 1 ) );
                if( ! on )
                    continue;
                const auto dxc( geo.x_cc( a, 1 )
                                - geo.x_cc( a, 0 ) );
                b_d.mgd.src_dat.at( i )[ 0 ]
                    += c4 * e_dir / dxc;
            }
        return;
    };
};

//  The flux limiter  lambda( R ):
//  0 = 1/3  ( the pure - diffusion gate );
//  1 = Levermore - Pomraning;  2 = Minerbo.

__device__ __forceinline__ type::float_t fld_lam
( const type::float_t & R, const int & limiter )
{
    if( limiter == 0 )
        return type::float_t( 1.0 / 3.0 );
    const type::float_t r( R );
    if( r < type::float_t( 1e-12 ) )
        return type::float_t( 1.0 / 3.0 );
    type::float_t lam;
    if( limiter == 1 )
    {
        const type::float_t r2( 1 / r );
        const type::float_t ch
            ( r > type::float_t( 20 ) ? 1
            : 1 / tanh( r ) );
        lam = ( ch - r2 ) / r;
    }
    else
    {
        lam = 1 / ( 1 + sqrt( 1 + 4 * r * r ) );
        if( lam > type::float_t( 1.0 / 3.0 ) )
            lam = type::float_t( 1.0 / 3.0 );
    }
    return lam;
};

//  Freeze the coefficients into  co_arr  ( level 0 ).
//  The kappa profile gives the SHAPE ( D ~ lam / kappa,
// sig ~ kappa )  against the uniform normalization;  the
// limiter's  lambda  is evaluated on the current  E  --
// the full scaling:  fld - method.md § 4.

struct freeze_fun_t
{
    int           var_coef;
    int           kappa_var;
    int           limiter;
    type::float_t d_unif;
    type::float_t s_unif;
    type::float_t kap0, kap1;
    type::float_t c_light;
    type::float_t rho;
    type::float_t dt;
    type::float_t e_floor;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        if( ! var_coef )
        {
            b_d.co_arr.at( i )[ 0 ] = d_unif;
            b_d.co_arr.at( i )[ 1 ] = s_unif;
            return;
        };
        const auto lam
            ( limiter == 0
            ? type::float_t( 1.0 / 3.0 )
            : lam_of_R( b_d, i ) );
        const auto kap( kap_of_x( geo, i ) );
        const auto f_k( kap / kap0 );
        b_d.co_arr.at( i )[ 0 ] = d_unif / f_k
            * type::float_t( 3 ) * lam;
        b_d.co_arr.at( i )[ 1 ] = s_unif * f_k;
        return;
    };

    private:
    //  kappa( x )  with the optional power - law shape.
    template< class geo_T, class idx_T >
    __device__ __forceinline__ auto kap_of_x
    ( const geo_T & geo, const idx_T & i ) const
    {
        type::float_t kap( kap0 );
        if( kappa_var )
        {
            const auto x( geo.x_cc( 0, i[ 0 ] ) );
            kap = kap0 + ( kap1 - kap0 ) * x;
            if( kap < e_floor )
                kap = e_floor;
        }
        return kap;
    };

    //  lambda( R )  on the current field;  R =
    // kappa rho |grad E| / E  with  kappa rho = 1 / ( 3
    // D_unif )  (  d_unif  is the dt - scaled  D ).
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ auto lam_of_R
    ( const bdt_T & b_d, const idx_T & i ) const
    {
        const auto & geo( b_d.geo );
        type::float_t g2( 0 );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            const int nax( geo.n_ceff[ a ] );
            if( nax < 2 )
                continue;
            type::idx_t j( i ), k( i );
            -- j[ a ];  ++ k[ a ];
            const auto dE
                ( b_d.mgd.fld.at( k )[ 0 ]
                - b_d.mgd.fld.at( j )[ 0 ] );
            const auto dx_c
                ( geo.x_cc( a, 1 ) - geo.x_cc( a, 0 ) );
            const auto g( dE
                / ( type::float_t( 2 ) * dx_c ) );
            g2 += g * g;
        }
        const auto e_c( b_d.mgd.fld.at( i )[ 0 ] );
        const auto k_rho
            ( d_unif > e_floor
            ? dt / ( type::float_t( 3 ) * d_unif )
            : type::float_t( 0 ) );
        const type::float_t R
            (    e_c > e_floor
              && k_rho > type::float_t( 0 )
            ? sqrt( g2 ) / e_c / k_rho
            : type::float_t( 0 ) );
        return fld_lam( R, limiter );
    };
};

//  Fill the level - 0  phybnd ghost slots of  mgd.fld
// with the Robin closure  E_g = E_i + dx F / D  ( the
// inflow ghost sits HIGHER;  fld - method.md § 3 )  so
// the limiter's  R  at the boundary cells is right.

struct ghost_fun_t
{
    int           robin[ 3 ][ 2 ];
    type::float_t f_inc;
    type::float_t d_unif;

    __host__ ghost_fun_t
    ( const int ( & rb )[ 3 ][ 2 ],
      const type::float_t & f,
      const type::float_t & dv )
    {
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
                robin[ a ][ k ] = rb[ a ][ k ];
        f_inc = f;
        d_unif = dv;
        return;
    };
    __host__ ghost_fun_t(  ) = default;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        for( int a = 0; a < geo.n_dim; ++ a )
            for( int k = 0; k < 2; ++ k )
            {
                if( ! geo.phybnd_flag[ a ][ k ] )
                    continue;
                const int nax( geo.n_ceff[ a ] );
                if( nax < 2 )
                    continue;
                fill_plane( b_d, a, k, nax );
                return;  //  one plane per pass
            }
        return;
    };

    private:
    template< class bdt_T >
    __device__ __forceinline__ void fill_plane
    ( const bdt_T & b_d, const int & a,
      const int & k, const int & nax ) const
    {
        const auto & geo( b_d.geo );
        const int i_b( k == 0 ? 0 : nax - 1 );
        type::idx_t i;
        i[ 0 ] = 0;  i[ 1 ] = 0;  i[ 2 ] = 0;
        i[ a ] = i_b;
        const auto e_i( b_d.mgd.fld.at( i )[ 0 ] );
        const auto dx_c
            ( geo.x_cc( a, 1 ) - geo.x_cc( a, 0 ) );
        const auto e_g
            ( robin[ a ][ k ] > 0
            ? e_i + dx_c * f_inc / d_unif
            : e_i );        //  free:  E_g = E_i
        //  Uniform slab fill  ( only the axis - aligned
        // neighbours are read;  cheap either way ).
        type::idx_t q;
        for( q[ 2 ] = 0;
             q[ 2 ] < geo.n_ceff[ 2 ]; ++ q[ 2 ] )
        for( q[ 1 ] = 0;
             q[ 1 ] < geo.n_ceff[ 1 ]; ++ q[ 1 ] )
        for( q[ 0 ] = 0;
             q[ 0 ] < geo.n_ceff[ 0 ]; ++ q[ 0 ] )
        {
            if( q[ a ] != i_b )
                continue;
            type::idx_t g( q );
            g[ a ] = k == 0 ? -1 : nax;
            b_d.mgd.fld.at( g )[ 0 ] = e_g;
        }
        return;
    };
};

//  Snapshot the iterate into  e_iter  ( the Picard
// change measure ).

struct snap_fun_t
{
    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        b_d.e_iter.at( i )[ 0 ]
            = b_d.mgd.fld.at( i )[ 0 ];
        return;
    };
};

//  The Picard change:  max | E - e_iter |  and  max | E |
// through the module - level device buffers  ( atomics ).

struct chg_fun_t
{
    type::float_t * chg_d;
    type::float_t * fmax_d;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        const auto e_n( b_d.mgd.fld.at( i )[ 0 ] );
        const auto e_o( b_d.e_iter.at( i )[ 0 ] );
        const auto d( fabs( e_n - e_o ) );
        if( d > * chg_d )
            utils::atomic_max( chg_d, d );
        if( fabs( e_n ) > * fmax_d )
            utils::atomic_max( fmax_d, fabs( e_n ) );
        return;
    };
};

//  Restrict the frozen coefficients onto the coarse
// level  ( the 2 x 2 x 2 arithmetic mean,  the mirror of
// mg_cc_rcoarse_k ).

struct rcoarse_fun_t
{
    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_c, const bdt_T & b_f ) const
    {
        const auto & geo ( b_c.geo );
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= geo.n_ceff.prod(  ) )
            return;
        type::idx_t  i;
        i[ 0 ] =   t % geo.n_ceff[ 0 ];
        i[ 1 ] = ( t / geo.n_ceff[ 0 ] )
                 % geo.n_ceff[ 1 ];
        i[ 2 ] =   t / ( geo.n_ceff[ 0 ]
                         *  geo.n_ceff[ 1 ] );
        restrict_cell( b_c, b_f, i );
        return;
    };

    private:
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ void restrict_cell
    ( const bdt_T & b_c, const bdt_T & b_f,
      const idx_T & i ) const
    {
        const auto & geo( b_c.geo );
        const auto m_dim( 1 << geo.n_dim );
        //  All  8  slots  (  D dt,  sig,  chi x 6  --  the
        // M1 path;  the FLD path leaves  2.. 7  zero ).
        type::float_t sum[ 8 ] = { 0, 0, 0, 0,
                                   0, 0, 0, 0 };
        int n_s( 0 );
        type::idx_t j, dj;
        for( int l = 0; l < m_dim; ++ l )
        {
            for( int a = 0; a < 3; ++ a )
                dj[ a ] = ( a < geo.n_dim )
                    ? ( ( l >> a ) & 1 ) : 0;
            for( int a = 0; a < 3; ++ a )
                j[ a ] = 2 * i[ a ] + dj[ a ];
            const auto * c_f = b_f.co_arr.at( j );
            for( int n = 0; n < 8; ++ n )
                sum[ n ] += c_f[ n ];
            ++ n_s;
        }
        auto * c_c = b_c.co_arr.at( i );
        for( int n = 0; n < 8; ++ n )
            c_c[ n ] = sum[ n ] / n_s;
        return;
    };
};

//  SMR coefficient ghosts  ( the mirror of the nonideal
// rfill / xfill family,  with only the 2 scalar
// components;  derivation:  fld - method.md § 8 ).
//  Same - level face:  the ghost slab copies the
// neighbour's boundary cells verbatim.

template< class bdt_T >
__global__ void fld_rfill_k
( const bdt_T b_d, const bdt_T b_n,
  const int a, const int s )
{
    const auto & geo( b_d.geo );
    const int    t1( utils::mod3( a + 1 ) ),
                 t2( utils::mod3( a + 2 ) );
    const int    t ( blockIdx.x * blockDim.x
                     + threadIdx.x );
    if( t >= geo.n_ceff[ t1 ] * geo.n_ceff[ t2 ] )
        return;
    type::idx_t i, j;
    i[ a  ] = s ? geo.n_ceff[ a ] : -1;
    j[ a  ] = s ? 0 : b_n.geo.n_ceff[ a ] - 1;
    i[ t1 ] = t % geo.n_ceff[ t1 ];
    i[ t2 ] = t / geo.n_ceff[ t1 ];
    j[ t1 ] = i[ t1 ];   j[ t2 ] = i[ t2 ];
    const auto * src( b_n.co_arr.at( j ) );
    auto       * dst( b_d.co_arr.at( i ) );
    for( int q = 0; q < 8; ++ q )
        dst[ q ] = src[ q ];
    return;
};

//  Cross - level,  FINE side:  the ghost slot reads the
// covering coarse cell verbatim  ( the org mapping of
// the nonideal  xfill ).

template< class bdt_T >
__global__ void fld_xfill_k
( const bdt_T b_f, const bdt_T b_c,
  const int a, const int s,
  const type::idx_t org_f, const type::idx_t org_c )
{
    const auto & gf( b_f.geo );
    const auto & gc( b_c.geo );
    const int    t1( utils::mod3( a + 1 ) ),
                 t2( utils::mod3( a + 2 ) );
    const int    t ( blockIdx.x * blockDim.x
                     + threadIdx.x );
    if( t >= gf.n_ceff[ t1 ] * gf.n_ceff[ t2 ] )
        return;
    type::idx_t i, j;
    i[ a ] = s ? gf.n_ceff[ a ] : -1;
    j[ a ] = s ? 0 : gc.n_ceff[ a ] - 1;
    i[ t1 ] = t % gf.n_ceff[ t1 ];
    i[ t2 ] = t / gf.n_ceff[ t1 ];
    bool in( true );
    for( int b = 0; b < 3; ++ b )
    {
        if( b == a )
            continue;
        j[ b ] = ( org_f[ b ] * gf.n_ceff[ b ] + i[ b ] )
               / 2 - org_c[ b ] * gc.n_ceff[ b ];
        if( j[ b ] < 0 || j[ b ] >= gc.n_ceff[ b ] )
            in = false;
    }
    if( ! in )
        return;
    const auto * src( b_c.co_arr.at( j ) );
    auto       * dst( b_f.co_arr.at( i ) );
    for( int q = 0; q < 8; ++ q )
        dst[ q ] = src[ q ];
    return;
};

//  Cross - level,  COARSE side:  the ghost slot receives
// the average of the covering 2 x 2 fine boundary cells.

template< class bdt_T >
__global__ void fld_xfill_c_k
( const bdt_T b_c, const bdt_T b_f,
  const int a, const int s,
  const type::idx_t org_c, const type::idx_t org_f )
{
    const auto & gc( b_c.geo );
    const auto & gf( b_f.geo );
    const int    t1( utils::mod3( a + 1 ) ),
                 t2( utils::mod3( a + 2 ) );
    const int    t ( blockIdx.x * blockDim.x
                     + threadIdx.x );
    if( t >= gc.n_ceff[ t1 ] * gc.n_ceff[ t2 ] )
        return;
    type::idx_t i, j;
    i[ a ] = s ? gc.n_ceff[ a ] : -1;
    j[ a ] = s ? 0 : gf.n_ceff[ a ] - 1;
    i[ t1 ] = t % gc.n_ceff[ t1 ];
    i[ t2 ] = t / gc.n_ceff[ t1 ];
    type::float_t sum[ 2 ] = { 0, 0 };
    int n_s( 0 );
    for( int e1 = 0; e1 < 2; ++ e1 )
    for( int e2 = 0; e2 < 2; ++ e2 )
    {
        bool in( true );
        for( int b = 0; b < 3; ++ b )
        {
            if( b == a )
                continue;
            const int e( b == t1 ? e1 : e2 );
            j[ b ] = 2 * ( org_c[ b ] * gc.n_ceff[ b ]
                           + i[ b ] ) + e
                   - org_f[ b ] * gf.n_ceff[ b ];
            if( j[ b ] < 0 || j[ b ] >= gf.n_ceff[ b ] )
                in = false;
        }
        if( ! in )
            continue;
        const auto * src( b_f.co_arr.at( j ) );
        for( int q = 0; q < 8; ++ q )
            sum[ q ] += src[ q ];
        ++ n_s;
    }
    if( n_s == 0 )
        return;
    auto * dst( b_c.co_arr.at( i ) );
    for( int q = 0; q < 8; ++ q )
        dst[ q ] = sum[ q ] / n_s;
    return;
};

////////////////////////////////////////////////////////////
//  Generic kernel functions  ( one - line forwarders
// only,  the dynamics / prototypes pattern ).

template< class bdt_T, class fun_T >
__global__ void fld_fun_k
( const bdt_T b_d, const fun_T fun )
{
    return fun( b_d );
};

template< class bdt_T, class fun_T >
__global__ void fld_fun2_k
( const bdt_T b_c, const bdt_T b_f, const fun_T fun )
{
    return fun( b_c, b_f );
};

////////////////////////////////////////////////////////////
//  The FLD driver

template< class prx_T >
struct fld_t : fld_mg_base_t
{
    using base_t = fld_mg_base_t;
    using  dev_t = device::base_t;
    using f_t    = type::float_t;

    ////////// Data //////////
    bool     enabled;
    //  Knobs  ( the  [fld]  pars;  scaling notes:
    //  fld - method.md § 1 ):
    f_t      d_unif;        //  D  ( NOT dt - scaled )
    f_t      sig_unif;      //  c kappa_a rho
    int      var_coef;
    int      kappa_var;
    int      limiter;       //  0/1/2 = 1/3/LP/Minerbo
    int      n_picard;
    f_t      r_tol;
    f_t    * p_chg_d;
    f_t    * p_fmax_d;
    f_t      kap0, kap1;
    f_t      c_light;
    f_t      rho;
    f_t      src_emit;
    f_t      e_floor;
    int      s_on;          //  per - cell src on
    int      freeze_byp;    //  host owns the co freeze
    int      t_coup;        //  0 = off
    int      closure;       //  0 = FLD scalar D,  1 = M1
    int      lte_elim;      //  LTE analytic elimination
    int      args_newton;   //  [fld] newton closure
    int      args_steady;   //  [fld] steady (no E^n)
    int      steady_src;    //  wired into src_fun
    f_t      elim_sig;      //  the LTE mask threshold
    int      f_accum;       //  F rebuild accumulates
    int      chi_mode;      //  0 = Levermore,  1 = fixed
    f_t      chi_a[ 3 ];    //  fixed diag  (  test )
    f_t      a_rad;         //  radiation constant a
    f_t      t_matt;        //  matter temperature T
    int      robin[ 3 ][ 2 ];
    f_t      rbnd [ 3 ][ 2 ];   //  g1
    f_t      f_inc;         //  Marshak inflow flux
    int      dir[ 3 ][ 2 ]; //  Dirichlet anchor faces
    f_t      e_dir;         //  the anchor E
    int      mg_lvl;

    ////////// Functions //////////
    __host__ fld_t(  )
    {
        enabled   = false;
        d_unif    = f_t( 0 );
        sig_unif  = f_t( 0 );
        var_coef  = 0;
        kappa_var = 0;
        limiter   = 0;
        n_picard  = 1;
        r_tol     = f_t( 1e-4 );
        p_chg_d   = nullptr;
        p_fmax_d  = nullptr;
        kap0      = f_t( 0 );
        kap1      = f_t( 0 );
        c_light   = f_t( 1 );
        rho       = f_t( 1 );
        src_emit  = f_t( 0 );
        e_floor   = f_t( 1e-24 );
        s_on      = 0;
        freeze_byp = 0;
        t_coup    = 0;
        closure   = 0;
        lte_elim  = 0;
        args_newton = 0;
        args_steady = 0;
        steady_src  = 0;
        elim_sig  = f_t( 1 );
        f_accum   = 0;
        chi_mode  = 0;
        chi_a[ 0 ] = chi_a[ 1 ] = chi_a[ 2 ] = f_t( 1 )
                      / 3;
        a_rad     = f_t( 1 );
        t_matt    = f_t( 0 );
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
            {
                robin[ a ][ k ] = 0;
                rbnd [ a ][ k ] = f_t( 1 );
                dir  [ a ][ k ] = 0;
            }
        e_dir     = f_t( 0 );
        f_inc     = f_t( 0 );
        mg_lvl    = -1;
        return;
    };

    virtual void read( const input & args ) override
    {
        enabled = args.get< int >
            ( "fld", "enabled", enabled ? 1 : 0 ) > 0;
        d_unif   = args.get< f_t >
            ( "fld", "d_unif", d_unif );
        sig_unif = args.get< f_t >
            ( "fld", "sig_unif", sig_unif );
        var_coef = args.get< int >
            ( "fld", "var_coef", var_coef );
        limiter  = args.get< int >
            ( "fld", "limiter", limiter );
        kappa_var = args.get< int >
            ( "fld", "kappa_var", kappa_var );
        n_picard  = args.get< int >
            ( "fld", "n_picard", n_picard );
        r_tol     = args.get< f_t >
            ( "fld", "r_tol", r_tol );
        kap0     = args.get< f_t >
            ( "fld", "kappa", kap0 );
        if( kap0 <= f_t( 0 ) )
            kap0 = f_t( 1 );
        kap1     = args.get< f_t >
            ( "fld", "kappa1", kap1 );
        c_light  = args.get< f_t >
            ( "fld", "c_light", c_light );
        rho      = args.get< f_t >
            ( "fld", "rho", rho );
        src_emit = args.get< f_t >
            ( "fld", "src_emit", src_emit );
        s_on     = args.get< int >
            ( "fld", "s_on", s_on );
        closure  = args.get< int >
            ( "fld", "closure", 0 );
        lte_elim = args.get< int >
            ( "fld", "lte_elim", 0 );
        args_newton = args.get< int >
            ( "fld", "newton", 0 );
        args_steady = args.get< int >
            ( "fld", "steady", 0 );
        steady_src  = args_steady;
        elim_sig = args.get< f_t >
            ( "fld", "elim_sig", f_t( 1 ) );
        f_accum  = args.get< int >
            ( "fld", "f_accum", 0 );
        chi_mode = args.get< int >
            ( "fld", "chi_mode", 0 );
        {
            std::vector< f_t > c_a;
            args( "fld", "chi_a", c_a );
            if( c_a.size(  ) == 3 )
                for( int a = 0; a < 3; ++ a )
                    chi_a[ a ] = c_a[ a ];
        };
        freeze_byp = args.get< int >
            ( "fld", "freeze_byp", freeze_byp );
        t_coup   = args.get< int >
            ( "fld", "t_coup", t_coup );
        a_rad    = args.get< f_t >
            ( "fld", "a_rad", a_rad );
        t_matt   = args.get< f_t >
            ( "fld", "t_matt", t_matt );
        e_floor  = args.get< f_t >
            ( "fld", "e_floor", e_floor );
        f_inc    = args.get< f_t >
            ( "fld", "f_inc", f_inc );
        mg_lvl   = args.get< int >
            ( "multigrid", "mg_lvl", mg_lvl );
        read_robin( args );
        if( enabled )
        {
            enroll_opt(  );
            guard_singular(  );
        }
        return base_t::read( args );
    };

    //  Coarsening limit  ( 0 = single level / pure GS ).
    virtual int lvl_lim( mesh::mesh_t & mesh ) override
    {
        if( mg_lvl >= 0 )
            return mg_lvl;
        return base_t::lvl_lim( mesh );
    };

    virtual void init( const input & args,
                       mesh::mesh_t & mesh ) override
    {
        base_t::init( args, mesh );
        p_chg_d  = p_dev->malloc< f_t >( 1, false );
        p_fmax_d = p_dev->malloc< f_t >( 1, false );
        //  The M1 flux field starts ZEROED  (  the first
        // chi freeze sees the isotropic  F = 0  closure;
        //  malloc memory is garbage ):
        if( closure )
            for( auto & d : * this )
            {
                auto b_h = prx_T::d
                    ( prx_T::dual( d ).d_mg( 0 ) );
                p_dev->f_mset( b_h.f_arr.dat, 0,
                    b_h.f_arr.n_size, 0 );
            }
        return;
    };

    virtual void step( mesh::mesh_t & mesh ) override
    {
        if( ! enabled )
            return;
        auto & dev  = * this->p_dev;
        auto & opt  = static_cast< fld_operation_t & >
                      ( * this->p_opt );
        const auto  dt  = mesh.p_cyc->dt(  );
        opt.d_unif = d_unif * dt;
        opt.s_unif = sig_unif * dt;
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
                opt.rbnd[ a ][ k ] = rbnd[ a ][ k ];

        launch_src( mesh, dt );
        solve_picard( mesh, dt );
        advance_clock( mesh );
        return;
    };

    protected:
    //  The  [fld] robin_lo / robin_hi  bitmasks.
    void read_robin( const input & args )
    {
        const auto d_lo = args.get< int >
            ( "fld", "dir_lo", 0 );
        const auto d_hi = args.get< int >
            ( "fld", "dir_hi", 0 );
        for( int a = 0; a < 3; ++ a )
        {
            const auto lo = args.get< int >
                ( "fld", "robin_lo", 0 );
            const auto hi = args.get< int >
                ( "fld", "robin_hi", 0 );
            robin[ a ][ 0 ] = ( lo >> a ) & 1;
            robin[ a ][ 1 ] = ( hi >> a ) & 1;
            dir  [ a ][ 0 ] = ( d_lo >> a ) & 1;
            dir  [ a ][ 1 ] = ( d_hi >> a ) & 1;
        }
        e_dir = args.get< f_t >( "fld", "e_dir", e_dir );
        return;
    };

    //  Enroll the custom operator  ( BEFORE  base read,
    // else the default Poisson operator installs ).
    void enroll_opt(  )
    {
        this->n_fld = 1;
        this->template enroll< prx_T, fld_operation_t >
            (  );
        auto & opt = static_cast
            < fld_operation_t & > ( * this->p_opt );
        opt.var_coef = ( var_coef != 0 );
        opt.m1_on    = ( closure != 0 );
        opt.vac_bnd  = ( lte_elim != 0 );
        opt.sig_elim = ( lte_elim != 0 );
        opt.elim_sig = elim_sig;
        opt.newton   = ( args_newton != 0 );
        opt.steady   = ( args_steady != 0 );
        opt.c_free4  = c_light * f_t( 0.25 ) * args_steady;
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
                opt.robin[ a ][ k ] = robin[ a ][ k ];
        return;
    };

    //  Pure Neumann + zero absorption is the Poisson null
    // space -- refuse it.
    void guard_singular(  )
    {
        if(    sig_unif <= f_t( 0 ) && ! var_coef
            && ! f_inc )
            throw std::runtime_error
                ( "rad::fld:  zero absorption with no "
                  "Robin inflow leaves a singular "
                  "operator;  set sig_unif / var_coef "
                  "/ f_inc." );
        return;
    };

    //  Source seeding  ( E^n + dt S + a T^4 )  and the
    // Robin inhomogeneity  F dt / dx.
    void launch_src( mesh::mesh_t & mesh, const f_t & dt )
    {
        auto & dev = * this->p_dev;
        robin_p_t rp;
        //  The Marshak pair rate  (  mirrors
        // enroll_opt 's  c_free4 )  and the anchor gate:
        const f_t c_free4( c_light * f_t( 0.25 )
                           * args_steady );
        bool dir_any( false );
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
                dir_any = dir_any || ( dir[ a ][ k ] > 0 );
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
            {
                rp.robin[ a ][ k ] = robin[ a ][ k ];
                rp.rsrc [ a ][ k ] = f_t( 0 );
                if( robin[ a ][ k ] > 0 )
                    rp.rsrc [ a ][ k ] = f_inc * dt
                        * this->n_ax( mesh, a );
            }
        const f_t t_emit
            ( t_coup > 0
            ? sig_unif * dt * a_rad
              * t_matt * t_matt * t_matt * t_matt
            : f_t( 0 ) );
        for( auto & d : * this )
        {
            auto b_f = prx_T::d
                ( prx_T::dual( d ).d_mg( 0 ) );
            dev.launch
                ( fld_fun_k< decltype( b_f ),
                  src_fun_t >,
                  rsc_in( b_f ), d.stream,
                  b_f, src_fun_t{ src_emit * dt,
                                  t_emit, s_on,
                                  closure, steady_src } );
            if( f_inc != f_t( 0 ) )
                dev.launch
                ( fld_fun_k< decltype( b_f ),
                  robin_p_t >,
                  rsc_copy( b_f ), d.stream,
                  b_f, rp );
            //  The Dirichlet anchor  (  the Marshak
            // inflow half;  shares the operator's
            // c_free4  --  active only in  steady  mode,
            // matching  enroll_opt ):
            if( dir_any && c_free4 != f_t( 0 ) )
                dev.launch
                ( fld_fun_k< decltype( b_f ),
                  anchor_fun_t >,
                  rsc_copy( b_f ), d.stream,
                  b_f, anchor_fun_t{ { { dir[ 0 ][ 0 ],
                      dir[ 0 ][ 1 ] }, { dir[ 1 ][ 0 ],
                      dir[ 1 ][ 1 ] }, { dir[ 2 ][ 0 ],
                      dir[ 2 ][ 1 ] } }, c_free4,
                      e_dir } );
        }
        return;
    };

    //  Freeze -> solve,  wrapped in the Picard outer loop
    // when the limiter is on  ( fld - method.md § 4 ).
    void solve_picard( mesh::mesh_t & mesh, const f_t & dt )
    {
        auto & dev = * this->p_dev;
        auto & opt = static_cast< fld_operation_t & >
                     ( * this->p_opt );
        const bool picard( var_coef && limiter != 0 );
        for( int i_pc = 0;
             i_pc < ( picard ? n_picard : 1 ); ++ i_pc )
        {
            freeze_fun_t fz{ var_coef ? 1 : 0,
                kappa_var, limiter, d_unif * dt,
                sig_unif * dt, kap0, kap1, c_light, rho,
                dt, e_floor };
            rcoarse_fun_t rc;
            snap_fun_t  sn;
            if( ! freeze_byp )
                for( auto & d : * this )
                {
                    auto & dd( prx_T::dual( d ) );
                    //  b_f  is reassigned to the coarse
                    // blocks inside  launch_freeze  (  its
                    // restriction loop )  --  re - fetch
                    // the level - 0  block afterwards:
                    auto b_f = prx_T::d( dd.d_mg( 0 ) );
                    launch_freeze( dev, d, dd, b_f, fz, rc,
                                   sn, picard, dt );
                }
            this->sync_streams( mesh );
            if( var_coef )
                fill_co_ghost( mesh );
            if( closure )
            {
                //  src = s_base  then  the  m1_couple
                // defect  (  per Picard round  --  never
                // accumulates ):
                for( auto & d : * this )
                {
                    auto & dd( prx_T::dual( d ) );
                    auto b_f = prx_T::d( dd.d_mg( 0 ) );
                    dev.launch
                    ( fld_fun_k< decltype( b_f ),
                      m1_srcrst_fun_t >,
                      rsc_copy( b_f ), d.stream,
                      b_f, m1_srcrst_fun_t{ } );
                    dev.launch
                    ( fld_fun_k< decltype( b_f ),
                      m1_couple_fun_t >,
                      rsc_copy( b_f ), d.stream,
                      b_f, m1_couple_fun_t{ } );
                }
                this->sync_streams( mesh );
            }
            base_t::step( mesh );      //  the V - cycles
            if( closure )
                for( auto & d : * this )
                {
                    auto & dd( prx_T::dual( d ) );
                    auto b_f = prx_T::d( dd.d_mg( 0 ) );
                    f_of_e_fun_t fe{ };
                    fe.dt = dt;
                    fe.accum = f_accum;
                    dev.launch
                    ( fld_fun_k< decltype( b_f ),
                      f_of_e_fun_t >,
                      rsc_copy( b_f ), d.stream,
                      b_f, fe );
                }
            if( ! picard )
                break;
            if( picard_converged( mesh, i_pc ) )
                break;
        }
        return;
    };

    //  The M1 chi freeze of one block  (  the  rsc_copy
    // convention;  the sig freeze has already run ).
    template< class bdt_T >
    void launch_m1_freeze( device::base_t & dev,
        mesh::block::dual_t & d, bdt_T & b_f,
        const f_t & dt )
    {
        m1_freeze_fun_t mf{ };
        mf.dt_c2 = dt * dt * c_light * c_light;
        mf.chi.c_l = c_light;
        mf.t_mode = chi_mode;
        for( int a = 0; a < 3; ++ a )
            mf.ca[ a ] = chi_a[ a ];
        dev.launch
        ( fld_fun_k< bdt_T, m1_freeze_fun_t >,
          rsc_copy( b_f ), d.stream, b_f, mf );
        return;
    };

    //  Snapshot + ghost closure + freeze + coarse
    // restriction of one block.
    template< class dual_T, class bdt_T >
    void launch_freeze( device::base_t & dev,
        mesh::block::dual_t & d, dual_T & dd,
        bdt_T & b_f, const freeze_fun_t & fz,
        const rcoarse_fun_t & rc, const snap_fun_t & sn,
        const bool & picard, const f_t & dt )
    {
        auto & opt = static_cast< fld_operation_t & >
                     ( * this->p_opt );
        if( picard )
        {
            dev.launch
                ( fld_fun_k< bdt_T, snap_fun_t >,
                  rsc_copy( b_f ), d.stream, b_f, sn );
            dev.launch
                ( fld_fun_k< bdt_T, ghost_fun_t >,
                  rsc_copy( b_f ), d.stream, b_f,
                  ghost_fun_t{ opt.robin, f_inc,
                               d_unif } );
        }
        dev.launch
            ( fld_fun_k< bdt_T, freeze_fun_t >,
              rsc_copy( b_f ), d.stream, b_f, fz );
        //  The M1 chi freeze  (  BEFORE the coarse
        // restriction so the  chi  slots restrict ):
        if( closure )
            launch_m1_freeze( dev, d, b_f, dt );
        for( int l = -1; l >= 1 - dd.n_lvl(  ); -- l )
        {
            auto b_c = prx_T::d( dd.d_mg( l ) );
            dev.launch
                ( fld_fun2_k< decltype( b_c ),
                  rcoarse_fun_t >,
                  rsc_copy( b_c ), d.stream,
                  b_c, b_f, rc );
            b_f = b_c;
        }
        return;
    };

    //  Fill the level - 0  co_arr ghost slabs from the
    // neighbours  ( the SMR coefficient exchange;  the
    // same - level faces verbatim,  the cross - level
    // sides fine - reads - coarse / coarse - averages -
    // fine ).   Same - rank neighbours only  ( cross -
    // rank runs one block per rank in the current
    // isolation stage ).   fld - method.md § 8.
    void fill_co_ghost( mesh::mesh_t & mesh )
    {
        auto & dev = * this->p_dev;
        for( auto & d : * this )
        {
            const auto & reg_h( d.h(  ).reg );
            auto & dd( prx_T::dual( d ) );
            auto b_h = prx_T::d( dd.d_mg( 0 ) );
            for( const auto & nbp :
                     mesh.block( reg_h ).neighbors )
            {
                const auto & nb( nbp.first );
                if( nb.mode != 0 )
                    continue;
                const auto & rg( nbp.second );
                //  Skip only the TRUE self  ( no
                // offset );  a zero - offset neighbour
                // on a periodic axis is the block's own
                // wrap  ( the single - block seam )  --
                // keep it  ( the ghost takes the block's
                // own opposite side ).
                bool self( true );
                for( int b = 0; b < 3; ++ b )
                    self = self && ( nb[ b ] == 0 );
                if( self )
                {
                    bool wrap( false );
                    for( int b = 0; b < 3; ++ b )
                        wrap = wrap
                          || mesh.tree(  ).periodic[ b ];
                    if( ! wrap )
                        continue;
                }
                if( mesh.block( rg ).rank
                    != mesh.block( reg_h ).rank )
                    continue;
                int a( -1 ), side( -1 );
                for( int b = 0; b < 3; ++ b )
                {
                    if( nb[ b ] > 1 )
                    {   a = b;   side = 1; }
                    else if( nb[ b ] < 0 )
                    {   a = b;   side = 0; }
                }
                if( a < 0 )
                    continue;
                auto b_n( prx_T::d
                    ( prx_T::dual( this->data( rg ) )
                        .d_mg( 0 ) ) );
                launch_ghost_pair
                    ( dev, d, b_h, b_n, a, side,
                      rg.level - reg_h.level,
                      reg_h, rg );
            }
        }
        this->sync_streams( mesh );
        return;
    };

    //  One neighbour pair's coefficient ghost fill.
    template< class bdt_T >
    void launch_ghost_pair( device::base_t & dev,
        mesh::block::dual_t & d, const bdt_T & b_h,
        const bdt_T & b_n, const int & a,
        const int & side, const int & d_lvl,
        const mesh::region_logic_t & reg_h,
        const mesh::region_logic_t & reg_n )
    {
        const auto & geo_h( b_h.geo );
        const int t1( utils::mod3( a + 1 ) ),
                  t2( utils::mod3( a + 2 ) );
        const int n_fc( geo_h.n_ceff[ t1 ]
                        * geo_h.n_ceff[ t2 ] );
        const int n_th( std::min( n_fc, 256 ) );
        const dim3 n_bl( ( n_fc + n_th - 1 ) / n_th );
        type::idx_t org_h, org_n;
        for( int b = 0; b < 3; ++ b )
        {
            org_h[ b ] = reg_h.x[ b ];
            org_n[ b ] = reg_n.x[ b ];
        }
        if( d_lvl == 0 )
            dev.launch
                ( fld_rfill_k< bdt_T >,
                  std::make_tuple( n_bl, dim3( n_th ), 0 ),
                  d.stream, b_h, b_n, a, side );
        else if( d_lvl > 0 )
            //  I am the COARSE side.
            dev.launch
                ( fld_xfill_c_k< bdt_T >,
                  std::make_tuple( n_bl, dim3( n_th ), 0 ),
                  d.stream, b_h, b_n, a, side,
                  org_h, org_n );
        else
            //  I am the FINE side.
            dev.launch
                ( fld_xfill_k< bdt_T >,
                  std::make_tuple( n_bl, dim3( n_th ), 0 ),
                  d.stream, b_h, b_n, a, side,
                  org_h, org_n );
        return;
    };

    //  The Picard change measure  ( device reduce +
    // host compare against  r_tol ).
    bool picard_converged( mesh::mesh_t & mesh,
                           const int & i_pc )
    {
        auto & dev = * this->p_dev;
        dev.f_mset( p_chg_d,  0, sizeof( f_t ), 0 );
        dev.f_mset( p_fmax_d, 0, sizeof( f_t ), 0 );
        for( auto & d : * this )
        {
            auto b_f = prx_T::d
                ( prx_T::dual( d ).d_mg( 0 ) );
            dev.launch
                ( fld_fun_k< decltype( b_f ),
                  chg_fun_t >,
                  rsc_copy( b_f ), d.stream,
                  b_f, chg_fun_t{ p_chg_d, p_fmax_d } );
        }
        this->sync_streams( mesh );
        f_t chg_h( 0 ), fmax_h( 0 );
        dev.cp( chg_h,  * p_chg_d,  1 );
        dev.cp( fmax_h, * p_fmax_d, 1 );
        dev.sync_all_streams(  );
        const f_t rel
            ( chg_h / ( fmax_h + f_t( 1e-30 ) ) );
        if( to_print && mesh.p_com->is_root(  ) )
            std::cout << "FLD Picard " << i_pc
                      << " :  rel chg = " << rel
                      << std::endl;
        return rel < r_tol;
    };

    //  The pure - radiation isolation enrolls no dynamics
    // module,  so the driver owns the clock advance.
    void advance_clock( mesh::mesh_t & mesh )
    {
        auto & cyc( * mesh.p_cyc );
        cyc.redc_dt_start ( mesh );
        cyc.redc_dt_finish( mesh );
        return;
    };

    public:
    //  The domain - size cell count along an axis  ( for
    // the boundary - cell  dx  of the Robin source ).
    __host__ int n_ax( mesh::mesh_t & mesh,
                       const int & a ) const
    {
        const auto & p0
            ( * this->p_blk_local->cbegin(  ) );
        const auto & geo = prx_T::dual( * p0.second )
                            .d_mg( 0 ).geo;
        const auto n = geo.n_ceff[ a ];
        return n > 1 ? n : 1;
    };

    ////////// Launch resources ////////
    template< class bdt_T > __host__
    auto rsc_copy( const bdt_T & b_d ) const
    {
        const int n_cell = b_d.geo.n_ceff.prod(  );
        const int n_th   = std::min( n_cell, 256 );
        const dim3 n_bl  ( ( n_cell + n_th - 1 ) / n_th );
        return std::make_tuple( n_bl, dim3( n_th ), 0 );
    };
    template< class bdt_T > __host__
    auto rsc_in( const bdt_T & b_d ) const
    {
        int n_cell = 1;
        for( int a = 0; a < 3; ++ a )
            n_cell *= b_d.geo.n_ceff[ a ]
                      + 2 * b_d.mgd.fld.n_gh[ a ];
        const int n_th   = std::min( n_cell, 256 );
        const dim3 n_bl  ( ( n_cell + n_th - 1 ) / n_th );
        return std::make_tuple( n_bl, dim3( n_th ), 0 );
    };
};

};                          // namespace rad::fld
