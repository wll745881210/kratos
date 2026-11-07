#pragma once

#include "../../../utilities/types/crtp.h"
#include "../../../utilities/functions.h"
#include "../../../io/args/input.h"
#include "../../../device/device.h"
#include "../../../mesh/types/dat_3d.h"
#include "../../../mesh/geometry/region.h"
#include "../../../mesh/geometry/geometry.h"
#include "../../dynamics/hydro/types/block_data.h"
#include "../corrector/corrector.h"
#include "dat.h"

#include <memory>

namespace chemistry
{
class base_t;
};

namespace chemistry::solver
{
////////////////////////////////////////////////////////////
// Kernel

template< class sol_T, class bdt_T , class rls_T,
          class eos_T, class spc_T > __global__
void ker( const sol_T sol, const bdt_T bdt, const rls_T rls,
          const eos_T eos, const spc_T work )
{
    return sol( bdt, rls, eos, work );
}

////////////////////////////////////////////////////////////
// Global workspace -- Plan B for shared memory

struct work_glob_t
{
    ////////// Data //////////
    using wdt_t = mesh::dat_3d_t< char >;
    std::shared_ptr< device::base_t > p_dev_w;
    mesh::lex_map_t< mesh::region_logic_t, wdt_t > bmap;

    ////////// Functions //////////
    void setup( const mesh::region_t &, const mesh::geo_t &,
                const int & );
    void free (  );
    
    static wdt_t arg // Null-safe kernel argument
    ( work_glob_t * p, const mesh::region_logic_t & reg )
    {
        return p != nullptr ? p->bmap[ reg ] : wdt_t{   };
    };
};

////////////////////////////////////////////////////////////
// Numerical integrator for thermochemistry

struct base_t
{
    ////////// Data //////////
    bool                      cache_rate;
    int                         iter_max;
    //  [chemistry] dbg_iter:  cells using >= dbg_iter
    //  internal steps printf their count + location + T at
    //  the end of the solve ( the stiff - cell forensics;
    //  0 = off ).
    int                        dbg_iter;
    int                             n_th;
    int                              dim; // Energy included
    type::float_t                   rtol;
    type::float_t                   atol;
    type::float_t                   ytol;    
    type::float_t             etol_ratio;
    work_glob_t            * p_work_glob;

    //////////  Host-side Interface //////////
    __host__ virtual void  init
    ( const  input & args, :: chemistry::base_t & mod );
    __host__ virtual void  free( device::base_t & dev );
    __host__ virtual std::tuple< dim3, dim3, int >
    resource ( const mesh::block::dual_t &  dual,
               const int   & size_sh = 0 ) const;
    __host__ virtual int size_sh(  ) const { return 0; };
};

template< class     cor_T = corrector  :: base_t  ,
          class     dat_T = dat_t< type::float_t >,
          class derived_T = crtp ::      dummy_t >
struct default_t : base_t
{
    ////////// Types //////////
    __crtp_def_self__( default_t, derived_T, cor_T, dat_T );
    enum kind_t { kr, ross };

    ////////// Data //////////
    kind_t            kind;
    cor_T             corr;
    type::float_t expl_lim;     // Explicit limit

    ////////// Host-side Interface //////////
    __host__ virtual void  init
    ( const  input & args, :: chemistry::base_t & mod )
    {
        const auto kind = args.get< std::string >
                 ( "chemistry", "solver",  "kr" );
        if( kind. find( "ross" )   != std::string::npos )
            this->kind = ross;
        else if( kind.find( "kr" ) != std::string::npos )
            this->kind =   kr;
        else
            throw std::runtime_error
               ( "Cannot identify ODE solver " + kind );
        expl_lim = args.get< type::float_t >
            ( "chemistry", "explicit_lim", 1e2 );
        base_t    ::init( args, mod );
        return corr.init( args, mod );
    };

    template < class prx_T , class mod_T ,
               class rls_T , class eos_T >  __host__ void
    integrate( mod_T & mod , mesh::block::dual_t & d ,
               const rls_T & rls,  const eos_T & eos )
    {
        if( iter_max < 1 )
            return;

        const auto rsc = resource( d, rls.size_sh(  )
                                    + eos.size_sh(  ) );
        const auto & self( get_self(  ) );
        const auto & bdt ( prx_T::d( d.d (  ) ) );
        const auto & work( work_glob_t::arg( p_work_glob,
                                             bdt.reg ) );
        rls.prep_host    ( * this, bdt, eos ) ;
        return mod.p_dev->launch( ker, rsc, d.stream, self,
                                  bdt, rls, eos, work );
    };

    __host__ virtual int size_sh(  ) const
    {
        return dat_T::size_sh( dim, n_th, cache_rate );
    };

    ////////// Device-side Functions //////////
    template < class rls_T, class bdt_T, class eos_T >
    __device__ __forceinline__ bool solve
    ( dat_T       & d, const rls_T & r ,
      const bdt_T & b, const eos_T & e ) const
    {
        if( kind == ross )
            return solve_ross( d, r, b, e );
        return solve_kr( d, r, b, e );
    };

    template < class rls_T, class bdt_T, class eos_T >
    __device__ bool solve_kr
    ( dat_T       & d, const rls_T & r ,
      const bdt_T & b, const eos_T & e ) const;

    template < class rls_T, class bdt_T, class eos_T >
    __device__ bool solve_ross
    ( dat_T       & d, const rls_T & r ,
      const bdt_T & b, const eos_T & e ) const;

    template< class F, class G > __device__ __forceinline__
    F pow_( const F & a, const G & b ) const
    {
        return a < 1e-32f ? 0 : exp2f( b * log2f( a ) );
    };

    __device__ __forceinline__
    type::float_t get_atol( dat_T & d ) const
    {
        return __min1( atol, ytol / d.rho_cgs );
    };

    template < class d_T >
    __device__  bool is_lu_failed( d_T & d ) const
    {
        using     f_t = typename d_T::f_t;
        const int i_th( threadIdx.x );
        __shared__ int lu_bad;
        if( i_th  == 0 )
            lu_bad = 0 ;
        d.lu_fail  = false;
        __syncthreads(  );
        for( int i = i_th; i < d.dim; i += blockDim.x )
        {
            const f_t p( d.a( i, i ) );
            if( __isnan( p ) || p == 0 ||
                __abs1 ( p ) > 1e30f )
                lu_bad = 1;
        }
        __syncthreads(  );
        if( lu_bad > 0 )
        {
            d.lu_fail = true;
            for( int i = i_th; i < d.dim; i += blockDim.x )
                d.err[ i ] = 1e6f; // forces stp_ctrl reject
            stp_ctrl ( d ) ;
            return true;
        }
        return false;
    };
    
    __device__ bool stp_ctrl( dat_T & d ) const
    {
        using f_t = typename dat_T::f_t;
        __syncthreads(  );
        
        f_t errm( 0 );
        for( int i = 0; i < d.dim; ++ i )
            errm += d.err[ i ];
        errm = __sqrt1( errm / d.dim );
        
        // utils::block_reduce
        //     < f_t, utils::sum >( d.err, d.dim );
        // const auto errm = __sqrt1 ( d.err[ 0 ] / d.dim );
        
        static constexpr  f_t safe( 0.9f ),
               fac1( 5 ), fac2 ( 1.f / 6 );
        if( __isnan( errm ) || __isinf( errm ) )
            errm = 1e2;

        auto fac = __max1( fac2, __min1
              ( fac1, pow_( errm, 0.25f ) / safe ) );
        auto h_new = d.h / fac;
        if( errm < 1 )
        {
            if( ! d.first_step )
            {
                auto fac_pred = ( d.h_old / d.h ) *
                    pow_( errm * errm / d.err_old,
                          0.25f )  / safe ;
                fac_pred = __max1( fac2, __min1
                                   ( fac1, fac_pred ) );
                fac      = __max1( fac, fac_pred );
                h_new    = d.h   / fac;
            }
            d.first_step = false;
            d.h_old      =   d.h;
            d.err_old    = __max1( 0.01f, errm );
            if( d.rejected )
                h_new = ( d.h > 0 ? __min1( h_new, d.h )
                                  : __max1( h_new, d.h ) );
            d.rejected = false;
            d.t       -=   d.h;
            d.h        = h_new;
            return        true;
        }
        d.rejected =  true;
        d.h        = h_new;
        return       false;
    };

    template< class bdt_T, class rls_T ,
              class eos_T, class spc_T >
    __forceinline__ __device__   void operator(  )
    ( const bdt_T & bdt, const rls_T &  rls ,
      const eos_T & eos, const spc_T & work ) const
    {
        __dyn_shared__( char, var_sh );
        char * base( var_sh );
        if( work )
        {
            type:: idx_t idx_c;
            idx_c[ 0 ] = blockIdx.x;
            idx_c[ 1 ] = blockIdx.y;
            idx_c[ 2 ] = blockIdx.z;            
            base = ( char * ) work.at( idx_c );
        }
        dat_T  d      ( dim,  base, this->cache_rate );
        d  .load      ( bdt,     eos );
        rls.prep_dev  ( d, bdt,  eos );
         __syncthreads(  );
        for( d.iter = iter_max; d.iter > 0; )
            if( get_self(  ).solve( d, rls, bdt, eos ) )
                break;
        if( dbg_iter > 0 && threadIdx.x == 0
            && iter_max - d.iter >= dbg_iter )
            printf( "CHEMITER %d at %d %d %d  T %e\n",
                    iter_max - d.iter,
                    int( blockIdx.x ), int( blockIdx.y ),
                    int( blockIdx.z ), double( d.T ) );
        corr.template
             fix< type::float2_t >( d, bdt, eos, true );
        return d.save( bdt, eos );
    };

    __device__ bool try_explicit( dat_T & d ) const
    {
        if( expl_lim <= 0 )
            return false;
        const int i_th( threadIdx.x );        
        __shared__ int do_explicit;
        if( threadIdx.x == 0 )
            do_explicit =  0 ;
        for( int i = i_th; i < d.dim; i += blockDim.x )
        {
            const auto f_e = __max1
                ( __abs1( d.f[ i ] ), type::float_min );
            if( __abs1( d.y[ i ] ) > f_e * expl_lim * d.h )
                atomicAdd( & do_explicit, 1 );
        }
        __syncthreads(  );
        if( do_explicit == d.dim )
        {
            -- d.iter;
            for( int i = i_th; i < d.dim; i += blockDim.x )
                d.y[ i ] += d.f[ i ] * d.h;
            d.t -=   d.h;
            d.h  = __min1( 1.5f * d.h, d.t );
            return true;
        }        
        return false;
    };

    template< class bdt_T, class eos_T, class rls_T >
    __device__ void fallback
    ( dat_T & d, const bdt_T & bdt,
      const eos_T & eos, const rls_T & rls ) const
    {
        const int i_th( threadIdx.x );
        -- d.iter;
        for( int i = i_th; i < d.dim; i += blockDim.x )
            d.y[ i ] = d.y_old[ i ];
        __syncthreads(  );
        rls.rate( d, bdt, eos );

        d.h = 1e0f;
        for( int i = 0; i < d.dim; ++ i )
        {
            const auto h_tst = __abs1( d.y_old [ i ] )
                / __max1( __abs1( d.f[ i ] ), 1e-30f );
            d.h = __min1( d.h, h_tst );
        }
        d.h *= 0.5f;
        d.t -=  d.h;        
        for( int i = i_th; i < d.dim; i += blockDim.x )
            d.y[ i ] = d.f[ i ]  * d.h + d.y_old[ i ];
        return;
    };
};

////////////////////////////////////////////////////////////
// Rosenbrock Kaps-Rentrop

template < class cor_T, class dat_T, class derived_T >
template < class rls_T, class bdt_T, class     eos_T >
__device__ bool default_t< cor_T, dat_T,   derived_T >::
solve_kr( dat_T       &   d, const  rls_T & rls ,
          const bdt_T & bdt, const  eos_T & eos ) const
{
    using     f_t = typename dat_T::f_t;
    constexpr f_t a[ 3 ] = { 2.f,    1.92f, 0.24f };
    constexpr f_t c[ 6 ] =
    { -8.f, 14.88f, 2.4f, -0.896f, -0.432f, -0.4f };
    constexpr f_t b[ 4 ] =
    { 2.1111111f, 0.5f,  0.23148148f,  1.1574074f };
    constexpr f_t e[ 4 ] =
    { 0.3148148f, 0.1944444444 , 0.f,  1.1574074f };
    constexpr f_t safety( 0.9f );
    constexpr f_t h_min         ( type::float_min );


    auto & g0    = d.x[ 0 ];
    auto & g1    = d.x[ 1 ];
    auto & g2    = d.x[ 2 ];
    auto & f_old = d.x[ 3 ];
    auto & g3    = d.     f;
    auto & err   = d.     f;

    const int i_th( threadIdx.x );

    rls.jac( d, bdt, eos );
    __syncthreads(  );
    d.h  = __min1( d.h, d.t );

    if( try_explicit( d ) )
        return d.t <= d.h;
    for( int i = i_th; i < d.dim; i += blockDim.x )
    {
        d.y_old[ i ] = d.y[ i ];
        f_old  [ i ] = d.f[ i ];
    }
    __syncthreads(  );

    const auto atol( get_self(  ).get_atol( d ) );    
    for( -- d.iter; d.iter > 0; -- d.iter )
    {
        if constexpr( std::is_same_v< f_t, float > )
            if( d.h < h_min || isnan( d.h ) )
                break;

        const f_t hinv( 1 / d.h );
        for( int i = i_th; i < d.dim; i += blockDim.x )
        {
            for( int j = 0; j < d. dim; ++ j )
                d.a( i, j ) = - d.J( i,    j );
            d.a ( i, i )   += 2 * hinv;
        }
        d.a.decompose(  );
        if( is_lu_failed( d ) )
            continue;

        for( int i = i_th; i < d.dim; i += blockDim.x )
            g0[ i ] = f_old[ i ];
        d.a.back_sub( g0 );

        for( int i = i_th; i < d.dim; i += blockDim.x )
             d.y[ i ] = d.y_old[ i ] + a[ 0 ] * g0[ i ];
        rls.rate( d, bdt, eos );
        for( int i = i_th; i < d.dim; i += blockDim.x )
            g1[ i ] = d.f[ i ] + c[ 0 ] * g0[ i ] * hinv;
        d.a.back_sub( g1 );

        for( int i = i_th; i < d.dim; i += blockDim. x )
             d.y[ i ] = d. y_old[ i ] + a[ 1 ] * g0[ i ]
                                      + a[ 2 ] * g1[ i ];
        rls.rate( d, bdt, eos );
        for( int i = i_th; i < d .dim; i += blockDim.x )
            g2[ i ] = d.f[ i ] +  hinv *
                    ( c[ 1 ] * g0[ i ] + c[ 2 ] * g1[ i ] );
        d.a.back_sub( g2 );
        for( int i = i_th; i < d .dim; i += blockDim.x )
            g3[ i ] = d.f[ i ] + hinv *( c[ 3 ] * g0[ i ]
                    + c[ 4 ] * g1[ i ] + c[ 5 ] * g2[ i ] );
        d.a.back_sub( g3 ) ;

        for( int i = i_th; i < d.dim; i += blockDim.x )
        {   // v--- prep. for the corr. step, diff.  n only
            d.y[ i ] = d.    y_old[ i ]
                     + b[ 0 ] * g0[ i ] + b [ 1 ] * g1[ i ]
                     + b[ 2 ] * g2[ i ] + b [ 3 ] * g3[ i ];
            err[ i ] = e[ 0 ] * g0[ i ] + e [ 1 ] * g1[ i ]
                     + e[ 2 ] * g2[ i ] + e [ 3 ] * g3[ i ];
            err[ i ] = __abs1( err[ i ] )
                     / ( i == d.i_en ? etol_ratio : 1 ) ;
            if( __isnan( err[ i ] ) || err[ i ] > 1e30f )
            {
                d.err[ i ] = 1e20f;
                continue;
            }
            const auto y_ref = __max1
            ( __abs1 ( d.y[ i ] ), __abs1( d.y_old[ i ] ) );
            err  [ i ] /= atol + rtol * y_ref;
            d.err[ i ]  = err[ i ] * err[ i ];
        }
        if( stp_ctrl( d ) )
            break;
    }
    __syncthreads(  );
    if( d.h < h_min || isnan( d.h ) )
    {
        d.lu_fail = true;
        fallback( d, bdt, eos, rls );
    } // Fallback to stupid Euler method when LU fails
    corr.template fix< f_t > ( d, bdt, eos, false );
    if( d.t  < 1e-5f * d.t0 )
    {
        d.t    =  0;
        return true;        // Stop ( errm > 1e4 : failed )
    }
    return  false;          // Succeeded, but continue
};                          // Rosenbrock Kaps-Rentrop

////////////////////////////////////////////////////////////
// Rosenbrock Stiffly Stable method (NR ver. 3rd, pp. 936)

template < class cor_T , class dat_T, class derived_T >
template < class rls_T , class bdt_T, class     eos_T >
__device__ bool  default_t< cor_T, dat_T,   derived_T >::
solve_ross( dat_T       &   d, const  rls_T & rls ,
            const bdt_T & bdt, const  eos_T & eos ) const
{
    using     f_t = typename dat_T::f_t;
    constexpr f_t a21(  0.1544000000000000e+01f );
    constexpr f_t a31(  0.9466785280815826e+00f );
    constexpr f_t a32(  0.2557011698983284e+00f );
    constexpr f_t a41(  0.3314825187068521e+01f );
    constexpr f_t a42(  0.2896124015972201e+01f );
    constexpr f_t a43(  0.9986419139977817e+00f );
    constexpr f_t a51(  0.1221224509226641e+01f );
    constexpr f_t a52(  0.6019134481288629e+01f );
    constexpr f_t a53(  0.1253708332932087e+02f );
    constexpr f_t a54( -0.6878860361058950e+00f );
    constexpr f_t c21( -0.5668800000000000e+01f );
    constexpr f_t c31( -0.2430093356833875e+01f );
    constexpr f_t c32( -0.2063599157091915e+00f );
    constexpr f_t c41( -0.1073529058151375e+00f );
    constexpr f_t c42( -0.9594562251023355e+01f );
    constexpr f_t c43( -0.2047028614809616e+02f );
    constexpr f_t c51(  0.7496443313967647e+01f );
    constexpr f_t c52( -0.1024680431464352e+02f );
    constexpr f_t c53( -0.3399990352819905e+02f );
    constexpr f_t c54(  0.1170890893206160e+02f );
    constexpr f_t c61(  0.8083246795921522e+01f );
    constexpr f_t c62( -0.7981132988064893e+01f );
    constexpr f_t c63( -0.3152159432874371e+02f );
    constexpr f_t c64(  0.1631930543123136e+02f );
    constexpr f_t c65( -0.6058818238834054e+01f );
    constexpr f_t gam(  0.2500000000000000e+00f );
    constexpr f_t d21(  0.1012623508344586e+02f );
    constexpr f_t d22( -0.7487995877610167e+01f );
    constexpr f_t d23( -0.3480091861555747e+02f );
    constexpr f_t d24( -0.7992771707568823e+01f );
    constexpr f_t d25(  0.1025137723295662e+01f );
    constexpr f_t d31( -0.6762803392801253e+00f );
    constexpr f_t d32(  0.6087714651680015e+01f );
    constexpr f_t d33(  0.1643084320892478e+02f );
    constexpr f_t d34(  0.2476722511418386e+02f );
    constexpr f_t d35( -0.6594389125716872e+01f );
    constexpr f_t h_min       ( type::float_min );   

    auto & g0    = d.x[ 0 ];
    auto & g1    = d.x[ 1 ];
    auto & g2    = d.x[ 2 ];
    auto & g3    = d.x[ 3 ];
    auto & g4    = d.x[ 4 ];
    auto & g5    = d.x[ 5 ];
    auto & f_old = d.x[ 6 ];
    const int i_th( threadIdx.x );

    rls.jac( d, bdt, eos );
    __syncthreads(  );
    d.h = __min1( d.h, d.t );
    if( try_explicit( d ) )
        return d.t <= d.h;
    
    for( int i = i_th; i < d.dim; i += blockDim.x )
    {
        d.y_old[ i ] = d.y[ i ];
        f_old  [ i ] = d.f[ i ];
    }
    __syncthreads(  );

    const auto atol( get_self(  ).get_atol( d ) );
    for( -- d.iter; d.iter > 0; -- d.iter )
    {
        if constexpr( std::is_same_v< f_t, float > )
            if( d.h < h_min || isnan( d.h ) )
                break;
        
        const f_t hinv( 1 / d.h );
        for( int i = i_th; i < d.dim; i += blockDim.x )
        {
            for( int j = 0; j < d. dim; ++ j )
                d.a( i, j ) = - d.J( i,    j );
            d.a ( i, i )   += 1 / gam * hinv;
        }
        d.a.decompose(  );
        if( is_lu_failed( d ) )
            continue;

        for( int i = i_th; i < d.dim; i += blockDim.x )
            g0[ i ] = f_old[ i ];
        d.a.back_sub( g0 );


        for( int i = i_th; i < d.dim; i += blockDim.x )
            d.y[ i ] = d.y_old[ i ] + a21 * g0[ i ];
        rls.rate( d, bdt, eos );
        for( int i = i_th; i < d.dim; i += blockDim.x )
            g1[ i ] = d.f[ i ] + c21 * g0[ i ] * hinv ;
        d.a.back_sub( g1 );

        for( int i = i_th; i < d.dim; i += blockDim.x )
            d.y[ i ] = d. y_old[ i ]
                     + a31 * g0[ i ] + a32 * g1[ i ];
        rls.rate( d, bdt, eos );
        for( int i = i_th; i < d.dim; i += blockDim.x )
            g2[ i ] =  d.f[ i ] + hinv *
                ( c31 * g0[ i ] + c32  * g1[ i ] );
        d.a.back_sub  ( g2 );


        for( int i = i_th; i < d.dim; i += blockDim.x )
            d.y[ i ] = d. y_old[ i ] + a41 * g0[ i ]
                     + a42 * g1[ i ] + a43 * g2[ i ];
        rls.rate( d, bdt, eos );
        for( int i = i_th; i < d.dim; i += blockDim.x )
            g3[ i ] =  d.f[ i ] + hinv *
                ( c41 * g0[ i ] + c42  * g1[ i ] +
                  c43 * g2[ i ] ) ;
        d.a.back_sub  ( g3 );


        for( int i = i_th; i < d.dim; i += blockDim.x )
            d.y[ i ] = d. y_old[ i ] + a51 * g0[ i ]
                     + a52 * g1[ i ] + a53 * g2[ i ]
                     + a54 * g3[ i ] ;
        rls.rate( d, bdt, eos );
        for( int i = i_th; i < d.dim; i += blockDim.x )
            g4[ i ] =  d.f[ i ] + hinv *
                ( c51 * g0[ i ] + c52  * g1[ i ] +
                  c53 * g2[ i ] + c54  * g3[ i ] );
        d.a.back_sub  ( g4 );


        for( int i = i_th; i < d.dim; i += blockDim.x )
            d.y[ i ] += g4[ i ];
        rls.rate( d, bdt, eos );
        for( int i = i_th; i < d.dim; i += blockDim.x )
            g5[ i ] =  d.f[ i ] + hinv *
                ( c61 * g0[ i ] + c62  * g1[ i ] +
                  c63 * g2[ i ] + c64  * g3[ i ] +
                  c65 * g4[ i ] ) ;
        d.a.back_sub  ( g5 );

        auto & err =   g5;
        for( int i = i_th; i < d.dim; i += blockDim.x )
        {
            d.y[ i ] += g5 [ i ];
            err[ i ] /= atol + rtol * __max1
            ( __abs1 ( d.y[ i ] ), __abs1( d.y_old[ i ] ) );
            err[ i ] = __abs1( err[ i ] )
                     / ( i == d.i_en ? etol_ratio : 1 ) ;
            d.err[ i ] = err[ i ] * err[ i ];
        }
        if( stp_ctrl( d ) )
            break;
    }
    if( d.h < h_min || isnan( d.h ) )
    {
        d.lu_fail = true;
        fallback( d, bdt, eos, rls );
    } // Fallback to stupid Euler method when LU fails
    corr.template fix< f_t >( d, bdt, eos, false );
    if( d.t  < 1e-5f * d.t0 )
    {
        d.t    =  0;
        return true;
    }
    return false;
};                          // Rosenbrock Stiffly Stable

};                          // namespace chemistry::solver
