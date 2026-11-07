#pragma once

#include "../../../utilities/mapping/reduction.h"
#include "../../../utilities/phys/constants.h"
#include "../../../types.h"
#include "../../indices.h"
#include "lu.h"

namespace chemistry::solver
{
////////////////////////////////////////////////////////////
// Data holder for the reactions

template < class f_T, int n_aux = 7 >
struct dat_t
{
    ////////// Types //////////
    using    f_t        = f_T;
    using   f2_t        = type::float2_t;    
    using this_t = dat_t< f_T, n_aux >;

    ////////// Step-size Ctrl //////////
    bool  first_step;
    bool    rejected;
    bool     lu_fail;
    f_t      err_old;
    f_t        h_old;

    ////////// Data //////////
    int         i_en;       // Idx for energy
    int         iter;       // # of iteration
    int          dim;       // Dimensions
    f_t           t0;       // Time span
    f_t            t;       // Time remaining
    f_t            h;       // Step length
    f_t            T;       // Temperature
    f_t            e;       // Internal energy density
    f_t      rho_cgs;       // Mass density
    f_t        * err;       // Error
    f_t        *  pj;       // Jacacobian (ptr)
    f_t        *   f;       // Reaction rates
    f_t        *   y;       // "Mole abund" := n * m_A / rho
    f_t        * f_c;       // Cache for reaction rates    
    f_t *      y_old;       // "Old" y before one step
    f_t * x[ n_aux ];       // Aux arrays
    lu_t< f_t >    a;       // The "principal" matrix
    type::idx_t  idx;       // Index of the current cell

    f2_t *     y_fid;       // "Old" y before all steps    
    f2_t *       y_2;       // High-prec y

    ////////// Host-side functions //////////
    __host__ static int size_sh
    ( const int  & n, const int & n_th,
      const bool & cache_f )
    {
        const auto res = decltype( a ) ::size( n )
            + sizeof ( * this_t::  f ) * n * ( 5 + n_aux )
            + sizeof ( * this_t:: pj ) * n * n
            + sizeof ( * this_t::f_c ) * n * n_th * cache_f;
        return utils::align( res,  8 ) ;
    };

    ////////// Functions //////////
    __device__ dat_t(  ) = default;

    __device__ dat_t( const int  & dim,  char * p,
                      const bool & cache_f )
    {
        this->setup( dim, p, cache_f );
        return;
    };

    __device__ void * setup( const int  & dim,  char * p,
                             const bool & cache_f )
    {
        this -> dim = dim ;
        i_en  = this->dim - 1;
        pj    = decltype( pj    )( p );
        f     = decltype( f     )( pj  + dim  * dim );
        y_old = decltype( y_old )( f          + dim );
        y     = decltype( y     )( y_old      + dim );
        y_fid = decltype( y_fid )( y          + dim );
        f_c   = decltype( f_c   )( y_fid      + dim );
        f_t   * q     = ( f_t * )( f_c + cache_f  *
                                   dim * blockDim.x );
        if( ! cache_f )
            f_c = nullptr;
        for( int i = 0; i < n_aux; ++ i )
            x[ i ] =  q + i * dim;
        y_2 = decltype( y_2 )( x[ 0 ] );
        
        void * ret = a.setup
             ( ( char * ) ( q + n_aux * dim ), dim );
        a.set_aux( x[ 0 ], x[ 2 ] );
        first_step =   true;
        rejected   =  false;
        lu_fail    =  false;
        err        = x[ 0 ];
        T          =     -1;
        idx[ 0 ]   = blockIdx.x;
        idx[ 1 ]   = blockIdx.y;
        idx[ 2 ]   = blockIdx.z;
        return p + utils::align
             ( int( size_t( ret ) - size_t( p ) ), 8 );
    };

    __forceinline__ __device__ auto & J
    ( const int & i, const int & j )
    {
        return pj[ int( i ) * int( dim ) + j ];
    };
    __forceinline__ __device__
    void f_cache( const int & i, const f_t & r )
    {
        if( f_c != nullptr )
            f_c[ threadIdx.x * dim + i ] += r;
        else
            atomicAdd( f + i, r );
        return;
    };

    __forceinline__ __device__ void f_reduce(  )
    {
        __syncthreads(  );
        if( f_c == nullptr )
            return;
        for( int i = threadIdx.x; i < dim; i += blockDim.x )
        {
            for( int j = 1; j < blockDim.x; ++ j )
                f_c[ i ] += f_c[ j * dim + i ];
            f[ i ] = f_c[ i ];
        }
        return;
    };

    __forceinline__ __device__ void  f_memset_mt(  )
    {
        __syncthreads(  );
        const int m = dim
            * ( f_c == nullptr ? 1 : blockDim.x );
        for ( int i = threadIdx.x; i < m; i += blockDim.x )
            i [ f_c == nullptr ? f : f_c ] = 0 ;
        return;
    };
    
    __forceinline__ __device__ void jf_memset_mt(  )
    {
        __syncthreads  (     ) ;
        int m = ( int )( dim ) * ( int )  ( dim + 1 );
        for ( int i = threadIdx.x; i < m; i += blockDim.x )
            pj [ i ] = 0;
        return f_memset_mt(  );
    };
    
    template < class bdt_T, class eos_T >  __device__
    void load( const bdt_T & bdt, const eos_T & eos )
    {
        auto * u_src = bdt.hyd.u.at( idx );
        f2_t rho_f2( u_src[ idx::rho ] );
        f_t  rho( rho_f2 );
        if( __isnan( rho ) || __isinf( rho ) ||
            rho < eos.d_floor )
        {
            rho    = eos.d_floor;
            rho_f2 = eos.d_floor;
            if( threadIdx.x ==  0 )
                u_src[ idx::rho ] = rho;
        }
        rho_cgs = rho * eos.unit.rho0;
        __shared__ f_t e_cgs;
        if( threadIdx.x == 0 )
        {
            e_cgs = eos.e_g_cgs ( u_src );
            y [ i_en ] =  e_cgs / rho_cgs;
        }   // Specific internal energy ( in erg / g )
        __syncthreads(  );
        this->t  = bdt.dt_0 * eos.unit.t0;
        this->t0 = this->t;
        this->h  = this->t;
        u_src   += eos.n_hyd;

        for( int l = threadIdx.x; l < dim; l += blockDim.x )
        {
             if( l != i_en )
             {
                 y_fid[ l ] = u_src[ l ] / rho_f2;
                 if( y_fid[ l ] < 0 || isnan( y_fid[ l ] )
                                    || isinf( y_fid[ l ] ) )
                     y_fid[ l ]  = 0;
                 y_2[ l ] = y_fid[ l ];
                 y  [ l ] = y_fid[ l ];
             } // v--- *** SPECITAL NOTES BELOW ***
        }   // Y == n_i * ma / rho, u_src == n_i * ma / rho0
        __syncthreads(  );
        this->e =  e_cgs / eos.unit.ene0;
        return; // Y (and y, y_old) for molar abundance
    };

    template < class bdt_T, class eos_T >  __device__
    void save( const bdt_T & bdt, const eos_T & eos )
    {
        __syncthreads(  );
        f2_t * g = ( f2_t * )( x[ 2 ] );
        for( int l = threadIdx.x; l < dim; l += blockDim.x )
             if( l != i_en )
             {
                 y_2[ l ]  = utils::max( y_2[ l ], 0 );
                 g  [ l ]  = y_2[ l ] * eos.mcgs ( l );
             }
        utils::block_reduce
             < f2_t, utils::sum > ( g, dim - 1 ) ;
        if( __isnan( g[ 0 ]    ) || __isinf( g[ 0 ]  )  ||
            __isnan( y[ i_en ] ) || __isinf( y[ i_en ] ) )
            return;
        auto * u_tgt  = bdt.hyd.u.at( idx );
        const auto rho( u_tgt[ idx::rho ] );
        if( threadIdx.x == 0 )
        {
            bdt.T_hyd( idx ) = this->T;
            const auto de = -e + __max1
                ( y[ i_en ] * rho_cgs / eos.unit.ene0, 0 );
            if( ! ( __isnan( de ) || __isinf( de ) ) )
                u_tgt[ idx::ene ] += de;
        }
        u_tgt += eos.n_hyd;
        f2_t n_conv( rho * phys::cgs::ma / g[ 0 ] );
        __syncthreads(  );
        for( int l = threadIdx.x; l < dim; l += blockDim.x )
             if( l != i_en )
                 u_tgt[ l ] = y_2[ l ] * n_conv;
        return;
    };
};
};                          // namespace chemistry::solver
