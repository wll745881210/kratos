#pragma once

////////////////////////////////////////////////////////////
//  Face-centered Laplacian implicit solver for the magnetic
//  diffusion
//
//      ( I - dt eta nabla^2 ) B_new = B_old ,
//
//  per-component ( the Laplacian form;  SPD ).
//  This is the foundation and the smoother for the
//  face-centered multigrid.
//
//  Style:  the operations live in the implicit_t object
//  ( device member functions,  derivable / overridable );
//  the __global__ kernels are thin forwarders.   The
//  compute paths use type::float_t ( mixed precision:
//  FP32 by default,  FP64 via PRECISION = 2 ).
//
//  Boundary:  a periodic ghost wrap suffices for the
//  single-block validation here;  the multigrid hierarchy
//  will use its own keeper ( act_bnd_l ) for the general
//  case.

#include "../../../../../../src/mesh/mesh.h"
#include "../../../../../../src/utilities/types/crtp.h"
#include "../../../../../../src/utilities/mapping/reduction.h"
#include "../../../../../../src/utilities/functions.h"
#include "../../../../../../src/modules/dynamics/mhd/types/block_data.h"
#include <utility>

namespace mhd_ee::ni
{
using type:: float_t;


////////////////////////////////////////////////////////////
//  Thin kernels ( forward declarations;  definitions follow
//  the object ).

template< class bdt_T, class imp_T, bool TO_X >
__global__ void ni_imp_cp_k( const bdt_T, const imp_T & );
template< class bdt_T, class imp_T, class eta_T >
__global__ void ni_gs_k
( const bdt_T, const imp_T &, const eta_T &, const int );
template< class bdt_T, class imp_T, auto V >
__global__ void ni_imp_wrap_k( const bdt_T, const imp_T & );
template< class bdt_T, class imp_T, class eta_T >
__global__ void ni_imp_res_k
( const bdt_T, const imp_T &, const eta_T &, float_t * );

////////////////////////////////////////////////////////////
//  The implicit-diffusion operator object

template< class derived_T = crtp::dummy_t >
struct implicit_t
{
    __crtp_def_self__( implicit_t, derived_T );

    ////////// Device-side operations //////////

    //  Face diffusivity:  average of the two cells adjacent
    //  to face ( i, a ).
    template< class bdt_T, class idx_T, class eta_T >
    __device__ __forceinline__ float_t eta_face
    ( const bdt_T & b_d, const idx_T & i, const int & a,
      const eta_T & f_eta )                       const
    {
        float_t  eo0, eh0, ea0, eo1, eh1, ea1;
        auto       k ( i );  -- k[ a ];
        f_eta.eval( b_d, i, eo0, eh0, ea0 );
        f_eta.eval( b_d, k, eo1, eh1, ea1 );
        return float_t( 0.5 ) * float_t( eo0 + ea0
                                       + eo1 + ea1 );
    };

    //  Laplacian of a face array v at face ( i, a ):
    //  sum_d ( v[i+e_d] - 2 v[i] + v[i-e_d] ) / dx_d^2.
    template< class bdt_T, class idx_T, class arr_T >
    __device__ __forceinline__ float_t laplacian
    ( const bdt_T & b_d, const idx_T & i, const int & a,
      const arr_T & v )                           const
    {
        const auto & geo = b_d.geo;
        float_t      acc ( 0 );
        for( int d = 0; d < geo.n_dim; ++ d )
        {
            const float_t h ( geo.dx_f( d, i ) );
            auto kp ( i );  ++ kp[ d ];
            auto km ( i );  -- km[ d ];
            acc += (   v.at( kp )[ a ]
                     - float_t( 2 ) * v.at( i )[ a ]
                     + v.at( km )[ a ] ) / ( h * h );
        }
        return acc;
    };

    //  Whether ( i, a ) is a valid ( owned ) face of
    //  component a:  i[a] in [0, n_ceff[a]],  tangential
    //  i[d] in [0, n_ceff[d]).
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ bool valid_face
    ( const bdt_T & b_d, const idx_T & i,
      const int & a ) const
    {
        const auto & geo = b_d.geo;
        if( a >= geo.n_dim || i[ a ] > geo.n_ceff[ a ] )
            return false;
        for( int d = 0; d < geo.n_dim; ++ d )
            if( d != a && i[ d ] >= geo.n_ceff[ d ] )
                return false;
        return true;
    };

    //  One red-black Gauss-Seidel sweep ( stg = 0 / 1 ).
    //  The unknown x is cg_x;  the right-hand side is bf
    //  slice 0.
    template< class bdt_T, class eta_T >
    __device__ void gs_sweep
    ( const bdt_T & b_d, const int & stg,
      const eta_T & f_eta )
      const
    {
        const auto & geo  = b_d.geo;
        const int    n_dm = geo.n_dim;
        const float_t dtc = b_d.dt;
        auto       & x    = b_d.fld.cg_x;
        const auto & b    = b_d.fld.bf;
        type::idx_t  i    = { 0, int( blockIdx.x ),
                                   int( blockIdx.y ) };
        const int n_feff  = blockDim.x
            + ( blockIdx.z + 1 == gridDim.z ? 1 : 0 );
        for( int j = threadIdx.x; j < n_feff;
             j += blockDim.x )
        {
            i[ 0 ] = blockDim.x * blockIdx.z + j;
            for( int a = 0; a < n_dm; ++ a )
            {
                if( ! valid_face( b_d, i, a ) )
                    continue;
                if( ( ( i[ 0 ] + i[ 1 ] + i[ 2 ] ) & 1 )
                    != stg )
                    continue;
                const float_t eta ( eta_face( b_d, i, a,
                                              f_eta ) );
                float_t num ( b.at( i, 0 )[ a ] );
                float_t den ( 1 );
                for( int d = 0; d < n_dm; ++ d )
                {
                    const float_t h ( geo.dx_f( d, i ) );
                    const float_t w
                        ( dtc * eta / ( h * h ) );
                    auto kp ( i );  ++ kp[ d ];
                    auto km ( i );  -- km[ d ];
                    num += w * (   x.at( kp )[ a ]
                                 + x.at( km )[ a ] );
                    den += float_t( 2 ) * w;
                }
                x.at( i )[ a ] = num / den;
            }
        }
        return;
    };

    //  Residual  | b - A x |  at face ( i, a ),  A x = x
    //  - dt eta nabla^2 x.   Returns 0 for invalid faces.
    template< class bdt_T, class idx_T, class eta_T >
    __device__ __forceinline__ float_t residual
    ( const bdt_T & b_d, const idx_T & i, const int & a,
      const eta_T & f_eta )                       const
    {
        if( ! valid_face( b_d, i, a ) )
            return float_t( 0 );
        const float_t dtc ( b_d.dt );
        const auto   &  x ( b_d.fld.cg_x );
        const auto   &  b ( b_d.fld.bf );
        const float_t  ax ( x.at( i )[ a ]
             - dtc * eta_face( b_d, i, a, f_eta )
                   * laplacian( b_d, i, a, x ) );
        return utils::abs( b.at( i, 0 )[ a ] - ax );
    };

    //  Periodic ghost wrap of a face field v ( comp a ).
    template< class bdt_T, class idx_T, class arr_T >
    __device__ __forceinline__ void wrap_face
    ( const bdt_T & b_d, const idx_T & i, const int & a,
      arr_T & v )                                   const
    {
        const auto & geo = b_d.geo;
        const int    n_dm = geo.n_dim;
        type::idx_t src ( i );
        bool        gho ( false );
        for( int d = 0; d < n_dm; ++ d )
        {
            const int per = geo.n_ceff[ d ];
            const int hi  = ( d == a ? per : per - 1 );
            if( i[ d ] < 0 || i[ d ] > hi )
            {
                gho      = true;
                src[ d ] = ( ( i[ d ] % per ) + per ) % per;
            }
        }
        if( gho )
            v.at( i )[ a ] = v.at( src )[ a ];
        return;
    };

    //  Copy operations ( used to stage the initial guess
    //  and to write the result back ).
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ void cp_to_x
    ( const bdt_T & b_d, const idx_T & i,
      const int & a ) const
    {
        if( get_self(  ).valid_face( b_d, i, a ) )
            b_d.fld.cg_x.at( i )[ a ] =
                b_d.fld.bf  .at( i, 0 )[ a ];
        return;
    };
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ void cp_to_bf
    ( const bdt_T & b_d, const idx_T & i,
      const int & a ) const
    {
        if( get_self(  ).valid_face( b_d, i, a ) )
            b_d.fld.bf  .at( i, 0 )[ a ] =
                b_d.fld.cg_x.at( i )[ a ];
        return;
    };

    //  Launch resource covering the FULL padded box
    //  ( including the ghost shell ) --  used by the wrap
    //  kernel ( the ghosts must be wrapped too ).
    template< class bdt_T >
    __host__ std::tuple< dim3, dim3, int > wrap_res
    ( const bdt_T & b_d ) const
    {
        const auto & nc = b_d.fld.cg_x.n_cell;
        const int n_th = nc[ 0 ] < 128
            ? int( nc[ 0 ] ) : 128;
        return std::make_tuple
            ( dim3( nc[ 1 ], nc[ 2 ],
                    ( int( nc[ 0 ] ) + n_th - 1 ) / n_th ),
              dim3( n_th ), 0 );
    };

    ////////// Host-side driver //////////

    //  Solve  ( I - dt eta nabla^2 ) B_new = bf  by
    //  red-black Gauss-Seidel,  in place on bf ( slice 0 ).
    //  p_res is a device scalar for the residual reduction.
    template< class mod_T, class mesh_T, class dev_T,
              class itg_T, class eta_T >
    __host__ void solve
    ( mod_T & mod, mesh_T & mesh, dev_T & dev,
      const itg_T & f_itg, const eta_T & f_eta,
      const float_t & dtc, float_t * p_res,
      const int max_it = 500,
      const float_t tol = 1.e-6 ) const
    {
        const auto & self = get_self(  );
        //  Initial guess:  x = bf ( the old field ).
        for( auto & d : mod )
        {
            auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
            b_d.dt = float_t( dtc );
            b_d.dt   = float_t( dtc );
            const auto res = f_itg.field_res( b_d );
            dev.launch( ni_imp_cp_k  < decltype( b_d ),
                                       this_t, true >,
                        res, d.stream, b_d, self );
            dev.launch( ni_imp_wrap_k< decltype( b_d ),
                                       this_t,
                                       & decltype( b_d.fld )
                                       ::cg_x >,
                        self.wrap_res( b_d ), d.stream, b_d,
                        self );
        }
        dev.sync_all_streams(  );
        //  Gauss-Seidel iteration.
        int     it    ( 0 );
        float_t h_fin ( -1 );
        for( ; it < max_it; ++ it )
        {
            for( int stg = 0; stg < 2; ++ stg )
            {
                for( auto & d : mod )
                {
                    auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                    b_d.dt = float_t( dtc );
                    const auto res = f_itg.field_res( b_d );
                    dev.launch( ni_gs_k< decltype( b_d ),
                                         this_t, eta_T >,
                                res, d.stream, b_d, self,
                                f_eta, stg );
                }
                dev.sync_all_streams(  );
                for( auto & d : mod )
                {
                    auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                    b_d.dt = float_t( dtc );
                    const auto res = f_itg.field_res( b_d );
                    dev.launch( ni_imp_wrap_k
                                < decltype( b_d ), this_t,
                                  & decltype( b_d.fld )
                                  ::cg_x >,
                                self.wrap_res( b_d ),
                                d.stream, b_d, self );
                }
                dev.sync_all_streams(  );
            }
            //  Convergence check every few sweeps.
            if( it % 4 == 3 || it == max_it - 1 )
            {
                float_t zero ( 0 );
                dev.cp( p_res, & zero, 1 );
                for( auto & d : mod )
                {
                    auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                    b_d.dt = float_t( dtc );
                    //  The residual kernel reduces into a
                    //  __dyn_shared__ scratch ->  needs
                    //  smem ( field_res gives 0 ).
                    const auto res0
                        = f_itg.field_res( b_d );
                    const auto res  = std::make_tuple
                        ( std::get< 0 >( res0 ),
                          std::get< 1 >( res0 ),
                          std::get< 1 >( res0 ).x
                          * int( sizeof( float_t ) ) );
                    dev.launch( ni_imp_res_k
                                < decltype( b_d ), this_t,
                                  eta_T >,
                                res, d.stream, b_d, self,
                                f_eta, p_res );
                }
                dev.sync_all_streams(  );
                float_t h ( 0 );
                dev.cp( & h, p_res, 1 );
                mesh.p_com->reduce_all( h, comm::max );
                h_fin = h;
                if( h < tol )
                    break;
            }
        }
        printf
            (
            "  [implicit] dtc = %e, GS it = %d, res = %e\n",
            float_t( dtc ), it, float_t( h_fin ) );
        //  Write the result back into bf ( slice 0 ).
        for( auto & d : mod )
        {
            auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
            b_d.dt = float_t( dtc );
            const auto res = f_itg.field_res( b_d );
            dev.launch( ni_imp_cp_k  < decltype( b_d ),
                                       this_t, false >,
                        res, d.stream, b_d, self );
            dev.launch( ni_imp_wrap_k< decltype( b_d ),
                                       this_t,
                                       & decltype( b_d.fld )
                                       ::bf >,
                        self.wrap_res( b_d ), d.stream, b_d,
                        self );
        }
        dev.sync_all_streams(  );
        return;
    };
};

////////////////////////////////////////////////////////////
//  Thin kernels ( forward to the object )

template< class bdt_T, class imp_T, bool TO_X >
__global__ void ni_imp_cp_k
( const bdt_T b_d, const imp_T & f_imp )
{
    const auto & geo  = b_d.geo;
    type::idx_t  i    = { 0, int( blockIdx.x ),
                               int( blockIdx.y ) };
    const int n_feff  = blockDim.x + ( blockIdx.z + 1 ==
                                       gridDim.z ? 1 : 0 );
    for( int j = threadIdx.x; j < n_feff; j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            if( TO_X )  f_imp.cp_to_x ( b_d, i, a );
            else        f_imp.cp_to_bf( b_d, i, a );
        }
    }
    return;
};

template< class bdt_T, class imp_T, class eta_T >
__global__ void ni_gs_k
( const bdt_T b_d, const imp_T & f_imp, const eta_T & f_eta,
  const int stg )
{
    return f_imp.gs_sweep( b_d, stg, f_eta );
};

template< class bdt_T, class imp_T, auto V >
__global__ void ni_imp_wrap_k
( const bdt_T b_d, const imp_T & f_imp )
{
    const auto & geo  = b_d.geo;
    auto       & v    = b_d.fld.*V;
    const auto & ng   = v.n_gh;
    const auto & nc   = v.n_cell;
    type::idx_t  i;
    //  Cover the full padded box ( including the ghost
    //  shell );  logical indices start at - n_gh.
    i[ 0 ] = blockDim.x * int( blockIdx.z )
             + int( threadIdx.x ) - ng[ 0 ];
    i[ 1 ] = int( blockIdx.x ) - ng[ 1 ];
    i[ 2 ] = int( blockIdx.y ) - ng[ 2 ];
    if( i[ 0 ] >= nc[ 0 ] - ng[ 0 ] )
        return;
    for( int a = 0; a < geo.n_dim; ++ a )
        f_imp.wrap_face( b_d, i, a, v );
    return;
};

template< class bdt_T, class imp_T, class eta_T >
__global__ void ni_imp_res_k
( const bdt_T b_d, const imp_T & f_imp, const eta_T & f_eta,
  float_t * out )
{
    __dyn_shared__( float_t, mx );
    float_t      acc  ( 0 );
    const auto & geo  = b_d.geo;
    type::idx_t  i    = { 0, int( blockIdx.x ),
                               int( blockIdx.y ) };
    const int n_feff  = blockDim.x + ( blockIdx.z + 1 ==
                                       gridDim.z ? 1 : 0 );
    for( int j = threadIdx.x; j < n_feff; j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        for( int a = 0; a < geo.n_dim; ++ a )
            acc = utils::max
                ( acc, f_imp.residual( b_d, i, a, f_eta ) );
    }
    mx[ threadIdx.x ] = acc;
    __syncthreads(  );
    utils::block_reduce< float_t, __max1 >
        ( mx, blockDim.x );
    if( threadIdx.x == 0 )
        utils::atomic_max( out, mx[ 0 ] );
    return;
};

};  // namespace mhd_ee::ni
