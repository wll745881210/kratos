#pragma once

#include "../../../../mesh/types/dat_1d.h"

namespace dynamics::hydro::integrator
{
////////////////////////////////////////////////////////////
// Types

using mesh::dat_1d_t;

template< class f_T >
struct flux_info_t
{
    f_T  vol; f_T    f_a[ 2 ];
    f_T * fl; f_T * flux[ 2 ];
};

////////////////////////////////////////////////////////////
// Default data holder for hydrodynamic Godunov integrators

template< class bdt_T, class f_T, class f2_T >
struct  data_t
{
    using  f_t =  f_T;
    using f2_t = f2_T;

    bool is_uniform;               int    n_gh;
    int        axis;               int  n_ceff;
    int        step;               int  n_feff;
    int   blk_shift;               int  n_ctot;

    dat_1d_t< f2_T >   u;
    dat_1d_t<  f_T >   q;  dat_1d_t< f_T >  xf;
    dat_1d_t<  f_T >  ql;  dat_1d_t< f_T >  xv;
    dat_1d_t<  f_T >  qr;  dat_1d_t< f_T > flx;

    bool is_refined_comm[ 2 ];

    __device__ data_t(  ) = default;
    template < class eos_T > __device__
    data_t( const bdt_T &  b_d, const int & axis,
            const int   & step, void  *   shared,
            const eos_T & eos );
    __device__ void init
    ( const bdt_T &  b_d, const int & n_int,
      void  *   shared );

    template< class idx_T >  __forceinline__ __device__
    void  dispatch( idx_T & i, const bdt_T & b ) const;
    __device__ void  load_geometry ( const bdt_T & b );
    __device__ void  load_primative( const bdt_T & b );
};

////////////////////////////////////////////////////////////
// Constructor

template< class bdt_T, class f_T, class f2_T >
template< class eos_T >
__device__ data_t < bdt_T, f_T, f2_T >::data_t
( const bdt_T & b_d, const int & axis_, const int & step_,
  void  *    shared, const eos_T & eos )
    : axis( axis_ ), step( step_ )
{
    init( b_d, b_d.hyd.w.n_int, shared );
    load_geometry ( b_d );
    load_primative( b_d );
    return;
}

template< class bdt_T, class f_T, class f2_T >
__device__ void data_t < bdt_T, f_T, f2_T >::init
( const bdt_T & b_d, const int & n_int, void * shared )
{
    const auto & geo = b_d.geo;
    n_ceff     = blockDim.x;
    blk_shift  = blockDim.x * blockIdx.z;
    u.init( shared, n_ceff, n_int,   0 );
    // u has to have "original" n_ceff size unconditionally.
    // The final "n_feff" is the # of faces to be solved .
    is_refined_comm[ 0 ] = geo.refine_flag[ axis ] [ 0 ] &
                         ( blockIdx.z ==             0 ) ;
    is_refined_comm[ 1 ] = geo.refine_flag[ axis ] [ 1 ] &
                         ( blockIdx.z == gridDim.z - 1 ) ;
    if( is_refined_comm[ 0 ] )
    {
        blk_shift += 1;
        n_ceff    -= 1;
    }
    if( is_refined_comm[ 1 ] )
        n_ceff    -= 1;
    else if( axis > 0 && blockIdx.z < gridDim.z - 1 )
        n_ceff    -= 1;

    is_uniform  = geo.is_uniform[ axis ];
    n_gh        = b_d.hyd.order (      );
    n_feff      = n_ceff + 1;
    n_ctot      = n_ceff + 2 * n_gh;    
    shared = ql . init ( shared, n_feff, n_int,    0 );
    shared = qr . init ( shared, n_feff, n_int,    0 );
    shared = q  . init ( shared, n_ceff, n_int, n_gh );
    shared = flx. init ( shared, n_feff, n_int,    0 );
    if( ! is_uniform )
    {
        shared = xv .init( shared, n_feff, n_gh );
        shared = xf .init( shared, n_feff, n_gh );
    }
    return;
}

////////////////////////////////////////////////////////////
// Functions

template < class   bdt_T, class  f_T, class f2_T >
template < class   idx_T  >  __forceinline__
__device__ void   data_t< bdt_T, f_T, f2_T >::dispatch
( idx_T & i, const bdt_T  & b_d )  const
{
    i[ utils::mod3( axis + 1 ) ] = blockIdx.x;
    i[ utils::mod3( axis + 2 ) ] = blockIdx.y;
    return;
}

template< class bdt_T, class f_T, class f2_T > __device__
void data_t < bdt_T, f_T, f2_T >::load_geometry
( const bdt_T & b_d )
{
    if( is_uniform )
        return;
    for( int i = threadIdx.x; i <= n_ctot; i += blockDim.x )
    {
        const auto j ( i + blk_shift - n_gh );
        xf[ i ]  = b_d.geo.x_fc( axis,    j );
        xv[ i ]  = b_d.geo.x_cc( axis,    j );
    }
    return;
}

template< class bdt_T, class f_T, class f2_T > __device__
void data_t < bdt_T, f_T, f2_T >::load_primative
( const bdt_T & b_d )
{
    idx_t i;
    this->dispatch( i, b_d );
    const int i_th  =     threadIdx.x;
    const int n_var = b_d.hyd.w.n_int;
    const int n_tot = this->q.n_spa * n_var;
    for(  int j = i_th; j < n_tot; j += blockDim.x )
    {
        int i_s, n;
        utils::divmod( i_s, n, j, n_var );
        i_s      -=         q.n_gh;        
        i[ axis ] = i_s + this->blk_shift;
        q( i_s, n ) = b_d.hyd.w.at( i )[ n ];
    }
    return;
}

};         // namespace dynamics::hydro::integrator
