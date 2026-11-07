#pragma once

#include "../../../../utilities/types/crtp.h"
#include "../../prototypes/integrator.h"
#include "../types/block_data.h"
#include "data.h"
#include <tuple>

namespace dynamics::hydro::integrator
{
////////////////////////////////////////////////////////////
// Integrator for hydrodynamics problems; using CRTP

using dual_t = mesh::block::dual_t;

template< class derived_T = crtp::dummy_t >
struct base_t : ::dynamics::integrator::base_t
{
    ////////// CRTP functions //////////
    __crtp_def_self__( base_t, derived_T );

    ////////// Resource //////////
    template< class bdt_T >
    __host__ std::tuple < dim3, dim3, int > resource
    ( const  bdt_T & b_d,   const int & axis ) const;

    template< class idx_T > __device__
    void dispatch ( idx_T & i_3d, int & i_1d ) const;

    template< class idx_T, class dat_T, class bdt_T >
    __device__ void dispatch
    ( idx_T & i_3d, int & i_1d, int & i_1d_p ,
      const  dat_T  & d, const  bdt_T &  b_d ) const;

    //////////  Load/save fluxes and dimensions  //////////
    template< class fif_T, class dat_T ,
              class bdt_T, class idx_T >
    __device__  void load_flux_sfc
    ( fif_T & f, dat_T & d , const bdt_T &  b_d, idx_T & i,
      const int  &    i_1d , const int   & axis,
      const int  &  a0 = 0 ) const ;

    ////////// Device-side interfaces //////////
    template < class dat_T  , class bdt_T > __device__
    void slice_flux( dat_T &, const bdt_T & )  const;
    template < class dat_T  , class bdt_T > __device__
    void  save_flux( dat_T &, const bdt_T & )  const;

    template < class dat_T ,  class bdt_T, class eos_T >
    __device__ void  add_flux
    ( dat_T &, const bdt_T &, const eos_T & )    const ;
    template < class dat_T ,  class bdt_T, class eos_T >
    __device__ void  operator (  )
    ( dat_T &, const bdt_T &, const eos_T & )    const ;

    template< class  f2_T, class bdt_T ,
              class fif_T, class idx_T >
    __device__ __forceinline__ f2_T  flux_to_du
    ( const fif_T & f, const bdt_T & b_d, const idx_T & i ,
      const int &  ax, const int   & n )  const
    {
        f2_T res ( 0 );
        if( f.f_a[ 0 ]  > 0 )
            res += f.f_a[ 0 ] * f.flux[ 0 ][ n ];
        if( f.f_a[ 1 ]  > 0 )
            res -= f.f_a[ 1 ] * f.flux[ 1 ][ n ];
        return res;
    };

    ////////// Host-side interface //////////
    template< class bdt_T,   class  eos_T ,
              class rec_T,   class  rie_T > __host__ void
    calc_flx( const bdt_T &, const dual_t &, const eos_T &,
              const rec_T &, const  rie_T &, const int   &,
              device::base_t &  )   const ;

    ////////// Integrate the refined surface //////////
    template< class bdt_T >
    __host__ std::tuple< dim3, dim3, int >
    resource_ref    ( const bdt_T &, const int & ) const;
    template< class bdt_T > __host__ bool to_intg_flx
    ( const bdt_T &,  const int   &, const int & ) const;

    template< class bdt_T  , class  eos_T > __host__ void
    intg_flx( const bdt_T &, const dual_t & , const eos_T &,
              const int   &, device::base_t & ) const;

    template< class bdt_T, class eos_T > __device__ void
    add_flux( const bdt_T &, const eos_T &, const int & ax,
              const bool  & side, const int & step ) const;
};

////////////////////////////////////////////////////////////
// Resource of computing: allocation and dispatching

template< class derived_T >
template< class     bdt_T >  __host__
std::tuple< dim3, dim3, int > base_t< derived_T >::resource
( const bdt_T & b_d, const int & axis ) const
{
    //////////////////////////////////////////////////
    // Grid and block dimensions
    const int  n_var = b_d. hyd.u.n_int;
    const int  n_th  = std::min
        ( b_d.geo.n_ceff[ axis ] , n_th_lim  );
    if  ( b_d.geo.n_ceff[ axis ] % n_th != 0 )
        throw std::runtime_error( "Indivisible stride: "
                                  "integrator::base_t. " );
    const dim3  n_bl ( b_d.geo.n_ceff[ ( axis + 1 ) %  3 ],
                       b_d.geo.n_ceff[ ( axis + 2 ) %  3 ],
                       b_d.geo.n_ceff[   axis   ] / n_th );
    const int & n_ceff = n_th;
    const int & n_feff = n_ceff + 1;
    const int & n_ctot = n_ceff + 2 * b_d.geo.n_gh;
    const int & n_ftot = n_ctot + 1;

    //////////////////////////////////////////////////
    // Size of dynamically allocated shared memory;
    // See data.h for the amount of data
    using  f_t = typename decltype( b_d.hyd ):: f_t;
    using f2_t = typename decltype( b_d.hyd )::f2_t;
    static const size_t s_f ( sizeof(  f_t ) );
    int s_sh  = ( n_ctot + 3 * n_feff + 1 ) * s_f * n_var;
    if( ! b_d.geo.is_uniform [ axis ] )
        s_sh += 2 * n_ftot * s_f;

    return std::make_tuple( n_bl, dim3( n_th ), s_sh );
}

template< class derived_T >
template< class     idx_T > __device__
void base_t< derived_T >::dispatch
( idx_T & i_3d, int & i_1d ) const
{
    // Note: this offset is different from data_t
    i_1d      = threadIdx.x;
    i_3d[ 0 ] =  blockIdx.z * blockDim.x + threadIdx.x;
    return;
}

template< class derived_T >
template< class idx_T, class dat_T, class bdt_T > __device__
void base_t< derived_T >::dispatch
( idx_T & i_3d, int & i_1d_c, int & i_1d_p,
  const   dat_T & d , const bdt_T & b_d  ) const
{   // For adding the fluxes to conserved quantitites
    d              .dispatch( i_3d,    b_d ); // ax_1, ax_2
    get_self(  )   .dispatch( i_3d, i_1d_c ); // axis, i_1d
    i_1d_p = i_1d_c - d.is_refined_comm[ 0 ]; // prim, flux
    return;
}

////////////////////////////////////////////////////////////
// Load fluxes and dimensions

template< class derived_T >
template< class fif_T, class dat_T ,
          class bdt_T, class idx_T >
__device__ void base_t< derived_T >::load_flux_sfc
( fif_T & f, dat_T & d, const bdt_T &  b_d , idx_T & i,
  const int  &    i_1d, const int   & axis, const int & a0 )
  const
{
    const auto &  geo = b_d.  geo;
    f.vol      =  geo.volume( i );
    const auto  dt_dv = b_d. dt  / f.vol ;
    const bool null_l( geo.refine_flag[ axis ][ 0 ] &&
                       i[ axis ] ==                  0 ) ;
    const bool null_r( geo.refine_flag[ axis ][ 1 ] && 1 +
                       i[ axis ] == geo.n_ceff[ axis ] ) ;
    f.flux[ 0 ] = ( axis != a0 ? b_d.hyd.f [ axis ].at( i ):
                  ( null_l ? 0 : d.flx.at( i_1d ) ) );
    f.f_a [ 0 ] = ( null_l ? -1 :
                    geo.surface( i, axis ) * dt_dv );
    ++  i [ axis ];
    f.f_a [ 1 ] = ( null_r ? -1 :
                    geo.surface( i, axis ) * dt_dv );
    f.flux[ 1 ] = ( axis != a0 ? b_d.hyd.f [ axis ].at( i ):
                  ( null_r ? 0 : d.flx.at( i_1d + 1 ) ) );
    --  i [ axis ];
    return ;
}

////////////////////////////////////////////////////////////
// Device-side interfaces

template < class derived_T >
template < class dat_T, class bdt_T > __device__
void base_t< derived_T >::save_flux
( dat_T & d, const bdt_T & b_d ) const
{
    idx_t i;
    d.dispatch( i, b_d ) ;
    const int n_var    = b_d.hyd. w.n_int;
    const int n_feff   = d.n_feff
        - ( d.axis == 0 && blockIdx.z < gridDim.z - 1 );
    const int n_tot    =   n_feff * n_var;
    const auto    &  f = b_d.hyd.f[ d.axis ];
    for( int j = threadIdx.x; j < n_tot; j += blockDim.x )
    {
        int    i_s, n;
        utils::divmod( i_s, n, j,  n_var );
        i   [ d.axis ] = i_s + d.blk_shift;
        f.at( i )[ n ] = d.flx ( i_s,  n );
    }
    return;
}

template < class derived_T >
template < class dat_T, class bdt_T > __device__
void base_t< derived_T >::slice_flux
( dat_T & d, const bdt_T & b_d ) const
{
    const auto & geo = b_d.geo;
    const auto &   f = b_d.hyd.f[ d.axis ];
    const bool slice_l( geo.coarse_flag[ d.axis ][ 0 ] &&
                        blockIdx.z ==              0 ) ;
    const bool slice_r( geo.coarse_flag[ d.axis ][ 1 ] &&
                        blockIdx.z == gridDim.z -  1 ) ;
    if( ! ( slice_l || slice_r ) )
        return;
    idx_t i;
    d.dispatch( i, b_d );
    if( slice_l )
    {
        i [ d.axis ] = 0;
        auto * f_tgt = f.at( i );
        for( int n = 0 ; n < f.n_int; ++ n )
            f_tgt[ n ] = d.flx( 0,       n );
    }
    if( slice_r )
    {
        i [ d.axis ] = geo.n_ceff[ d. axis ];
        auto * f_tgt = f.at( i );
        for( int n = 0 ; n < f.n_int; ++ n )
            f_tgt[ n ] = d.flx( d.n_feff - 1,  n );
    }
    return;
}

template < class derived_T >
template < class dat_T, class bdt_T, class eos_T >
__device__ void base_t< derived_T >:: add_flux
( dat_T & d, const bdt_T & b_d, const eos_T & eos ) const
{
    const auto & self = get_self(  );
    int    i[ 3 ],    i_1d_c, i_1d_p;
    self.dispatch( i, i_1d_c, i_1d_p, d, b_d );
    using  f_t = typename decltype( b_d.hyd ):: f_t;
    using f2_t = typename decltype( b_d. hyd )::f2_t;
    hydro::integrator::flux_info_t< f_t >          f;
    auto * u     = d.u.at( i_1d_c );
    auto * u_tgt = self.load( u , b_d.hyd.u,  i, d.step );
    const    auto &   n_var = b_d.hyd.n_fld(  );
    for( int axis = 0; axis < b_d.geo.  n_dim; ++  axis )
    {
        self.load_flux_sfc ( f, d, b_d, i, i_1d_p, axis );
        for( int n  = 0; n < n_var; ++ n )
            u[ n ] += self.template flux_to_du
                    < f2_t >( f, b_d, i, axis, n );
    }
    for ( int  n = 0 ; n < n_var; ++ n )
        u_tgt[ n ] = u [ n ];
    return;
}

template < class derived_T >
template < class dat_T, class bdt_T, class eos_T >
__device__ void base_t< derived_T >::operator(  )
( dat_T & d, const bdt_T & b_d, const eos_T & eos ) const
{
    __syncthreads(  );
    if( d.axis != 0 )
        get_self(   ). save_flux( d, b_d );
    else if( b_d.geo.coarse_flag[ d.axis ][ 0 ] ||
             b_d.geo.coarse_flag[ d.axis ][ 1 ]  )
        get_self(   ).slice_flux( d, b_d );
    if( d.axis == 0 )
    {
        __syncthreads(  );
        get_self(   ).  add_flux( d, b_d, eos )  ;
    }
    return;
}

////////////////////////////////////////////////////////////
// Invoke the integrator

template< class derived_T >
template< class bdt_T, class eos_T,
          class rec_T, class rie_T >
__host__  void base_t< derived_T >::calc_flx
( const    bdt_T &   b_d , const dual_t &     d,
  const    eos_T & f_eos , const rec_T  & f_rec,
  const    rie_T & f_rie , const int    &  step,
  device::base_t &   dev ) const
{
    const  auto & f_itg  = get_self             (   )  ;
    using   f_t = typename decltype( b_d.hyd )::  f_t  ;
    using  f2_t = typename decltype( b_d.hyd ):: f2_t  ;
    using dat_t = data_t < decltype( b_d ), f_t, f2_t >;
    for( int ax = b_d.geo.n_dim - 1; ax  >= 0 ; -- ax )
        dev.launch
        ( ::dynamics::integrator::flux_kernel < dat_t >,
          f_itg.resource( b_d, ax ),   d.stream,  b_d  ,
          ax,   step, f_rec,  f_rie,   f_eos,   f_itg );
    return;
}

////////////////////////////////////////////////////////////
// Integrate the surface

template< class bdt_T, class eos_T, class itg_T >
__global__ void intg_ker( const bdt_T b_d, const eos_T eos ,
                          const int  axis, const bool side ,
                          const int  step, const itg_T itg )
{
    return itg.add_flux( b_d, eos, axis, side, step );
}

template< class derived_T >
template< class bdt_T >  __host__ std::tuple
< dim3, dim3, int > base_t< derived_T >::resource_ref
( const bdt_T & b_d, const int & axis  ) const
{
    const auto nc_ap1  = b_d.geo.n_ceff[ ( axis + 1 ) % 3 ];
    const int  n_th    = std::min( nc_ap1, n_th_lim ) ;
    if( nc_ap1 % n_th != 0 )
        throw std::runtime_error( "Indivisible stride: "
                                  "integrator::base_t. " );
    const dim3 n_bl( b_d.geo.n_ceff[ ( axis + 2 ) % 3 ],
                     1, nc_ap1  / n_th );
    return std::make_tuple( n_bl, dim3( n_th ), 0 );
}

template< class derived_T >
template< class     bdt_T >
__host__ bool base_t< derived_T >::to_intg_flx
( const bdt_T & b_d, const int & ax, const int & s ) const
{
    return b_d.geo.refine_flag[ ax ][ s ];
}

template < class derived_T >
template < class     bdt_T, class eos_T >
__device__ void base_t< derived_T >:: add_flux
( const bdt_T &  b_d, const eos_T & eos, const int & axis,
  const bool  & side, const int & step ) const
{
    idx_t i;
    i[ axis ] = side * b_d.geo.n_ceff[ axis ];
    i[ utils::mod3( axis + 2 ) ] =  blockIdx.x;
    i[ utils::mod3( axis + 1 ) ] = threadIdx.x
                    + blockIdx.z *  blockDim.x;
    auto * f = b_d.hyd.f[ axis ].at( i );
    const auto ds = b_d.geo.surface( i, axis );

    i[ axis ]    -= side;
    int s ( step == b_d.hyd.u.n_step - 1 ? 0 : step + 1 ) ;
    auto  *     u = b_d.hyd.u.at  ( i, s ) ;
    const auto dv = b_d.geo.volume( i ) * ( side ? -1 : 1 );
    const auto conv = b_d.dt * ds / dv;
    for( int n = 0; n < b_d.hyd.n_fld(  ); ++ n )
        u[ n ] += conv * f[ n ];
    return;
}

template< class derived_T >
template< class     bdt_T, class eos_T >
__host__  void base_t< derived_T >::intg_flx
( const bdt_T &  b_d, const dual_t &  d, const eos_T & eos, 
  const int   & step, device::base_t  &  dev ) const
{
    const  auto & f_itg = get_self (  );
    for( int ax = b_d.geo.n_dim -  1; ax >= 0 ; -- ax )
        for( int side = 0; side <= 1; ++ side )
            if( f_itg.to_intg_flx( b_d, ax, side ) )
                dev.launch( intg_ker, f_itg.resource_ref
                          ( b_d, ax ), d.stream, b_d, eos, 
                            ax, bool( side ), step, f_itg );
    return;
}

};            // namespace dynamics::hydro::integrator
