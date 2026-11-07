#pragma once

#include "../evolve/integrator.h"

namespace particle::cic
{
////////////////////////////////////////////////////////////
// Kernels

template< class itg_T, class pol_T, class map_T >
__global__ void ker_dep
( const itg_T itg, const pol_T pool, const map_T bmap )
{
    return itg.deposite( pool, bmap );
}

////////////////////////////////////////////////////////////
// Integrator type for cloud-in-cell (CIC) particles

template< typename T >
using itg_t = particle::integrate::base_t< T >;

template < class derived_T = crtp::dummy_t >
struct intg_t : crtp::helper_t< intg_t, derived_T, itg_t >

{
    ////////// Types //////////
    __crtp_def_self__( intg_t, derived_T )
    using super_t
        = crtp::helper_t< intg_t, derived_T, itg_t >;

    ////////// Functions //////////
    template< class pol_T, class map_T, class  com_T >
    __host__  void  post_proc
    ( pol_T & pool, map_T & bmap,
      com_T & comm, particle::base_t & mod )        
    {
        const auto & n_th( this->n_th ) ;
        int n_bl = ( pool.n_par_eff(  ) + n_th - 1 ) / n_th;
        if( n_bl > 0 )
        {
            const auto  lpar =  std::make_tuple
                ( dim3( n_bl ), dim3( n_th ), 0 );
            mod.p_dev->launch( ker_dep, lpar, mod.stream  ,
                               get_self(  ), pool,  bmap );
        }
        return super_t::post_proc( pool, bmap, comm, mod );
    };

    ////////// Device-side functions //////////
    template< class pol_T, class map_T > __device__ void
    deposite( const pol_T & pool, const map_T & bmap ) const
    {
        bool flag;
        typename pol_T::par_t par;
        const auto & i_par = pool.load( par, flag );
        if( flag )
        {
            par.find_cell( bmap  ) ;
            par.deposite ( bmap[ par.ib_l ], get_self(  ) );
        }
        return;
    };

    __device__ __forceinline__ type::float_t couple( ) const
    {
        return 1;
    };
};
};                              // namespace particle::cic
