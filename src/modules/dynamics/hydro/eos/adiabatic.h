#pragma once

#include "../../../../utilities/mapping/reduction.h"
#include "../../../../utilities/functions.h"
#include "../../indices.h"
#include "../../prototypes/eos.h"
#include "../types/block_data.h"

namespace dynamics::hydro::eos
{
////////////////////////////////////////////////////////////
// Hydrodynamic adiabatic, constant or variable gamma

template< class derived_T    = crtp:: dummy_t >
struct adiabatic_t : crtp::helper_t
< adiabatic_t , derived_T, ::dynamics::eos::interface_t >
{
    ////////// Types //////////
    __crtp_def_self__( adiabatic_t, derived_T );
    using super_t = crtp::helper_t< adiabatic_t ,
        derived_T, ::dynamics::eos::interface_t >;
    
    ////////// Data //////////
    int  n_hyd;

    ////////// Initialization //////////
    __host__ adiabatic_t       (  ) :       n_hyd( 5 ) {  };
    __host__ virtual int n_hvar(  ) const { return n_hyd; };

    ////////// Resource allocation //////////
    template < class blk_T >
    __host__ std::tuple< dim3, dim3, int >
    resource ( const  blk_T & b  ) const ;
    __device__ void dispatch( idx_t & i ) const ;

    ////////// Workers //////////
    template< class f_T, class f2_T >  __device__ f_T   c2p
    ( f_T * p, f2_T * c,  f_T * p_src, f2_T * c_src ) const;

    template< class f_T, class bdt_T, class idx_T >
    __device__  f_T dt
    ( const f_T   * v_p, const f_T   & cs2 ,
      const idx_T &   i, const bdt_T & b_d ) const;

    template< class bdt_T, class idx_T >
    __device__ __forceinline__ auto * load_prim
    ( const  bdt_T & b_d , const idx_T & i,
      const  int &  step ) const
    {
        return b_d.hyd.w.at( i       );
    }; 

    template< class bdt_T, class idx_T >
    __device__ __forceinline__ auto * load_cons
    ( const  bdt_T & b_d , const idx_T & i,
      const  int &  step ) const
    {
        return b_d.hyd.u.at( i, step );
    };

    ////////// Device-side interface //////////
    template < class f_T >  __device__ __forceinline__
    f_T et_cs( f_T & e_tot, f_T & c_s, f_T * w ) const
    {
        const auto & pre   = w[ idx::ene ];
        const auto & rho   = w[ idx::rho ];
        f_T cv_kb( 1.5f ), gamma( 1.667f );
        get_self(  ).w_prho_d( cv_kb, gamma, w, pre, rho );
        
        f_T  v2  ( 0 );
        for( int a = 0; a < 3; ++ a )
             v2 += w[ idx::mom + a ] * w[ idx::mom + a ];
        e_tot = pre * cv_kb  + rho * v2 / 2 ;
        c_s   = __sqrt1( gamma * __max1( pre / rho, 0 ) );
        return cv_kb;
    };
    template< class f_T, class g_T > __forceinline__
    __device__ f_T pre_cs2( f_T * w, const g_T & e_g ) const
    {
        f_T cv_kb( 1.5f ), gamma( 1.667f );
        get_self(  ).w_erho_d
            ( cv_kb, gamma, w, f_T( e_g ), w[ idx::rho ] );
        w[ idx::ene ] = e_g / cv_kb;
        return  gamma * w [ idx::ene ] / w[ idx::rho ];
    };
    template< class bdt_T > __device__ void operator(  )
    ( const bdt_T & b_d, const int & step ) const;

    template< class v_T, class f_T > __device__
    void w_erho_d( f_T & cv_kb, f_T & gamma, const v_T * w,
                   const f_T & e_g, const f_T & rho ) const
    {   // input: primative num frac, e_g, rho;
        gamma = this->gamma;
        cv_kb = 1 / ( gamma - 1 );
        return;
    };
    
    template< class v_T, class f_T > __device__
    void w_prho_d( f_T & cv_kb, f_T & gamma, const v_T * w,
                   const f_T & pre, const f_T & rho ) const
    {   // input: primative num frac, pre, rho;
        gamma = this->gamma;
        cv_kb = 1 / ( gamma - 1 );
        return;
    };
};

////////////////////////////////////////////////////////////
// Resource allocation

template< class derived_T >
template< class     blk_T >
__host__ std::tuple< dim3, dim3, int >
adiabatic_t< derived_T >::resource( const blk_T & b_d )const
{
    const auto &  hyd = b_d.hyd;
    const int  s_sh_s = n_hyd * type::size( hyd.u.dat )
               +  hyd.w.n_int * type::size( hyd.w.dat );

    const int  n_th = std::min( hyd.u.n_cell[ 0 ],
                                this->n_th_lim );
    if( b_d.geo.n_ceff[ 0 ] % n_th != 0 )
        throw std::runtime_error
            ( "Indivisible stride, c2p_t::convert." );
    const dim3  n_bl ( b_d.geo.n_ceff[ 1 ] ,
                       b_d.geo.n_ceff[ 2 ] ,
                       b_d.geo.n_ceff[ 0 ] / n_th );
    const int   s_sh = n_th * s_sh_s;
    return std::make_tuple( n_bl, dim3( n_th ), s_sh );
};

template   < class derived_T > __device__ void
adiabatic_t< derived_T > ::dispatch( idx_t & i ) const
{
    i[ 0 ] = ( threadIdx.x + blockDim.x * blockIdx.z );
    i[ 1 ] =    blockIdx.x ;
    i[ 2 ] =    blockIdx.y ;
    return;
}

////////////////////////////////////////////////////////////
// Device-side workers

template< class derived_T >
template< class f_T, class bdt_T, class idx_T >
__device__ f_T adiabatic_t< derived_T >::dt
( const f_T   * v_p, const f_T   & cs2 ,
  const idx_T &   i, const bdt_T & b_d ) const
{
    const auto &  geo = b_d.geo  ;
    f_T            dt ( FLT_MAX );
    const f_T cs = __sqrt1( cs2 );
    for( int a = 0;  a < geo.n_dim; ++ a )
        dt = __min1( dt, geo.dx_cfl ( a, i ) /
           ( cs + __abs1 ( v_p[ idx::mom + a ] ) ) );
    return   dt * this->cfl;
};

template< class       derived_T >
template< class f_T, class f2_T >
__device__ f_T adiabatic_t< derived_T >::c2p
( f_T * prim, f2_T * cons, f_T * p_src, f2_T * c_src ) const
{
    for( int n = 0; n < n_hyd; ++ n )
        cons[ n ] = c_src[ n ];

    auto & e_t = cons[ idx::ene ];
    auto & rho = cons[ idx::rho ];

    //////////////////////////////////////////////
    // Mass density floor
    if( __isnan( rho ) || rho  < this->d_floor )
    {
        rho                    = this->d_floor ;
        c_src [ idx::rho ]     = this->d_floor ;
    }
    prim[ idx::rho ]   =     rho;
    const auto rho_inv = 1 / rho;

    //////////////////////////////////////////////
    // Energy and momentum
    f2_T   e_k( 0. );
    for( int n = idx::mom; n < idx::mom + 3; ++ n )
    {
        prim[ n ] = rho_inv   * cons[ n ];
        e_k      += cons[ n ] * cons[ n ];
    }
    e_k           *= rho_inv / 2;
    const auto e_g = e_t - e_k;
    auto       cs2 = get_self(  ).pre_cs2( prim, e_g );
    auto &     pre = prim[ idx::ene ];
    auto pre_floor = this->t_floor * prim[ idx:: rho ];
    if( __isnan( pre ) || pre < pre_floor )
    {
        pre  = pre_floor;
        // Heal the conserved energy with the FLOORED
        // pressure: writing back the raw ( possibly
        // negative ) e_g leaves the cell permanently
        // pressureless.
        c_src[ idx::ene ]  = e_k + pre / ( this->gamma - 1 );
        cs2  = this->gamma * pre / rho;
    }
    for( int n = 0; n < n_hyd; ++ n )
        p_src[ n ] = prim[ n ];
    return cs2;
};

////////////////////////////////////////////////////////////
// Device-side interface

template < class  derived_T >
template < class      bdt_T > __device__
void adiabatic_t< derived_T >::operator(  )
( const bdt_T & b_d, const int & step ) const
{
    const auto & self = this->get_self(  );

    //////////////////////////////////////////////
    // Numbers and indices
    const int i_th =  threadIdx.x;
    const int n_th =   blockDim.x;

    //////////////////////////////////////////////
    // Dispatching dynamic shared memory and
    // access the conservatives / primitives
    using  f_t = typename decltype( b_d.hyd ):: f_t;
    using f2_t = typename decltype( b_d.hyd )::f2_t;

    __dyn_shared__( char, var_sh );
    auto * dt  =  (  f_t * ) ( var_sh );
    auto * vf2 =  ( f2_t * )   var_sh;
    auto * v_c = vf2 + i_th * n_hyd ;
    auto * v_p = ( f_t * )( vf2 + n_th * n_hyd )
               + i_th * b_d.hyd.w.n_int;

    idx_t  i;
    self.dispatch( i );
    auto  *  p = self.load_prim( b_d,   i, step );
    auto  *  c = self.load_cons( b_d,   i, step );
    f2_t c0[ 7 ];
    for( int n = 0; n < 7; ++ n )
        c0[ n ] = c[ n ];
    const auto cs2  = self.c2p ( v_p, v_c, p, c );

    //////////////////////////////////////////////////
    // Calculate dt
    if( step == 0 )
    {
        __syncthreads(  );
        dt[ i_th ] = self.dt ( v_p, cs2,   i, b_d );
        __syncthreads(  );        
        utils::block_reduce
              < f_t, __min1 >( dt,     blockDim.x );
        
        if( i_th == 0 )
            utils::atomic_min( b_d.p_dt,  dt[ 0 ] );
    }
    return;
}
};                     // namespace dynamics::hydro::eos
