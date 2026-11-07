#pragma once

#include "../../../src/comm/comm.h"
#include "grav_src.h"

namespace friction
{
////////////////////////////////////////////////////////////
// Hydrodynamic integrator

template< typename T >
using  rk2_t = dynamics::hydro::integrator::rk2_t< T >;

template< class derived_T = crtp::dummy_t >
struct stp_t : crtp::helper_t< stp_t, derived_T, rk2_t >
{
    ////////// Types //////////
    using float_t = type::float_t;
    using coord_t = type::coord_t;
    using super_t = rk2_t< stp_t< derived_T > >;

    ////////// Data //////////
    star_t               star;
    float_t *          p_grav;
    float_t *        p_grav_h;
    bool            stat_grav;

    int                n_step;

    ////////// Function //////////
    __host__ virtual void init
    ( const input & args,
      const dynamics::base_t & mod ) override
    {
        p_grav   = mod.p_dev->malloc_device< float_t >( 3 );
        p_grav_h = mod.p_dev->malloc_host  < float_t >( 3 );
        star    .init( args      );
        super_t::init( args, mod );
        mod.p_dev->f_mset( p_grav, 0,
                           3 * sizeof( * p_grav ), 0 );
        mod.p_dev->sync_all_streams(  );
        n_step   = this->time_order(  );

        auto & cyc = dynamic_cast< cycle_t & >
            ( * mod.p_mesh->p_cyc );
        cyc.f_pre = [ & ] ( const bool & stat_grav )
        {
            this->stat_grav = stat_grav;
        };
        cyc.f_acc = [ & ] (  )
        {
            if( ! stat_grav )
                return;
            mod.p_dev->sync_all_streams(  );
            mod.p_dev->cp( p_grav_h, p_grav, 3 );
            mod.p_dev->f_mset( p_grav, 0, 3 * sizeof
                             ( * p_grav ), 0 );
            mod.p_dev ->sync_stream( 0 );
            mod.p_mesh->p_com->reduce_all
                      ( p_grav_h, comm::sum, 3 );
            if( mod.p_mesh->p_com->is_root(  ) )
                std::cout << "acc = [ " << p_grav_h[ 0 ]
                          << ", " << p_grav_h[ 1 ] << ", "
                          << p_grav_h[ 2 ] << " ]\n";
        };
        cyc.f_out = [ & ] ( std::ofstream & fout )
        {
            fout << p_grav_h[ 0 ] << " " << p_grav_h[ 1 ]
                 << " " << p_grav_h[ 2 ] << "\n";
        };
        return;
    };

    template< class dat_T , class bdt_T, class eos_T >
    __device__ void add_flux
    ( dat_T &, const bdt_T &, const eos_T & )  const ;

    template< class bdt_T, class eos_T >
    __device__ __forceinline__ void
    add_flux( const bdt_T &  b_d , const eos_T &  eos,
              const int   & axis , const bool  & side,
              const int   & step ) const
    {
        return super_t::add_flux
             ( b_d, eos, axis, side, step );
    };

    __host__ virtual void finalize
    ( const::dynamics::base_t & mod ) override
    {
        mod.p_dev->free_device( p_grav   );
        mod.p_dev->free_host  ( p_grav_h );
        return;
    };

    template< class dat_T > __device__ __forceinline__
    bool f_grav( coord_t & x, coord_t       & acc,
                 dat_T   & d, const float_t & dm ) const
    {
        float_t r_2;
        bool in_center = star.get_acc( acc, r_2, x );
        if( d.step != this->n_step - 1 || ( ! stat_grav ) )
            return in_center;

        __shared__ float_t acc_sh[ 3 ];
        if( threadIdx.x < 3 )
            acc_sh[ threadIdx.x ] = 0;
        __syncthreads(  );
        if( ! in_center )
            for( int a = 0; a < 3; ++ a )
                atomicAdd( acc_sh + a,
                           -dm / star.g_m * acc[ a ] );
        __syncthreads(  );
        if( threadIdx.x < 3 )
            atomicAdd ( p_grav +  threadIdx.x ,
                        acc_sh [  threadIdx.x ] ) ;
        return in_center;
    };
};

////////////////////////////////////////////////////////////
// Divergence added to the conserved field

template < class derived_T >
template < class dat_T, class bdt_T, class eos_T >
__device__ void  stp_t< derived_T >:: add_flux
( dat_T & d, const bdt_T & b_d, const eos_T & eos ) const
{
    const auto &  geo = b_d.geo;
    const auto & self( * this );

    int i[ 3 ], i_1d_c, i_1d_p;
    self.dispatch  ( i, i_1d_c, i_1d_p, d, b_d );

    using  f_t = typename decltype( b_d.hyd )::   f_t;
    using f2_t = typename decltype( b_d.hyd )::  f2_t;
    dynamics::hydro::integrator::flux_info_t< f_t > f;

    auto * u     = d.u.at( i_1d_c );
    auto * u_tgt = self.load( u, b_d.hyd.u, i, d.step );
    const auto & n_var = b_d.hyd.u.n_int;
    const auto  &  rho = d.q( i_1d_p, idx::rho );
    const auto  &  pre = d.q( i_1d_p, idx::ene );
    const auto  dt_rho = b_d. dt  *        rho  ;

    coord_t   x;
    geo.x_cc( x, i );
    coord_t acc( 0 );
    const auto in_center = f_grav
        ( x, acc, d, rho * geo.volume( i ) );
    const auto *    vel  = & d.q( i_1d_p, idx::mom );

    f_t  de( 0 );
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        self.load_flux_sfc( f,  d, b_d, i, i_1d_p, a );
        for( int n  = 0; n  < n_var; ++ n )
            u[ n ] += self.template flux_to_du
                   < f2_t >( f, b_d, i, a, n ) ;        
        u[ idx::mom + a ] += dt_rho * acc[ a ] ;
        de                += dt_rho * acc[ a ] * vel[ a ];
    }
    u[ idx::ene ] += de;
    if( in_center )
        star.set_flow( u, x );
    for( int n = 0 ; n < n_var; ++ n )
        u_tgt[ n ] = u [ n ];
    return;
}

};                              // namespace cmz
