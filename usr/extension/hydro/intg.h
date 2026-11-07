#pragma once

#include "../../../src/modules/dynamics/hydro/hydro.h"

namespace extension::hydro
{
////////////////////////////////////////////////////////////
// Types

using          type::      float_t;
using          type::     float2_t;
using dual_t = mesh::block::dual_t;

////////////////////////////////////////////////////////////
//

template< typename T >
using  rk2_t = dynamics::hydro::integrator::rk2_t< T >;

template< class derived_T = crtp::dummy_t >
struct stp_t : crtp::helper_t< stp_t, derived_T, rk2_t >
{
    using super_t = rk2_t< stp_t< derived_T > >;

    ////////// Data //////////
    phys::unit_t< float_t > unit;
    type::coord_t            acc;

    ////////// Function //////////
    __host__ virtual void init
    ( const input & args, const dynamics::base_t & mod )
    {
        unit.init( args );
        acc = type::coord_t ( { 0., 0., 0. } );
        args       ( "prob", "acc", acc );
        return super_t::init( args, mod );
    };


    ////////// Add flux ////////
    template< class bdt_T, class eos_T >
    __device__ __forceinline__ void
    add_flux( const bdt_T &  b_d , const eos_T &  eos,
              const int   & axis , const bool  & side,
              const int   & step ) const
    {
        return super_t::add_flux
             ( b_d, eos, axis, side, step );
    };
    
    template< class dat_T, class bdt_T, class eos_T >
    __device__ void add_flux
    ( dat_T &, const bdt_T &, const eos_T & ) const;
};

template< class derived_T >
template< class dat_T, class bdt_T, class eos_T >
__device__ void stp_t< derived_T >::add_flux
( dat_T & d, const bdt_T & b_d, const eos_T & f_eos ) const
{
    const auto &  geo = b_d.geo;    
    const auto & self( * this );
    
    int    i[ 3 ],    i_1d_c, i_1d_p;
    self.dispatch( i, i_1d_c, i_1d_p, d, b_d );

    using  f_t = typename decltype( b_d.hyd ):: f_t;
    using f2_t = typename decltype( b_d.hyd )::f2_t;
    dynamics::hydro::integrator::flux_info_t< f_t > f;

    auto * u     = d.u.at( i_1d_c );
    auto * u_tgt = self.load( u , b_d.hyd.u,   i, d.step );
    const auto     rho = d.q( i_1d_p, idx::rho );
    const auto     pre = d.q( i_1d_p, idx::ene );    
    const auto & n_var = b_d.hyd.n_fld(  );
    for( int axis = 0; axis < b_d.geo.  n_dim; ++   axis )
    {
        self.load_flux_sfc  ( f, d, b_d, i, i_1d_p, axis );
        for( int n  = 0; n  < n_var; ++ n )
            u[ n ] += self.template flux_to_du
                    < f2_t >( f, b_d, i, axis, n );
    } 
    const auto rho_new = ( f_t( u[ idx::rho ] ) + rho ) / 2;
    const auto  dt_rho = b_d.dt * rho_new;
    f_t ek0( 0 ), ek1( 0 );
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        const auto p0 = f_t( u[ idx::mom + a ] );
        u[ idx::mom + a ] += acc[ a ] * dt_rho;
        const auto p1 = f_t( u[ idx::mom + a ] );
        ek0 += p0 * p0 / ( 2 * rho_new );
        ek1 += p1 * p1 / ( 2 * rho_new );
    }
    u[ idx::ene ] += ek1 - ek0;
    
    for( int n = 0 ; n < n_var; ++ n )
        u_tgt[ n ] = u [ n ];
    return;
}

};                         // namespace extension::hydro
