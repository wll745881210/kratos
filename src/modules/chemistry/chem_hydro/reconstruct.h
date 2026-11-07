#pragma once

#include "../../../utilities/types/my_type_traits.h"
#include "../../dynamics/hydro/reconstruct/plm.h"
#include "../corrector/stoichiometry.h"
#include "base_gen.h"

namespace chemistry::hydro
{
////////////////////////////////////////////////////////////
// Reconstruction with chemistry

template< typename T >
using rec_hy_t = dynamics::hydro::reconstruct::plm_t< T >;

template< template< class T > class base_T = rec_hy_t >
struct rec_t : base_gen_t, base_T< rec_t< base_T > >
{
    ////////// Types //////////
    // __crtp_def_self__( rec_t, derived_T );
    using super_t = base_T< rec_t< base_T > >;

    ////////// Data //////////
    ::chemistry::sto_t stoi;
    bool         correction;
    
    using base_gen_t::n_hva;
    using base_gen_t::n_sps;

    using super_t::q_ll, super_t::q_l;
    using super_t::q_rr, super_t::q_r;
    using super_t::        is_uniform;

    ////////// Host-side interface //////////
    __host__ virtual void set_chem
    ( const input & args, const sp_vec_t & species,
      device::base_t &  f_dev ) override
    {
        correction = true;
        const auto algo = args.get < std::string >
              ( "dynamics", "chem_recon",  "stoi" );
        if( algo.find( "stoi" ) != std::string::npos )
            stoi.init( f_dev, species );
        else if( algo.find( "none" ) != std::string::npos )
            correction = false;
        return base_gen_t::set_chem( args, species, f_dev );
    };

    ////////// Device-side interface //////////
    template< class dat_T, class bdt_T >
    __device__ void cals_d
    ( dat_T & d, const int & i, const bdt_T & bdt ) const
    {
        auto * aux = d.flx.at( i ) + n_hva;
        auto * ql  = d.ql .at( i ) ;
        auto * qr  = d.qr .at( i ) ;
        const auto n_tot( n_sps + n_hva );
        for( int n = n_hva - 1; n >= 0; -- n )
        {
            q_ll = d.q( i - 2, n );
            q_l  = d.q( i - 1, n );
            q_r  = d.q( i    , n );
            q_rr = d.q( i + 1, n );
            if(  is_uniform )
                this->uniform( d, i, n, bdt );
            else
                this->general( d, i, n, bdt );
        }        
        using f_t = type::prim_type_t
                  < decltype( this->cfl ) >;
        const f_t cfl( is_uniform ? 1  :  this->cfl );
        const f_t cbl( is_uniform ? 1  :  this->cbl );
        const f_t cfr( is_uniform ? 1  :  this->cfr );
        const f_t cbr( is_uniform ? 1  :  this->cbr );
        for( int n = n_hva; n < n_tot; ++ n )
        {
            const auto dql
                = ( d.q( i - 1, n ) - d.q( i - 2, n ) )
                * ( is_uniform ? 1 : this-> dxv_l_i ) ;
            const auto dqc
                = ( d.q( i + 0, n ) - d.q( i - 1, n ) )
                * ( is_uniform ? 1 : this-> dxv_c_i ) ;
            const auto dqr
                = ( d.q( i + 1, n ) - d.q( i + 0, n ) )
                * ( is_uniform ? 1 : this-> dxv_r_i ) ;
            ql[ n ] = d.q( i - 1, n ) + this->ql_coef
                *  ( cfl * dql + cbl * dqc );
            qr[ n ] = d.q( i    , n ) + this->qr_coef
                *  ( cfr * dqc + cbr * dqr );
        }
        return;
    };

    template< class dat_T, class bdt_T >
    __device__ void cals_s
    ( dat_T & d, const int & i, const bdt_T & bdt ) const
    {
        auto * aux = d.flx.at( i ) + n_hva;
        auto * ql  = d.ql .at( i ) ;
        auto * qr  = d.qr .at( i ) ;   
        const auto n_tot( n_sps + n_hva );
        super_t::cals( d, i, bdt );
        for( int n = n_hva; n < n_tot; ++ n )
        {   // v--- From corrected to corrections
            ql[ n ] -= d.q( i - 1, n );
            qr[ n ] -= d.q( i    , n );
        }   // v--- Project the corrections
        stoi.proj( ql + n_hva, aux );
        stoi.proj( qr + n_hva, aux );
        type:: float_t lim_l( 1 ), lim_r( 1 );
        for( int n = n_hva; n < n_tot; ++ n )
        {   // v--- Then clamp the corrections
            const auto ql0( d.q( i - 1, n ) );
            const auto qr0( d.q( i    , n ) );
            if( __abs1( ql[ n ] ) > __abs1( ql0 ) )
            {
                if( ql0 > 0 )
                    lim_l = __min1
                        ( lim_l, __abs1( ql0 / ql[ n ] ) );
                else
                    lim_l = 0;
            }
            if( __abs1( qr[ n ] ) > __abs1( qr0 ) )
            {
                if( qr0 > 0 )
                    lim_r = __min1
                        ( lim_r, __abs1( qr0 / qr[ n ] ) );
                else
                    lim_r = 0;
            }
        }   // None-negative slope limiter for corrections
        for( int n = n_hva; n < n_tot; ++ n )
        {
            ql[ n ] = d.q( i - 1, n ) + lim_l * ql[ n ];
            qr[ n ] = d.q( i + 0, n ) + lim_r * qr[ n ];
        }
        return;
    };

    template < class dat_T, class bdt_T >
    __device__ void calc( dat_T     & d , const bdt_T & b_d,
                          const int & i ) const
    {
        const  auto & self( * this );
        if( ! ( is_uniform = d.is_uniform ) )
            self.load_geo ( d.xf.at( i ), d.xv.at( i ) );
        if( stoi )
            self.cals_s  ( d, i, b_d );
        else if( ! correction )
            super_t::cals( d, i, b_d );
        else
            self.cals_d  ( d, i, b_d );
        return;
    };
};

};                         // namespace chemistry::hydro
