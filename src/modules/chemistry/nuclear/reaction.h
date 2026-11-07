#pragma once

#include "../reaction/reaction_list.h"

namespace chemistry::nuclear::reaction
{
////////////////////////////////////////////////////////////
// Reactions

template< class derived_T = crtp::dummy_t >
struct std_t : ::chemistry:: reaction:: std_t
             < crtp::helper< std_t, derived_T > >
{
    ////////// Types //////////
    using f_t = ::chemistry::reaction::base_t::   f_t;
    using       ::chemistry::reaction::base_t::  pars;
    using       ::chemistry::reaction::base_t::n_pars;
    using super_t = ::chemistry::reaction::std_t
          < crtp :: helper< std_t, derived_T > >;

    __crtp_def_self__( std_t, derived_T );    

    ////////// Host-side functions //////////
    __host__ virtual void setup_pars
    ( const ::chemistry::base_t &   mod,
            std::vector< int > & i_chem, const int & n_reac,
      const std::vector< f_t > &   pars,
      const std::vector< f_t > &   enes,
      const std::vector< f_t > & f_prod )
    {
        super_t::setup_pars
            ( mod, i_chem, n_reac, pars, enes, f_prod );
        std::map < int, int > n_reac_mult;
        f_t  fac ( 1 );           
        for( int i = 0; i < n_reac; ++ i )
        {
            if( n_reac_mult.count( i_chem[ i ] ) == 0 )
                n_reac_mult[ i_chem[ i ] ] = 0;
            ++  n_reac_mult[ i_chem[ i ] ];
            fac *= n_reac_mult[ i_chem[ i ] ];
        }
        this->pars[ 0 ] -= log( fac );
        return;
    };  // 

    ////////// Device-side functions //////////
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate_coef( dat_T     & d, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        const f_t T9   = d.T / 1e9f;
        const f_t T93  = __pow1( T9, 0.33333f );
        const f_t T932 = T93   * T93;
        f_t   res( pars[ 0 ] ), tmp ( 1  / T9 );
        for(  int i = 1; i < n_pars - 1; ++ i )
        {
            res += pars[ i ] * tmp;
            tmp *= T932;
        }
        res +=   pars[ n_pars - 1 ] * logf ( T9 );
        return __exp1( __max1( __min1 ( res, 70 ), -80 ) );
    };

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t dlnk_dlnT( dat_T     & d, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        const f_t T9    = d.T / 1e9f;
        const f_t T93   = __pow1(  T9 ,    0.33333f );
        const f_t T932  = T93   *  T93;
        f_t  res( pars[ n_pars - 1 ] ), tmp( 1 / T9 );
        for( int i = 1; i < n_pars - 1; ++ i )
        {
            res += pars[ i ] * tmp * f_t( 2 * i - 5 ) / 3;
            tmp *= T932;
        }
        return res;
    };
};
};              // namespace chemistry::nuclear::reaction
