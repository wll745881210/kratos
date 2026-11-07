#pragma once

#include "../../../../src/modules/chemistry/chemistry.h"
#include "../../algo/interp.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Enum of cooling kinds

struct mol_cool_t :
       ::chemistry::reaction::std_t< mol_cool_t >
{
    ////////// Type //////////
    using super_t
    = ::chemistry::reaction::std_t< mol_cool_t >;
    using     f_t = super_t::f_t;
    using intp1_t
        = ::extension::interp_t< type::float_t, 1 >;
    using intp2_t
        = ::extension::interp_t< type::float_t, 2 >;

    ////////// Data //////////
    int           i_cool;   // Idx in the cooling intp;

    intp1_t   f_vi_ln_l0;
    intp2_t   f_vi_ln_lte;
    intp1_t   f_ro_ln_l0;
    intp2_t   f_ro_alpha;
    intp2_t   f_ro_ln_lte;
    intp2_t   f_ro_ln_n12;

    ////////// Host-side Function //////////
    __host__ virtual bool parse
    ( const std::string & l , const input & rarg,
      const ::chemistry::base_t & chem );

    __host__ mol_cool_t & operator = ( mol_cool_t & );

    ////////// Device-side Functions //////////

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate_gen ( dat_T       &, const bdt_T & ,
                   const eos_T &, const rls_T & ) const;

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__ f_t
    rate_coef( dat_T       &   d, const bdt_T & b_d ,
               const eos_T & eos, const rls_T & rls ) const
    {
        return d.f_cool[ i_cool ].L( logf( d.T ) );
        // return rate_gen( d, b_d, eos, rls );
    };
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t dlnk_dlnT( dat_T     & d, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        return d.f_cool[ i_cool ].dlnL_dlnT( logf( d.T ) );
    };
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t  ene_reac( dat_T     & d, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {   // Multiplied by N_A for specific energy rate from Y
        // return -1;
        return -phys::cgs::NA;
    };  // specific energy per y abosorbed by rate
};                         // struct mol_cool_t

////////////////////////////////////////////////////////////
// Device-side functions

template< class dat_T, class bdt_T ,
          class eos_T, class rls_T >
__device__ mol_cool_t::f_t  mol_cool_t::rate_gen
( dat_T       &   d, const bdt_T & b_d ,
  const eos_T & eos, const rls_T & rls ) const
{
    const auto v_kps  = 1e-5f * __sqrt1
      ( 2 * phys::cgs::kb * d.T / eos.mcgs( i_chem[ 0 ] ) );
    const auto col = __min1
                   ( d.n_col[ i_chem [ 0 ] ], 1e31f );
    
    const auto lnnt = logf
        ( __max1( col, 1e-31f ) / __max1( v_kps, 1e-31f ) );
    const auto ln_T = logf( d.T );

    const f_t  pars [ 2 ] = { lnnt, ln_T };
    if( __isnan( lnnt + ln_T ) )
        return 1e-31f;

    auto  l0   = exp_( f_vi_ln_l0 ( ln_T ) );
    auto  lte  = exp_( f_vi_ln_lte( pars ) );
    auto l_vi( l0 <= 0 || lte <= 0 ? 0 :
               l0 * lte / ( lte + d.n_tot * l0 ) );
    if( lnnt > f_vi_ln_lte.x_max( 0 ) )
        l_vi *= expf( f_vi_ln_lte.x_max( 0 ) - lnnt );

    l0         = exp_( f_ro_ln_l0 ( ln_T ) );
    lte        = exp_( f_ro_ln_lte( pars ) );
    auto n12   = exp_( f_ro_ln_n12( pars ) );
    auto alpha =       f_ro_alpha ( pars )  ;
    f_t l_ro( 0 );
    if( lte > 0 && l0 > 0 && n12 > 0 && d.n_tot > 0 )
    {
        l_ro = l0 * d.n_tot + lte
             * ( 1 + pow_( d.n_tot / n12, alpha )
                 * ( 1 - n12 * l0 / lte ) );
        l_ro = l0 * lte / l_ro;
    }
    if( lnnt > f_ro_ln_lte.x_max( 0 ) )
        l_ro *= expf( f_ro_ln_lte.x_max( 0 ) - lnnt );

    // return d.n_tot * ( l_vi + l_ro ) * phys::cgs::NA;
    return d.n_tot * ( l_vi + l_ro );
} 

};                   // namespace extension::chemistry
