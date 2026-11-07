#pragma once

#include "../../../../src/modules/chemistry/chemistry.h"
#include "reaction.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// General reactions

struct list_t;

struct dust_reac_t
     : ::chemistry::reaction::std_t< dust_reac_t >
{
    ////////// Type //////////
    using  super_t
    =  ::chemistry::reaction::std_t< dust_reac_t >;
    using f_t = super_t::f_t;
    enum   kind_t { thermal, surface, sublimation };

    ////////// Data //////////
    kind_t         kind;

    ////////// Functions //////////
    __host__ dust_reac_t(  ) : kind( surface ) {  };

    __host__ virtual bool parse_gr
    ( const list_t &, const std::string         & ,
      const input  &, const ::chemistry::base_t & );
    __host__ virtual bool parse_thermal
    ( const list_t &, const ::chemistry::species_t & ,
      const ::chemistry::base_t & );
    __host__ virtual bool parse_sublimation
    ( const list_t &, const std::string         & ,
      const input  &, const ::chemistry::base_t & );
    
    ////////// Device-side Functions //////////
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate_coef( dat_T       &, const bdt_T & ,
                   const eos_T &, const rls_T & ) const;
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t dlnk_dlnT( dat_T       &, const bdt_T & ,
                   const eos_T &, const rls_T & ) const;
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t  ene_reac( dat_T       &, const bdt_T & ,
                   const eos_T &, const rls_T & ) const;

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate( dat_T       &   d, const bdt_T & bdt ,
              const eos_T & eos, const rls_T & rls ) const
    {
        f_t  res = get_self(  ).rate_coef
                 ( d, bdt, eos, rls );
        for( int i = 0; i < this->n_reac; ++ i )
             res *=  d. y [ this->i_chem [ i ] ]
                 * ( i > 0 ? d.rho_cgs : 1 );
        if( res == 0 )
            return 0 ;
        if( d.i_en > 0 )
        {
            const auto e_coef = get_self(  ).ene_reac
                              ( d , bdt, eos, rls );
            if constexpr( std::is_same_v< f_t, float > )
            {
                constexpr auto  f_max
                    = type::float_max  / 100 ;
                const auto tst = __abs1( res ) > 1 ?
                    __abs1( e_coef / ( f_max / res ) ) :
                    __abs1( e_coef /   f_max * res ) ;
                if( tst > 1 )
                    res = __abs1( f_max / e_coef )
                        * utils:: sgn ( res );
            }
            d.f_cache( d.i_en, res * e_coef );
            if( kind == thermal )
                atomicAdd( d.de_dt_d, res * e_coef );
        }
        for( int i = 0 ; i < n_chem; ++ i )
            d.f_cache( i_chem[ i ], res * f_chem[ i ] );
        return res;
    };
};                              // strcut dust_reac_t

template< class dat_T, class bdt_T ,
          class eos_T, class rls_T > __forceinline__
__device__ dust_reac_t::f_t  dust_reac_t::rate_coef
( dat_T       &   d, const bdt_T & b_d ,
  const eos_T & eos, const rls_T & rls ) const
{
    auto res = super_t::rate_coef( d, b_d, eos, rls );
    if( kind == thermal )
    {   // pars[ 0 ] is sigma; repair the fixed 300 K 
        res *= __sqrt1( d.y[ d.i_en ] * 300 /    d.T );
        res *= d.n_tot * ( d.T_dust - d.T ) / d.T_dust;
    }   // Note that d.y[ d.i_en ] is already the CGS of
        // some "velocity^2" (RMS), close enough to <v>^2;
        // The extra "/ d.T_dust" is to guarantee that
        // ene_reac returns de/dY
    return res;
}

template< class dat_T, class bdt_T ,
          class eos_T, class rls_T > __forceinline__
__device__ dust_reac_t::f_t  dust_reac_t::dlnk_dlnT
( dat_T       &   d, const bdt_T & b_d ,
  const eos_T & eos, const rls_T & rls ) const    
{
    if( kind == thermal )
        // dlnk_dlnT cap for the Jacobian: 1e-2 floor allowed
        // magnitudes up to ~100, near-singularizing the implicit
        // solve wherever T ~ T_dust (every plateau cell).  A 1e-1
        // floor (gain 10) was tested (bistab T-C): trajectory
        // invariant, wall -25% on the worst DI-crawl condition,
        // di_lu -22%; not adopted, all reference results use 1e-2.
        return 0.5f - utils::sgn( d.T_dust - d.T ) /
            __max1( __abs1( d.T_dust - d.T ) / d.T, 1e-2 );
    return super_t::dlnk_dlnT( d, b_d, eos, rls );
}

template< class dat_T, class bdt_T ,
          class eos_T, class rls_T > __forceinline__
__device__ dust_reac_t::f_t  dust_reac_t::ene_reac
( dat_T       &   d, const bdt_T & b_d ,
  const eos_T & eos, const rls_T & rls ) const        
{
    return ( kind ==  thermal
             ? phys::cgs::R * d.T_dust
             : super_t::ene_reac( d, b_d ,eos, rls ) );
}

};                     // namespace extension::chemistry
