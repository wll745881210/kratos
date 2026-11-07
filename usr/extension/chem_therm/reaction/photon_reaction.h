#pragma once

#include "../../../../src/modules/chemistry/chemistry.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Photochemical reactions

struct list_t;

struct phot_reac_t
     : ::chemistry::reaction::std_t< phot_reac_t >
{
    ////////// Type //////////
    using    super_t
     = ::chemistry::reaction::std_t< phot_reac_t >;
    using        f_t = super_t::f_t;
    enum   sh_kind_t { none, hydrogen, fitted };
    struct  shield_t
    {
        int    i_sp;  // Diff. from i_chem[ 0 ] w/ cross-sh
        f_t  N1, N2;
        f_t   a,  b;
        f_t   x,  y;

        __device__ __forceinline__
        f_t operator(  ) ( const f_t & N_sps ) const
        {
            auto N( utils::max( N_sps, 0 ) ) ;
            N = x * __exp1 ( - __pow1( N / N1, a ) ) +
                y * __exp1 ( - __pow1( N / N2, b ) ) ;
            return utils::max( utils::min( N,  1 ) , 0 );
        };
    };

    //////////  Data //////////
    int            i_phot;   // Idx in the photo-reac array
    sh_kind_t     sh_kind;   // Type of shielding formula
    int              i_sh;   // The index in the sh arr
    int          n_shield;   // Shielding by diff species
    int  i_hv_sh_lim[ 2 ];   // Limits of shielding hv
    f_t          *   mult;   // Multiplier  at each hv
    shield_t     * shield;   // Shielding data

    ////////// Host-side Functions //////////
    // __host__ phot_reac_t(  );
    __host__ virtual bool parse_ph
    ( list_t       &, const std::string &,
      const  input &, const ::chemistry:: base_t & );

    __host__ virtual void setup             ( void   * p0 );
    __host__ virtual void relocate_shifted  ( void   * p0 );
    __host__ virtual size_t  set_shift( const size_t &  s );
    __host__ virtual size_t  size     (                   );

    __host__ virtual void   print
    ( const ::chemistry::sp_vec_t & );

    ////////// Device-side Functions //////////
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    void prepare( dat_T       &, const bdt_T & ,
                  const eos_T &, const rls_T & ) const;
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t dlnk_dlnT( dat_T       &, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        return 0;
    };

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate_coef
    ( dat_T     & d, const bdt_T & ,
      const eos_T &, const rls_T & ) const
    {
        return d.p_cache[ i_phot ].zeta;
    };    
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t  ene_reac( dat_T     & d, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        return d.p_cache[ i_phot ].ene;
    };

    template < class  idx_T, class bdt_T, class eos_T >
    __device__ __forceinline__ f_t ext_mod
    ( const int   &, const idx_T & ,
      const bdt_T &, const eos_T & ) const;
};

template< class dat_T, class bdt_T ,
          class eos_T, class rls_T >
__device__ void   phot_reac_t::prepare
( dat_T       &   d, const bdt_T & b_d ,
  const eos_T & eos, const rls_T & rls ) const
{
    const auto *   flx = d.flx;
    const auto ext_mod( i_sh < 0 ?  1 : b_d.
                        rad.shield .at( d.idx )[ i_sh ] );

    auto & cache( d.p_cache[ i_phot ] );
    cache.zeta = 0;
    cache. ene = 0;
    for( int i_hv = 0; i_hv < rls.n_hv; ++ i_hv )
    {
        const auto & flx_eff( flx[ i_hv ] * rls.flx_conv );
        auto zeta_s = flx_eff * pars[ i_hv ]
            * ( i_hv  >= i_hv_sh_lim[ 0 ] &&
                i_hv  <= i_hv_sh_lim[ 1 ] ? ext_mod :  1 );
        cache.ene  += zeta_s * enes[ i_hv ] ;
        cache.zeta += zeta_s * mult[ i_hv ] ;
    } // v-- From erg per reaction to erg per molar abund Y
    cache.ene /= utils::max( cache.zeta, 1e-32f );
    cache.ene *= phys::cgs::NA;
    return;
}

template< class idx_T, class bdt_T, class eos_T >
__device__ __forceinline__ phot_reac_t::f_t
phot_reac_t::ext_mod
( const int   & i_hv, const idx_T & idx ,
  const bdt_T &  b_d, const eos_T & eos ) const
{
    if( i_hv < i_hv_sh_lim[ 0 ] || i_hv > i_hv_sh_lim[ 1 ] )
        return 1 ;
    const auto & unit( eos.unit );
    const auto * sp_col = b_d.rad.sp_col.at( idx );
    const auto col_conv = ( unit.rho0 / phys::cgs::ma )
                          * unit.  l0 ;
    if( sh_kind == hydrogen )
    {
        auto x( col_conv * sp_col[ i_chem[ 0 ] ] / 5e14f );
        auto T  = b_d.T_hyd( idx );
        auto b5 = 0.091f * __sqrt1( utils::max( T, 2.7f ) );
        const auto xb5p1  ( 1 + x /  b5 );
        const auto sqrt1px( __sqrt1( 1 + x ) );
        return 0.965 / ( xb5p1 *  xb5p1 ) + 0.035f
             / sqrt1px * __exp1( -8.5e-4f * sqrt1px );
    }   // Draine (2011), eq. 31.27
    else if( sh_kind == fitted )
    {
        f_t res( 1 );
        for( int n = 0; n < n_shield; ++ n )
        {
            const auto & i_sp = shield[ n ].i_sp;
            res *= shield[ n ]( col_conv * sp_col[ i_sp ] );
        }
        return res;
    }
    return 1;
}

};                      // namespace extension::chemistry
