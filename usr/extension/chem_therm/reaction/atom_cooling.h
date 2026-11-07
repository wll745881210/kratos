#pragma once

#include "../../../../src/modules/chemistry/chemistry.h"
#include "../../algo/interp.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Atomic cooling processes

struct atom_cool_t :
       ::chemistry::reaction::std_t< atom_cool_t >
{ 
    ////////// Type //////////
    using super_t
    =  ::chemistry::reaction::std_t< atom_cool_t >;
    using     f_t = super_t ::f_t;

    ////////// Data //////////
    int        i_cool;   // Idx in the cooling intp;
    
    ////////// Host-side Function //////////
    __host__ virtual bool parse
    ( const std::string & l , const input & rarg,
      const ::chemistry::  base_t & chem );
    __host__ virtual void print
    ( const ::chemistry::sp_vec_t &   );
    
    ////////// Device-side Functions //////////
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t     nu_nt( dat_T       &, const bdt_T & ,
                   const eos_T &, const rls_T & ) const;
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
               class eos_T, class rls_T > __device__ f_t
    dlnk_dlnT( dat_T       &   d, const bdt_T & b_d ,
               const eos_T & eos, const rls_T & rls ) const
    {
        return d.f_cool[ i_cool ].dlnL_dlnT( logf( d.T ) );
        // const auto theta = d.T / pars[ 2 ];
        // return theta < 1e-2f
        //      ? 0 : nu_nt( d, b_d, eos, rls ) / theta;
    }
    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t  ene_reac( dat_T       &, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        // return -1;
        return -phys::cgs::NA;
    };  // specific energy per y abosorbed by rate
};                         // struct atom_cool_t

////////////////////////////////////////////////////////////
// Device-side functions

template< class dat_T, class bdt_T ,
          class eos_T, class rls_T > __forceinline__
__device__ atom_cool_t::f_t  atom_cool_t::nu_nt
( dat_T       &   d, const bdt_T & b_d ,
  const eos_T & eos, const rls_T & rls ) const    
{   //  pars[ 3 ] := n_crit_n  ; pars[ 4 ] := n_crit_e
    f_t n_ratio_sum = d.n_tot  / pars[ 3 ]
                    + d.n_elec / pars[ 4 ];
    //  pars[ 1 ] := g_u / g_l ; pars[ 2 ] := E / kb 
    const auto nu_nl = exp_        ( -pars [ 2 ] / d.T )
        * pars [ 1 ] * n_ratio_sum / ( 1 + n_ratio_sum );
    return nu_nl / ( 1 + nu_nl );
}

template< class dat_T, class bdt_T ,
          class eos_T, class rls_T > __forceinline__
__device__ atom_cool_t::f_t  atom_cool_t::rate_gen
( dat_T       &   d, const bdt_T & b_d ,
  const eos_T & eos, const rls_T & rls ) const      
{
    const auto nu_nt = this->nu_nt( d, b_d, eos, rls );
    ////////////////////////////////////////
    // Escape probability estimation
    const auto v_kps  = 1e-5f * __sqrt1
      ( 2 * phys::cgs::kb * d.T / eos.mcgs( i_chem[ 0 ] ) );
    // pars[ 5 ] := Reference column density
    const auto tau_0
        = pars[ 5 ] * d.n_col[ i_chem[ 0 ] ] / v_kps;
    static const f_t tiny_tau( 1e-2f );
    f_t f_e ( 1 ) ; // Escape probability (Kwan & Krolik)
    if( tau_0 > 1e35f || __isinf( tau_0 ) ||
        __isnan( tau_0 ) )
        f_e = 0;
    else if( tau_0 > 1 )
    {
        const auto b = __sqrt1( logf( tau_0 ) )
                       / ( 1  + tau_0 /  1e5f );
        f_e = 1 / ( 1.77245f * tau_0 * ( 1.2f + b ) );
    }
    else if ( tau_0 > tiny_tau )
        f_e = ( 1 - exp_( -2 * tau_0 ) ) / ( 2 * tau_0 );
    ////////////////////////////////////////
    // return enes[ 0 ] * phys::cgs::NA * f_e
    //      * pars[ 0 ] * nu_nt; // pars[ 0 ] := Einstein A
    return enes[ 0 ] * f_e * pars[ 0 ] * nu_nt;
}   // Multiplied by N_A for specific energy rate from Y;

};                     // namespace extension::chemistry
