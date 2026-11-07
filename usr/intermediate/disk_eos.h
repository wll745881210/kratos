#pragma once

#include "../../src/modules/dynamics/mhd/nonideal/explicit_sts/nonideal.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  Cold - flow pressure floor for the implicit non - ideal
//  MHD leg.
//
//  The stock EOS floor is ABSOLUTE  ( p >= tmp_floor * rho,
//  tmp_floor = 1e-9 ):  in a cold supersonic wind the
//  thermal pressure is a 1e-4 fraction of the total energy,
//  the conservative truncation error drives p through zero
//  and the floor is numerically zero --  c_s collapses to
//  0,  and the next HLLD step explodes  ( the sp32
//  long - run crash ).   This wrapper adds a floor
//  RELATIVE to the kinetic energy,  p >= eps_p * rho v^2/2,
//  and heals the conserved total energy to match,  so a
//  floored cell is consistent instead of permanently
//  pressureless.   eps_p = 0  ( the default ) is
//  bit - identical to the stock behaviour.
//
//  [dynamics] p_floor_eps = 1e-6 .. 1e-3
//
//  Usage  ( a pgen / usr/intermediate consumer ):
//      interm::ni_eos_pfloor_t< ni_n::prof_eta_t >
//  in place of  ni_n::ni_eos_t< ni_n::prof_eta_t >  --
//  the CRTP derived_T carries the override,  so the
//  hydro base 's  self.c2p( ... )  lands here.
////////////////////////////////////////////////////////////

template< class eta_T >
struct ni_eos_pfloor_t
    : dynamics::mhd::nonideal::ni_eos_t
      < eta_T, ni_eos_pfloor_t< eta_T > >
{
    using this_t  = ni_eos_pfloor_t< eta_T >;
    using super_t = dynamics::mhd::nonideal::ni_eos_t
                  < eta_T, this_t >;

    type::float_t eps_p;

    __host__ ni_eos_pfloor_t(  )
    {
        eps_p = 0;
        return;
    };

    __host__ virtual void read( const input & args )
    {
        super_t::read( args );
        eps_p = args.get< type::float_t >
              ( "dynamics", "p_floor_eps", 0 );
        return;
    };

    //  Called from the hydro base 's cons2prim with the
    //  magnetic energy already removed from  c_src .
    template< class f_T, class f2_T > __device__
    f_T c2p( f_T * prim, f2_T * cons, f_T * p_src,
             f2_T * c_src ) const
    {
        f_T cs2 = super_t::c2p( prim, cons, p_src, c_src );
        if( eps_p <= f_T( 0 ) )
            return cs2;
        f_T e_k( 0 );
        for( int a = 0; a < 3; ++ a )
            e_k += prim[ idx::mom + a ] * prim[ idx::mom + a ];
        const f_T rho( prim[ idx::rho ] );
        e_k *= f_T( 0.5 ) * rho;
        f_T & pre( prim[ idx::ene ] );
        const f_T fl( eps_p * e_k );
        if( pre < fl )
        {
            pre = fl;
            //  Heal the conserved gas energy  ( the
            //  caller re - adds the magnetic part ).
            c_src[ idx::ene ] = f2_T
                ( e_k + pre / ( this->gamma - 1 ) );
            p_src[ idx::ene ] = pre;
            cs2 = this->gamma * pre / rho;
        }
        return cs2;
    };
};

};                          // namespace interm
