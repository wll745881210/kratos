#pragma once

#include "fld_base.h"

namespace rad::fld
{

////////////////////////////////////////////////////////////
//  The M1 Eddington - tensor evaluation  (  Levermore 1984
// maximum - entropy closure ).   The chi entries  (  six
// independent,  symmetric,  the lower - triangle order
// xx, xy, xz, yy, yz, zz )  are the deployment convention
// of  co_arr[ 2 .. 7 ].
//    chi( f ) = D_K( f ) n n +  ( 1 - D_K ) / 2 ( I - n n )
// with  n = F / |F|  and  the Kershaw diagonal
//    D_K( f ) = ( 3 + 4 f^2 )
//            / ( 5 + 2 sqrt( 4 - 3 f^2 ) )
// (  chi -> I / 3  as  f -> 0;   chi -> n n  as  f -> 1 ).
//  REALIZABILITY:  f = |F| / ( c E )  is clamped to
// [ 0, 1 - 1e-4 ]  BEFORE the evaluation  (  |F| > cE  is
// unphysical  ( a moment - closure violation );  the clamp
// keeps the closure well - posed  ( chi stays finite and
// realizable )  instead of NaN - ing the solve.
//  Units:  F and E in the same  ( code x lum_sc )  units;
// c_l is the  ( possibly reduced )  light speed in code
// units.

struct chi_of_e_f
{
    type::float_t   c_l;    //  light speed,  code units
    type::float_t   f_eps;  //  realizability clamp

    __host__ chi_of_e_f(  )
    {
        c_l   = 1;
        f_eps = 1e-4;
        return;
    };

    //  The six lower - triangle chi entries at  ( E, F ).
    template< class f_T >
    __device__ __forceinline__ void eval
    ( const f_T & e, const f_T * f,
      f_T ( & chi )[ 6 ] ) const
    {
        f_T fx( f[ 0 ] ), fy( f[ 1 ] ), fz( f[ 2 ] );
        const f_T f_abs
            ( __sqrt1( fx * fx + fy * fy + fz * fz ) );
        //  f  ( the reduced flux )  with the clamp:
        const f_T e_c
            ( __max1( e, f_T( f_eps ) ) );
        f_T fred( f_abs / ( c_l * e_c ) );
        if( fred > f_T( 1 ) - f_eps )
            fred = f_T( 1 ) - f_eps;
        //  Kershaw diagonal + the beam direction:
        const f_T f2( fred * fred );
        const f_T d_k
            ( ( f_T( 3 ) + f_T( 4 ) * f2 )
              / ( f_T( 5 ) + f_T( 2 ) * __sqrt1
                  ( f_T( 4 ) - f_T( 3 ) * f2 ) ) );
        f_T nx( 0 ), ny( 0 ), nz( 0 );
        if( f_abs > f_T( f_eps ) )
        {
            const f_T inv( f_T( 1 ) / f_abs );
            nx = fx * inv;  ny = fy * inv;  nz = fz * inv;
        };
        //  chi = d_k n n  +  ( 1 - d_k ) / 2  ( I - n n ):
        //  the diagonal entry  a a  is
        //      d_k n_a^2 + ( 1 - d_k ) ( 1 - n_a^2 ) / 2
        //  and the off - diagonal  a b:
        //      ( 3 d_k - 1 ) / 2 * n_a n_b.
        const f_T o_d( ( f_T( 3 ) * d_k - f_T( 1 ) )
                       / f_T( 2 ) );
        chi[ 0 ] = d_k * nx * nx
            + ( f_T( 1 ) - d_k ) * ( f_T( 1 ) - nx * nx )
              / f_T( 2 );
        chi[ 1 ] = o_d * nx * ny;
        chi[ 2 ] = o_d * nx * nz;
        chi[ 3 ] = d_k * ny * ny
            + ( f_T( 1 ) - d_k ) * ( f_T( 1 ) - ny * ny )
              / f_T( 2 );
        chi[ 4 ] = o_d * ny * nz;
        chi[ 5 ] = d_k * nz * nz
            + ( f_T( 1 ) - d_k ) * ( f_T( 1 ) - nz * nz )
              / f_T( 2 );
        return;
    };
};

};                          // namespace rad::fld
