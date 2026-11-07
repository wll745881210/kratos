#pragma once

#include "fld_base.h"
#include "fld_chi.h"

namespace rad::fld
{

////////////////////////////////////////////////////////////
//  The M1 flux rebuild:  F  from the solved  E  (  the
// Schur - complement elimination backwards ).   The E
// equation:
//      E - ∇·( D_t ∇E ) + sig E = rhs,
//      D_t  = dt^2 c^2 chi / ( 1 + sig_t )
// (  sig_t  the dt - scaled TOTAL extinction;  the M1
// driver freezes it in  co_arr  with  chi ).
//  The eliminated flux:
//      F  = - dt c^2 chi ∇E / ( 1 + sig_t )
//  Discrete:  the cell - centered  chi  (  the row's  own
// frozen values )  and the centered  E  gradient  (  the
// fld_op  dx  convention;  boundary faces use the  Robin
// ghost  so the normal derivative carries the closure ).
//  FLD - limit check:  chi = I / 3
// recovers  F = - dt c^2 ∇E / ( 3 ( 1 + sig_t ) )  -- the
// same face flux the  fld_op  stencil  transports.

struct f_of_e_fun_t
{
    //  co_arr  deployment  (  fld_data.h ):  [ 0 ] D dt,
    //  [ 1 ] sig,  [ 2.. 7 ]  the chi lower  triangle
    //  ( xx, xy, xz, yy, yz, zz )  carrying the  D_t
    // scale  (  dt^2 c^2 chi / ( 1 + sig ) ).
    static constexpr int  i_chi = 2;
    type::float_t        dt;     //  the F = -D_t dE / dt
    int                  accum;  //  +=  (  the F seed
                                 // accumulates on top )

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0_ = geo.n_ceff[ 0 ];
        const int    n1_ = geo.n_ceff[ 1 ];
        const int    n2_ = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0_ * n1_ * n2_ )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0_;
        i[ 1 ] = ( t / n0_ ) % n1_;
        i[ 2 ] =   t / ( n0_ * n1_ );
        const auto * co  = b_d.co_arr.at( i );
        auto       * f_o = b_d.f_arr.at( i );
        //  The centered gradient of  E  per axis  (  the
        // ghost slots close the physical  boundaries ):
        type::float_t g[ 3 ] = { 0, 0, 0 };
        type::idx_t jl( i ), jr( i );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            if( geo.n_ceff[ a ] < 2 )
                continue;
            const auto dx( geo.dx_f( a, i ) );
            jl[ a ] = i[ a ] - 1;
            jr[ a ] = i[ a ] + 1;
            g[ a ] = ( b_d.mgd.fld.at( jr )[ 0 ]
                       - b_d.mgd.fld.at( jl )[ 0 ] )
                     / ( 2 * dx );
            jl[ a ] = i[ a ];
            jr[ a ] = i[ a ];
        };
        //  The eliminated flux  (  physical,  per the
        // moment equation ):   F = - D_t ∇E / dt:
        const auto idt( type::float_t( 1 )
                        / ( dt > 0 ? dt
                          : type::float_t( 1 ) ) );
        for( int a = 0; a < 3; ++ a )
        {
            type::float_t s( 0 );
            for( int b = 0; b < 3; ++ b )
                s += chi_ab( co, a, b ) * g[ b ];
            f_o[ a ] = accum
                ? f_o[ a ] - s * idt
                :         - s * idt;
        };
        return;
    };

    private:
    //  chi_ab  from the lower - triangle  co_arr[ 2..7 ].
    __device__ __forceinline__ static
    type::float_t chi_ab
    ( const type::float_t * co,
      const int & a, const int & b )
    {
        const int idx[ 3 ][ 3 ] = { { 0, 1, 2 },
                                    { 1, 3, 4 },
                                    { 2, 4, 5 } };
        return co[ i_chi + idx[ a ][ b ] ];
    };
};

////////////////////////////////////////////////////////////
//  The M1 freeze:  the per - cell chi from the CURRENT
// ( E, F )  --  the Picard - frozen closure  (  the
// cc_freeze_r  pattern of the nonideal driver ).   F  is
// the  f_arr  rebuilt after the previous  E  solve  ( or
// the initial seed ).   The chi entries are stored  dt
// c^2 - scaled  so the  fld_op  stencil  (  the  c_face
// face mean )  assembles the M1 elliptic operator
// directly.

struct m1_freeze_fun_t
{
    chi_of_e_f            chi;    //  Levermore closure
    type::float_t      dt_c2;    //  dt^2 * c^2  scale
    //  Test mode  (  the T4 anisotropy gate ):  a fixed
    // diagonal  chi  =  diag( ca0, ca1, ca2 )  bypasses
    // the Levermore evaluation.
    int               t_mode;
    type::float_t    ca[ 3 ];

    __host__ m1_freeze_fun_t(  )
    {
        t_mode = 0;
        ca[ 0 ] = ca[ 1 ] = ca[ 2 ]
            = type::float_t( 1 ) / 3;
        return;
    };

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0_ = geo.n_ceff[ 0 ];
        const int    n1_ = geo.n_ceff[ 1 ];
        const int    n2_ = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0_ * n1_ * n2_ )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0_;
        i[ 1 ] = ( t / n0_ ) % n1_;
        i[ 2 ] =   t / ( n0_ * n1_ );
        const auto e( b_d.mgd.fld.at( i )[ 0 ] );
        const auto * f( b_d.f_arr.at( i ) );
        auto * co( b_d.co_arr.at( i ) );
        type::float_t c6[ 6 ];
        if( t_mode == 0 )
            chi.eval( e, f, c6 );
        else
        {
            for( int k = 0; k < 6; ++ k )
                c6[ k ] = 0;
            c6[ 0 ] = ca[ 0 ];   //  xx
            c6[ 3 ] = ca[ 1 ];   //  yy
            c6[ 5 ] = ca[ 2 ];   //  zz
        };
        //  The EXACT Schur tensor:  D_t = dt^2 c^2 chi /
        // ( 1 + sig ),  sig = dt c kappa rho  (  co[ 1 ],  the
        // sig = dt c kappa rho  (  co[ 1 ],  the scalar
        // freeze already wrote it ).   Thick limit
        // ( sig >> 1 )  ->  dt c chi / kappa rho  == the
        // FLD  D dt  with  lambda = chi:
        const auto den( type::float_t( 1 )
                        + co[ 1 ] );
        for( int k = 0; k < 6; ++ k )
            co[ 2 + k ] = dt_c2 * c6[ k ] / den;
        //  co[ 0 ]  (  the FLD D dt )  is left 0  in M1
        // mode  (  the operator never reads it ).
        return;
    };
};

////////////////////////////////////////////////////////////
//  The M1 non - diagonal defect  (  the  nonideal
// couple( )  pattern ):  the base stencil  (  fld_op )
// transports ONLY the diagonal  chi_aa  fluxes;  the
// cross terms  - dt c^2  d_a ( chi_ab d_b E )  with
// a != b  are assembled EXPLICITLY from the current
// iterate  and  subtracted from the solve's source
// (  the Richardson defect - correction ):
//      A_base E = src - A_offdiag E^k
// iterated to the fixed point  (  the driver's Picard
// outer loop,  [fld] n_picard ).

struct m1_couple_fun_t
{
    using f_ab = type::float_t;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0_ = geo.n_ceff[ 0 ];
        const int    n1_ = geo.n_ceff[ 1 ];
        const int    n2_ = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0_ * n1_ * n2_ )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0_;
        i[ 1 ] = ( t / n0_ ) % n1_;
        i[ 2 ] =   t / ( n0_ * n1_ );
        //  The cross - flux divergence  (  the outflow
        // difference of the  fld_op  convention ):
        //      div_a  q_ab,  q_ab = chi_ab d_b E
        // with the face - centered  chi_ab  two - cell
        // mean  (  the  c_face  pattern )  and the one -
        // sided  d_b E  at the face  (  the SAME  d_b E
        // the  f_of_e_fun  rebuild uses at the cells --
        // cell - centered here  --  keeping the couple
        // and the  F  rebuild consistent ).
        type::float_t g[ 3 ] = { 0, 0, 0 };
        {
            type::idx_t jl( i ), jr( i );
            for( int b = 0; b < geo.n_dim; ++ b )
            {
                if( geo.n_ceff[ b ] < 2 )
                    continue;
                const auto dx( geo.dx_f( b, i ) );
                jl[ b ] = i[ b ] - 1;
                jr[ b ] = i[ b ] + 1;
                g[ b ] = ( b_d.mgd.fld.at( jr )[ 0 ]
                           - b_d.mgd.fld.at
                             ( jl )[ 0 ] ) / ( 2 * dx );
                jl[ b ] = i[ b ];
                jr[ b ] = i[ b ];
            };
        };
        const auto * co( b_d.co_arr.at( i ) );
        const int idx[ 3 ][ 3 ] = { { 0, 1, 2 },
                                    { 1, 3, 4 },
                                    { 2, 4, 5 } };
        type::float_t acc( 0 );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            if( geo.n_ceff[ a ] < 2 )
                continue;
            const auto dxa( geo.dx_f( a, i ) );
            for( int k = 0; k < 2; ++ k )
            {
                //  the face neighbor  (  a, k ):
                type::idx_t j( i );
                j[ a ] = k == 0 ? i[ a ] - 1 : i[ a ] + 1;
                const auto * co_j
                    ( b_d.co_arr.at( j ) );
                for( int b = 0; b < 3; ++ b )
                {
                    if( a == b )
                        continue;
                    const f_ab fa
                        ( 0.5 * ( co[ 2 + idx[ a ][ b ] ]
                        + co_j[ 2 + idx[ a ][ b ] ] )
                          * g[ b ] );
                    acc += ( k == 0 ? -1 : 1 )
                           * fa / dxa;
                };
            };
        };
        //  RHS = src - A_off E  = src + dt c^2 div( off ):
        b_d.mgd.src_dat.at( i )[ 0 ] += acc;
        return;
    };
};

////////////////////////////////////////////////////////////
//  The M1 Picard round source reset:  src = s_base  (
// before the  m1_couple  accumulation --  the explicit
// defect must never accumulate across rounds ).

struct m1_srcrst_fun_t
{
    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0_ = geo.n_ceff[ 0 ];
        const int    n1_ = geo.n_ceff[ 1 ];
        const int    n2_ = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0_ * n1_ * n2_ )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0_;
        i[ 1 ] = ( t / n0_ ) % n1_;
        i[ 2 ] =   t / ( n0_ * n1_ );
        b_d.mgd.src_dat.at( i )[ 0 ]
            = b_d.s_base.at( i )[ 0 ];
        return;
    };
};

};                          // namespace rad::fld
