#pragma once

#include "../../../../utilities/mapping/mapping.h"

namespace dynamics::mhd::eos
{
////////////////////////////////////////////////////////////
// Types

using type::coord_t;

////////////////////////////////////////////////////////////
// MHD adiabatic, constant or variable gamma

template< class derived_T   = crtp:: dummy_t >
struct adiabatic_t : hydro::eos::adiabatic_t
     < crtp::helper< adiabatic_t , derived_T > >
{
    ////////// Types //////////
    using super_t = hydro::eos::adiabatic_t
        < crtp::helper< adiabatic_t , derived_T > >;
    
    ////////// Initialization //////////
    __host__ adiabatic_t(  ) { this->n_hyd = 8; };

    // The device-side operator() strides both the
    // conserved and the primitive staging by
    // hyd.n_fld() ( species ride along through the
    // EoS with the chemistry modules ), so the shared
    // memory must be sized by n_fld() as well. The
    // hydro base sizes its conserved staging by
    // n_hyd, which under-allocates whenever
    // n_fld() > n_hyd ( out-of-bounds __shared__
    // writes; silent for small n_th, a trap at
    // n_th ~ 64 ).
    template< class     blk_T >
    __host__ std::tuple< dim3, dim3, int >
    resource( const blk_T & b_d ) const
    {
        const auto &  hyd = b_d.hyd;
        const int  s_sh_s = hyd.n_fld( )
                            * type::size( hyd.u.dat )
                   +  hyd.w.n_int * type::size( hyd.w.dat );

        const int  n_th = std::min( hyd.u.n_cell[ 0 ],
                                    this->n_th_lim );
        if( b_d.geo.n_ceff[ 0 ] % n_th != 0 )
            throw std::runtime_error
                ( "Indivisible stride, mhd::eos::"
                  "adiabatic_t." );
        const dim3  n_bl ( b_d.geo.n_ceff[ 1 ] ,
                           b_d.geo.n_ceff[ 2 ] ,
                           b_d.geo.n_ceff[ 0 ] / n_th );
        const int   s_sh = n_th * s_sh_s;
        return std::make_tuple( n_bl, dim3( n_th ), s_sh );
    };

    ////////// Fast magnetosonic speed //////////
    template < class f_T > __device__
    f_T c_fms( const f_T *   w, const f_T *  b_2 ,
               const f_T & gam, const int & axis ) const;

    ////////// Time step //////////
    template< class f_T, class bdt_T, class idx_T >
    __device__  f_T dt
    ( const     f_T * v_p, const coord_t &  bc_2 ,
      const coord_t & b_m, const     f_T & gamma ,
      const   idx_T &   i, const   bdt_T &   b_d ) const;

    ////////// Device-side interface //////////
    template< class bdt_T > __device__ void operator(  )
    ( const bdt_T & b_d, const int & step ) const;

    using super_t::cvkb_xn_T_h;
    using super_t::   w_erho_d;
    using super_t::   w_prho_d;
};

////////////////////////////////////////////////////////////
// Fast magnetosonic speed

template< class derived_T >
template< class f_T >
__device__  f_T adiabatic_t< derived_T >::c_fms
( const f_T *   w, const f_T *  b_2 ,
  const f_T & gam, const int & axis ) const
{
    using utils::mod3;
    const auto & rho = w[ idx::rho ];
    const auto  cs_2 = w[ idx::ene ] * gam / rho;
    const auto v_ax2 =   b_2[ axis ]       / rho;
    const auto  v_t2 = ( b_2[ mod3( axis + 1 ) ] +
                         b_2[ mod3( axis + 2 ) ] ) / rho;
    const auto   q_2 = v_ax2  +  v_t2 + cs_2 ;
    const auto   r_2 = v_ax2  +  v_t2 - cs_2 ;
    return sqrt( 0.5f * ( q_2 + sqrt
               ( r_2  *   r_2 + 4  * cs_2 * v_t2 ) ) );
}

////////////////////////////////////////////////////////////
// Timestep

template< class derived_T >
template< class f_T, class bdt_T, class idx_T >
__device__  f_T adiabatic_t< derived_T >::dt
( const     f_T * v_p,  const coord_t &  bc_2 ,
  const coord_t & b_m,  const     f_T & gamma ,
  const   idx_T &   i,  const   bdt_T &   b_d ) const
{
    f_T      dt (  FLT_MAX );
    coord_t  bm_2     = bc_2;
    for( int a = 0; a < b_d.geo.n_dim; ++ a )
    {
        bm_2[ a ] = b_m [ a ] * b_m[ a ];
        const auto c_fms = this->c_fms
                         ( v_p, bm_2.x, gamma, a ) ;
        dt  = __min1( dt,  b_d.geo.dx_cfl( a, i ) /
            ( c_fms + __abs1 ( v_p[ idx::mom + a ] ) ) );
        bm_2[ a ] = bc_2[ a ];
    }
    return dt * this->cfl;
}

////////////////////////////////////////////////////////////
// Device-side interface

template < class  derived_T >
template < class      bdt_T >  __device__
void adiabatic_t< derived_T >::operator(  )
( const bdt_T & b_d, const int & step ) const
{
    const auto & self = this->get_self(  );

    //////////////////////////////////////////////
    // Numbers and indices
    const int i_th  =       threadIdx.x;
    const int n_th  =        blockDim.x;
    const int n_var = b_d.hyd.n_fld(  );
    const int d_i   =      i_th * n_var;

    //////////////////////////////////////////////
    // Dispatching dynamic shared memory and
    // access the conservatives / primitives
    using f_t  = typename decltype( b_d.fld ):: f_t;
    using f2_t = typename decltype( b_d.fld )::f2_t;
    __dyn_shared__( char,      var_sh );
    auto *  dt =  (  f_t * ) ( var_sh );
    auto * var =  ( f2_t * ) ( var_sh );
    auto * v_c = var + d_i;
    auto * v_p =  (  f_t * ) ( var + n_th * n_var ) + d_i;

    idx_t i;
    self.dispatch( i );
    auto *   p = (  f_t * ) b_d.hyd.w.at( i       );
    auto *   c = ( f2_t * ) b_d.hyd.u.at( i, step );

    //////////////////////////////////////////////////
    // Magnetic fields: Face-centered to cell-centered
    auto  * bf    = b_d.fld.bf.at( i, step );
    coord_t bc_2, b_max, bc;

    f2_t e_m ( 0 );
    for( int a = 0;  a < bc.size(  ); ++ a )
    {
        const auto   b_l  = bf [ a ];
        //  A one-cell axis has no +side face slot of its
        // own: the read below would land in a ghost that
        // no exchange ever fills ( no neighbor exists
        // along a thin axis ). Both faces of the cell are
        // the same periodic face, so the center is b_l.
        const bool   thin ( b_d.geo.n_ceff[ a ] < 2 );
        if( ! thin )
            ++ i [ a ];
        const auto   b_r  = thin ? b_l
                        : b_d.fld.bf.at( i, step )[ a ];
        if( ! thin )
            -- i [ a ];
        const auto   bc_m = ( b_l + b_r ) / 2;
        bc   [ a ] = bc_m;
        const auto   bcs2 = bc_m * bc_m;
        e_m       += bcs2;
        if( step  != 0 )        // Following two for dt
            continue;
        bc_2 [ a ] = bcs2;
        b_max[ a ] = __max1( __abs1( b_l ), __abs1( b_r ) );
    }
    e_m /= 2;

    //////////////////////////////////////////////
    // Load, convert, write-back, and in case of floor
    // The new c2p contract reads cons from c_src and
    // writes floor corrections back to c_src, so the
    // magnetic energy is excluded from the global u
    // around the call (per-cell thread ownership).
    for( int  n = 0; n < n_var; ++ n )
         v_c[ n ] =  c [ n ];
    c[ idx::ene ] -=   e_m;

    self.c2p( v_p, v_c, p, c );
    c[ idx::ene ] +=   e_m;
    const auto gamma = this->gamma;
    for( int n = 0; n < n_var; ++ n )
         p[ n ] = v_p[ n ];
    for( int a = 0; a < bc.size(  ); ++ a )
    {
        p[ idx::bcc + a ] = bc[ a ];
        //  Keep the conserved cell-centered B in sync with
        // the CT face fields ( the chemistry load reads the
        // magnetic energy from these slots; pure MHD never
        // does, so this is invisible there ).
        c[ idx::bcc + a ] = bc[ a ];
    }

    //////////////////////////////////////////////////
    // Cell-centered electric field
    auto * v  = v_p   +   idx::mom;
    auto * ec = b_d.fld.ec.at( i );
    ec[ 0 ] = bc[ 1 ] * v[ 2 ] - bc[ 2 ] * v[ 1 ];
    ec[ 1 ] = bc[ 2 ] * v[ 0 ] - bc[ 0 ] * v[ 2 ];
    ec[ 2 ] = bc[ 0 ] * v[ 1 ] - bc[ 1 ] * v[ 0 ];

    //////////////////////////////////////////////////
    // Calculate dt
    __syncthreads(  );
    if( step == 0 )
    {
        dt[ i_th ] = self.dt
                 ( v_p, bc_2, b_max, gamma,  i, b_d );
        utils::block_reduce
                 < f_t, __min1 >( dt,    blockDim.x );
        if( i_th == 0 )
            utils::atomic_min( b_d.p_dt, dt[ 0 ] );
    }
    return;
}
};                          // namespace dynamics::mhd::eos
