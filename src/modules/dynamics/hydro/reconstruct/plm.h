#pragma once

#include "../../../../mesh/types/dat_1d.h"
#include "../../prototypes/reconstruct.h"
#include "../../../../utilities/types/my_type_traits.h"

namespace dynamics::hydro::reconstruct
{
////////////////////////////////////////////////////////////
// Types

using mesh::dat_1d_t;
using type:: float_t;

////////////////////////////////////////////////////////////
// PLM reconstruction

template< class derived_T = crtp::dummy_t >
struct plm_t : public crtp::helper_t
< plm_t, derived_T, ::dynamics::reconstruct::interface_t >
{
    __crtp_def_self__( plm_t, derived_T );
    
    ////////// Data //////////
    mutable bool    is_uniform;
    mutable float_t q_ll,  q_l, q_r, q_rr;
    mutable float_t ql_coef,      qr_coef;
    mutable float_t cbr,   cfr, cbl,  cfl;
    mutable float_t dxv_l_i,    dxv_c_i, dxv_r_i;
    mutable float_t dxfr,       dxfl;

    ////////// Spatial Order //////////
    __host__  virtual  int space_order(  ) { return 2; };

    ////////// Operation //////////
    template < class   f_T > __device__ void load_geo
    ( const f_T * xf, const     f_T * xv ) const;
    template< class dat_T, class bdt_T > __device__
    void general( dat_T &, const int &, const int &,
                  const bdt_T & ) const;
    template< class dat_T, class bdt_T > __device__
    void uniform( dat_T &, const int &, const int &,
                  const bdt_T & ) const;    
    template< class dat_T, class bdt_T > __device__
    void cals( dat_T &, const int &,  const bdt_T & ) const;
    template < class dat_T, class bdt_T > __device__ void
    calc( dat_T &, const bdt_T &, const int & ) const;    
};

////////////////////////////////////////////////////////////
// Loading

template< class derived_T >
template< class f_T >  __device__ void plm_t< derived_T >::
load_geo( const f_T * xf, const f_T * xv ) const
{
    const auto dxv_l =   xv[ -1 ] - xv[ -2 ]  ;
    const auto dxv_c =   xv[  0 ] - xv[ -1 ]  ;
    const auto dxv_r =   xv[  1 ] - xv[  0 ]  ;
    cbl      = dxv_l / ( xv[ -1 ] - xf[ -1 ] );
    cfr      = dxv_r / ( xf[  1 ] - xv[  0 ] );
    dxfr     = xf[ 0 ] - xv[ -1 ];
    dxfl     = xv[ 0 ] - xf[  0 ];
    cfl      = dxv_c / dxfr ;
    cbr      = dxv_c / dxfl ;
    dxv_l_i  = 1     / dxv_l;
    dxv_c_i  = 1     / dxv_c;
    dxv_r_i  = 1     / dxv_r;
    return;
}

////////////////////////////////////////////////////////////
// PLM reconstruction

template< class derived_T >
template< class dat_T, class bdt_T >
__device__ void plm_t< derived_T >::general
( dat_T & d, const int & i, const int & n,
  const bdt_T & bdt ) const
{
    auto * ql = d.ql.at( i );
    auto * qr = d.qr.at( i );    
    const auto dql   = ( q_l  - q_ll ) * dxv_l_i;
    const auto dqc   = ( q_r  - q_l  ) * dxv_c_i;
    const auto dqr   = ( q_rr - q_r  ) * dxv_r_i;

    using f_t = type::prim_type_t< decltype( dql ) >;
    f_t norm( 1 );
    if constexpr( std::is_same_v< f_t, float > )
        if( __abs1( dqc ) > type::float_min ||
            __abs1( dqr ) > type::float_min ||
            __abs1( dql ) > type::float_min  )
        {
            norm = __max1( __abs1( dqr ), __abs1( dql ) );
            norm = 1 / __max1( __abs1( dqc ), norm );
        }
    const auto dqr2   = norm * dqr * dqr;
    const auto dql2   = norm * dql * dql;
    const auto dqc2   = norm * dqc * dqc;
    const auto dq2_l  = norm * dql * dqc;
    const auto dq2_r  = norm * dqr * dqc;     
    ql_coef  = dq2_l <= 0 ? 0
        :  dxfr * dq2_l / ( dql2 + dqc2 + dq2_l *
                          ( cfl  + cbl  - 2 ) );
    qr_coef  = dq2_r <= 0 ? 0
        : -dxfl * dq2_r / ( dqr2 + dqc2 + dq2_r *
                          ( cfl  + cbl  - 2 ) );
    ql[ n ] = q_l + ( dq2_l <= 0 ? 0
        : ql_coef * ( cfl * dql + cbl * dqc ) );
    qr[ n ] = q_r + ( dq2_r <= 0 ? 0
        : qr_coef * ( cfr * dqc + cbr * dqr ) );
    return;    
}


template< class derived_T >
template< class dat_T, class bdt_T >
__device__ void plm_t< derived_T >::uniform
( dat_T & d, const int & i, const int & n,
  const bdt_T & bdt ) const
{
    auto * ql = d.ql.at( i );
    auto * qr = d.qr.at( i );
    const auto dql = ( q_l  - q_ll );
    const auto dqc = ( q_r  - q_l  );
    const auto dqr = ( q_rr - q_r  );

    using f_t = type::prim_type_t< decltype( dql ) >;
    f_t norm( 1 );
    if constexpr( std::is_same_v< f_t, float > )    
        if( __abs1( dqc ) > type::float_min ||
            __abs1( dqr ) > type::float_min ||
            __abs1( dql ) > type::float_min  )
        {
            norm = __max1( __abs1( dqr ), __abs1( dql ) );
            norm = 1 / __max1( __abs1( dqc ), norm );
        }
    
    const auto dqr2   = norm * dqr * dqr;
    const auto dql2   = norm * dql * dql;
    const auto dqc2   = norm * dqc * dqc;
    const auto dq2_l  = norm * dql * dqc;
    const auto dq2_r  = norm * dqr * dqc; 
    ql_coef  = dq2_l <= 0 ? 0 :  dq2_l
                      / ( dql2 + dqc2 + dq2_l * 2 );
    qr_coef  = dq2_r <= 0 ? 0 : -dq2_r
                      / ( dqr2 + dqc2 + dq2_r * 2 );
    ql[ n ]  = q_l + ( dq2_l <= 0 ? 0
                        : ql_coef * ( dql + dqc ) );
    qr[ n ]  = q_r + ( dq2_r <= 0 ? 0
                        : qr_coef * ( dqc + dqr ) );
    return;
}

template< class derived_T >
template< class dat_T, class bdt_T >
__device__ void plm_t< derived_T >::cals
( dat_T & d, const int & i, const bdt_T & bdt ) const
{
    for( int n = 0; n < d.q.n_int; ++ n )
    {
        q_ll = d.q( i - 2, n );
        q_l  = d.q( i - 1, n );
        q_r  = d.q( i + 0, n );
        q_rr = d.q( i + 1, n );
        if(  is_uniform )
            get_self(  ).uniform( d, i, n, bdt );
        else
            get_self(  ).general( d, i, n, bdt );
    }
    return;
}

template< class derived_T >
template< class dat_T, class bdt_T >  __device__
void plm_t< derived_T >::calc
( dat_T & d, const bdt_T & bdt, const int & i ) const
{
    const  auto & self = get_self(  );
    is_uniform = d.is_uniform;
    if( ! is_uniform )
        self.load_geo( d.xf.at( i ), d.xv.at( i ) );
    return  self.cals( d, i, bdt );
}

};              // namespace dynamics::hydro::reconstruct
