#pragma once

#include "../../../../device/device.h"
#include "../../../../mesh/block/block.h"
#include "../../../../mesh/types/dat_1d.h"
#include "../../../../mesh/boundary/boundary.h"
#include "../../../../utilities/types/crtp.h"
#include "../../indices.h"
#include "../types/block_data.h"
#include <algorithm>
#include <tuple>

namespace dynamics::hydro::boundary
{
////////////////////////////////////////////////////////////
// Types

using             namespace mesh;
using             type:: float_t;
using mesh::boundary::holder_p_t;

////////////////////////////////////////////////////////////
// Base class for physical boundaries

template< class     prx_T = dynamics::hydro::proxy_t ,
          class derived_T = crtp::           dummy_t >
struct phys_base_t : mesh::boundary::base_t
{
    ////////// Types: CRTP functions //////////
    __crtp_def_self__( phys_base_t, derived_T, prx_T );
    
    ////////// Flags and indicators //////////
    int       side;
    bool copy_body;

    ////////// Constructor and initialzier //////////
    __host__ phys_base_t(  )
        : copy_body( true ), side( -1 ) {  };
    
    ////////// Resource and dispatching //////////
    template< class bdt_T > __host__ std::tuple
    < dim3, dim3, int > resource ( const bdt_T & ) const;
    __host__ void launch( holder_p_t &,  const int &,
                          device::base_t & ) const;

    ////////// Device-side interface //////////
    template< class bdt_T > __device__ void operator (  )
    ( const bdt_T & b_d, const int & step ) const;
};

////////////////////////////////////////////////////////////
// Host-side preparations

template< class prx_T , class derived_T >
template< class bdt_T > __host__
std::tuple< dim3, dim3, int >
phys_base_t< prx_T, derived_T >::resource
( const bdt_T & b_d ) const
{
    using f_t  = typename decltype( b_d.hyd )::f_t;
    const int  axis  = int( side  / 2 );
    auto  &    geo   = b_d.geo;
    const int  n_var = b_d.hyd.n_fld(   );
    const int  n_gh  = geo.n_gh;
    const int  n_bl  = geo.n_ceff[ ( axis + 2 ) % 3 ];
    const int  n_ce  = geo.n_ceff[ ( axis + 1 ) % 3 ];

    const int  n_th  = std::min( n_ce, n_th_lim );
    if( n_ce % n_th != 0 )
        throw std::runtime_error( "Indivisible stride, b"
                                  "oundary::phys_base_t." );
    int s_sh = n_th * 2 * n_gh * n_var * sizeof( f_t );
    return std::make_tuple( dim3( n_bl, n_ce /  n_th, 1 ),
                            dim3( n_th ), s_sh );
}

template < class  prx_T, class derived_T > __host__
void phys_base_t< prx_T, derived_T >::launch
( holder_p_t &  h, const int &  step,
  device::base_t & dev ) const
{
    const auto & b_d = prx_T::ref( h.p_d->d(  ) );
    const auto & self( get_self(  ) );
    dev.launch( mesh::boundary::phy_ker, self.resource
              ( b_d ), h.stream, b_d, self, step );
    dev.event_record     ( h.event,       h.stream );
    return dev.event_wait( h.p_d->stream, h. event );
}

////////////////////////////////////////////////////////////
// Device-side interface

template< class prx_T , class derived_T >
template< class bdt_T >  __device__
void phys_base_t< prx_T, derived_T > ::operator(  )
( const bdt_T & b_d, const int & step ) const
{
    __dyn_shared__( char, shared_mem );
    const int   axis = side >> 1;
    const bool right = side  & 1;

    const int  ax_p1 = utils::mod3( axis + 1 );
    const int  ax_p2 = utils::mod3( axis + 2 );
    const int n_ceff = b_d.geo .n_ceff[ axis ];
    const int  n_var = b_d.hyd .w.n_int;
    const int   n_gh = b_d.geo .n_gh;
    const int   n_th = blockDim.x;
    const auto  &  w = b_d. hyd.w;

    int i_3d[ 3 ];
    i_3d[ ax_p1 ] =  threadIdx.x + blockIdx.y * n_th ;
    i_3d[ ax_p2 ] =   blockIdx.x;

    using  f_t = typename decltype( b_d.hyd )::f_t;        
    dat_1d_t< f_t > buf;
    buf.init( ( f_t * ) ( shared_mem ) + n_gh * n_var
              * 2 * threadIdx.x, 0, n_var, n_gh );

    if( copy_body )
        for( int i = 0; i < n_gh; ++ i )
        {
            i_3d[ axis ] = ( right ? n_ceff - i - 1 : i );
            auto * p     = w.at( i_3d );
            auto * v     = buf.at(  i );
            for( int n = 0;   n < n_var; ++ n )
                v[ n ] = p[ n ];
        }    
    get_self(  ).hydro( buf, i_3d, b_d );
    for( int i = -n_gh; i < 0; ++ i )
    {
        i_3d[ axis ] = ( right ? n_ceff - i - 1 : i );
        auto * v     = buf.at(  i );
        auto * p     = w.at( i_3d );
        for( int n = 0;   n < n_var; ++ n )
            p[ n ] = v[ n ];
    }
    return;
}
};  // namespace dynamics::hydro::boundary
