#pragma once

#include "../../../../device/device.h"
#include "../../../../mesh/block/block.h"
#include "../../../../mesh/types/dat_1d.h"
#include "../../../../mesh/boundary/boundary.h"
#include "../../../../utilities/functions.h"
#include "../../../../utilities/types/crtp.h"
#include "../../indices.h"
#include "../types/block_data.h"
#include <tuple>

namespace dynamics::mhd::boundary
{
////////////////////////////////////////////////////////////
// Types

using             namespace mesh;
using             type:: float_t;
using mesh::boundary::holder_p_t;

////////////////////////////////////////////////////////////
// Kernel

template< class bdt_T, class bnd_T >
__global__ void fld_ker( const bdt_T bdt, const bnd_T bnd )
{
    bnd.field( bdt );
    return;
}

////////////////////////////////////////////////////////////
// Base class for physical boundaries

template< class     prx_T = dynamics::hydro::proxy_t ,
          class derived_T = crtp::           dummy_t >
struct phys_base_t : mesh::boundary::base_t
{
    __crtp_def_self__( phys_base_t, derived_T )

    ////////// Flags and indicators //////////
    int       side;
    bool copy_body;

    //  Derived functors may set this true to also pin the
    //  normal face component on the boundary face itself
    //  ( e.g. B_theta = 0 on the polar axis -- a gauge
    //  slot that CT deliberately skips but c2p reads, and
    //  which the radial boundaries would otherwise fill
    //  with copied interior values at corner columns ).
    static constexpr bool pin_bnd_face = false;

    ////////// Constructor and initialzier //////////
    __host__ phys_base_t(    )
           : copy_body( true ), side( -1 ) {  };

    ////////// Resource and dispatching //////////
    template< class bdt_T > __device__ __host__
    int sh_per_th( const bdt_T & b_d )  const;

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
template< class bdt_T >
__host__ std::tuple< dim3, dim3, int >
phys_base_t< prx_T, derived_T >::resource
( const bdt_T & b_d ) const
{
    const int axis   = int       ( side / 2 );
    auto  &   geo    = b_d.geo;
    //  A thin transverse axis carries NO ghost rows in
    //  hyd.w  ( hyd_data:  n_gh[ a ] = n_ceff[ a ] > 1 ?
    //  order : 0 )  --  the 2.5D grids:  zero its band.
    const int n_gh_a = ( geo.n_ceff[ ( axis + 1 ) % 3 ] > 1
                       ? geo.n_gh : 0 );
    const int n_gh_b = ( geo.n_ceff[ ( axis + 2 ) % 3 ] > 1
                       ? geo.n_gh : 0 );
    const int n_bl   = geo.n_ceff[ ( axis + 2 ) % 3 ]
                     + 2 * n_gh_b;

    const int n_ce   = geo.n_ceff[ ( axis + 1 ) % 3 ] ;
    if( n_th_lim < 2 * geo.n_gh )
        throw std::runtime_error( "Insufficient n_th_lim "
                                  "for MHD boundaries." );
    const int  n_th  = std::min( n_ce, n_th_lim );
    if( n_ce % n_th != 0 )
        throw std::runtime_error( "Indivisible stride, b"
                                  "oundary::phys_base_t." );
    const int n_row  = n_ce + 2 * n_gh_a;
    const int s_sh = n_th * sh_per_th( b_d );
    //  Each thread covers one row, ghost rows included:
    //  the grid is padded to cover them all regardless of
    //  the thread count ( thin transverse slabs ).
    return std::make_tuple(
        dim3( ( n_row + n_th - 1 ) / n_th, n_bl ),
        dim3( n_th ), s_sh );
}

template < class prx_T , class derived_T >
template < class bdt_T > __device__ __host__
int phys_base_t< prx_T, derived_T >::sh_per_th
( const bdt_T & b_d )   const
{
    using  f_t = typename decltype( b_d.hyd ):: f_t;
    return sizeof( f_t ) * ( 2 * b_d.hyd.order(  ) + 1 )
                           * ( b_d.hyd.n_fld(  ) + 3 * 2 );
}

template < class  prx_T, class derived_T > __host__
void phys_base_t< prx_T, derived_T >::launch
( holder_p_t &  h, const int &  step,
  device::base_t & dev ) const
{
    auto & b_d ( prx_T::d( h.p_d->d (  ) ) );
    const auto & self( get_self(  ) );
    dev.launch( mesh::boundary::phy_ker, self.resource
              ( b_d ), h.stream, b_d, self, step );
    dev.event_record     ( h.event,       h.stream );
    return dev.event_wait( h.p_d->stream, h. event );
}

////////////////////////////////////////////////////////////
// Device-side interface

template  < class prx_T , class derived_T >
template  < class bdt_T > __device__
void phys_base_t< prx_T, derived_T > ::operator(  )
( const bdt_T & b_d, const int & step ) const
{
    __dyn_shared__( char, shared_mem );
    const int   axis = side >> 1;
    const bool right = side  & 1;

    const int  ax_p1 = utils::mod3( 1 + axis );
    const int  ax_p2 = utils::mod3( 2 + axis );
    const auto & geo =  b_d.geo;
    const int n_ceff =  b_d.geo.n_ceff[ axis ];
    const int n_hvar =  b_d.hyd.n_fld(  );
    const int & n_gh = b_d.geo.n_gh;  // Hydro and field
    const int & n_th =  blockDim.x;
    const int & i_th = threadIdx.x;
    //  A thin transverse axis has no ghost band  ( see
    //  resource ):  its row mapping must match.
    const int n_gh_a = ( geo.n_ceff[ ax_p1 ] > 1 ? n_gh : 0 );
    const int n_gh_b = ( geo.n_ceff[ ax_p2 ] > 1 ? n_gh : 0 );

    int i_tgt[ 3 ], i_src[ 3 ], i_sfc[ 3 ];
    i_tgt[ ax_p2 ]  = blockIdx.y - n_gh_b;
    
    const auto f_i = [ & ]( const int & a, const int & i_m )
    {
        int res = i_tgt[ a ];
        if( geo .phybnd_flag[ a ][ 0 ] )
            res = utils::max( res, 0   );
        if( geo .phybnd_flag[ a ][ 1 ] )
            res = utils::min( res, i_m );
        return res;
    };
    i_src[ ax_p2 ] = f_i( ax_p2, geo.n_ceff[ ax_p2 ] - 1 );
    i_sfc[ ax_p2 ] = f_i( ax_p2, geo.n_ceff[ ax_p2 ]     );

    using  f_t = typename decltype( b_d.hyd )::f_t;
    dat_1d_t< f_t > hyd_buf, bc_buf, bf_buf;
    void * var_sh = ( void * )
       ( shared_mem + i_th * sh_per_th( b_d ) );
    var_sh = hyd_buf.init( var_sh, 1, n_hvar,  n_gh );
    var_sh =  bc_buf.init( var_sh, 1, 3,       n_gh );
    var_sh =  bf_buf.init( var_sh, 1, 3,       n_gh );
    i_tgt[ ax_p1 ]  = blockIdx.x * n_th + i_th - n_gh_a;
    if( i_tgt[ ax_p1 ] >= geo.n_ceff[ ax_p1 ] + n_gh_a )
        return;                          // padding
    i_src[ ax_p1 ]  = f_i( ax_p1, geo.n_ceff[ ax_p1 ] - 1 );
    i_sfc[ ax_p1 ]  = f_i( ax_p1, geo.n_ceff[ ax_p1 ]     );

    if( copy_body )
        for( int i = 0; i <= n_gh; ++ i )
        {
            i_src[ axis ]
                 = ( right  ? n_ceff - i - 1 : i );
            i_sfc[ axis ]
                 = ( right  ? n_ceff - i     : i );
            //  The normal component lives on the face
            //  lattice ( i_sfc ), the tangential ones on
            //  the cell lattice ( i_src ): the two differ
            //  by one index at the right-hand side.
            auto * hyd_s = b_d.hyd. w.at( i_src       );
            auto * bf_s  = b_d.fld.bf.at( i_sfc, step );
            auto * bf_c  = b_d.fld.bf.at( i_src, step );
            for( int  n  = 0; n < n_hvar; ++ n )
                hyd_buf( i, n ) = hyd_s[ n ];
            for( int a = 0; a < 3; ++ a )
            {
                bc_buf( i, a ) = hyd_s[ idx::bcc + a ];
                if( step >= 0 )
                    bf_buf( i, a ) = ( a == axis )
                        ? bf_s[ a ] : bf_c[ a ];
            }
        }
    get_self(  ).hydro( hyd_buf,         i_tgt, b_d );
    get_self(  ).field(  bc_buf, bf_buf, i_tgt, b_d );
    for( int i = -n_gh; i < 0; ++ i )
    {
        i_tgt[ axis ]  = ( right ? n_ceff - i - 1 : i );
        int i_fbc[ 3 ];
        for( int a = 0; a < 3; ++ a )
            i_fbc[ a ] = i_tgt[ a ];
        i_fbc[ axis ]  = ( right ? n_ceff - i     : i );
        // Face- and cell-lattice ghost targets: the normal
        // component goes to the ghost faces ( i_fbc ), the
        // tangential ones to the ghost cells ( i_tgt ).
        auto * hyd     = hyd_buf   .at( i           );
        auto * hyd_t   = b_d.hyd.w .at( i_tgt       );
        auto * bf_f    = b_d.fld.bf.at( i_fbc, step );
        auto * bf_c    = b_d.fld.bf.at( i_tgt, step );
        for( int n = 0; n < n_hvar; ++ n )
            hyd_t[ n ] = hyd[ n ];
        for( int a = 0;  a < 3;  ++ a )
        {
            hyd_t[ idx::bcc + a ] = bc_buf( i, a );
            if( step >= 0 )
            {
                if( a == axis )
                    bf_f[ a ] = bf_buf( i, a );
                else
                    bf_c[ a ] = bf_buf( i, a );
            }
        }
    }
    if( step >= 0 && get_self(  ).pin_bnd_face )
    {
        //  Pin the normal component on the boundary face.
        //  Runs for every in-plane slot ( incl. corner
        //  columns ); side ordering ( axis 0 before axis 1,
        //  see keeper_t::assign_phys ) lets this overwrite
        //  whatever the other boundaries copied there.
        i_tgt[ axis ] = ( right ? n_ceff : 0 );
        b_d.fld.bf.at( i_tgt, step )[ axis ] = 0;
    }
    return;
};

};                    // namespace dynamics::mhd::boundary
