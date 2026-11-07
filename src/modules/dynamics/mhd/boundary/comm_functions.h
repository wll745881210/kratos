#pragma once

#include "comm.h"

namespace dynamics::mhd::boundary
{
////////////////////////////////////////////////////////////
// Kernels

template< class bdt_T, class bnd_T > __global__  void
ker_edge( const bdt_T bdt, const bnd_T bnd, const int step )
{
    return bnd.edge_efield( bdt, step );
}

////////////////////////////////////////////////////////////
// Memory utilities

template< class prx_T, class derived_T > __host__ void
comm_t< prx_T, derived_T >::free( const holder_c_t & h )
{
    b_bsnd.free( h.p_dev->f_free );
    b_brcv.free( h.p_dev->f_free );
    b_ecom.free( h.p_dev->f_free );
    return super_t::free     ( h );
}

template< class prx_T, class  derived_T >
template< class d30_T, class      d31_T >
__host__  void comm_t< prx_T, derived_T >::alloc_dual
( d30_T   & a0, d31_T  &  a1, const  holder_c_t & h )
{
    auto dum    = [ & ] ( const  size_t & c ) -> void *
    {    return                           nullptr ; } ;
    using ft0 = std::remove_pointer_t< decltype( a0.dat ) >;
    using ft1 = std::remove_pointer_t< decltype( a1.dat ) >;
    a0.init( dum,  a0.n_cell, idx_t::null(  ), a0.n_int )  ;
    a1.init( dum,  a1.n_cell, idx_t::null(  ), a1.n_int )  ;
    a0.dat = ( ft0 * ) h.p_dev->malloc_device< char >
             ( a0.n_ctot   *  sizeof( ft0 ) +
               a1.n_ctot   *  sizeof( ft1 ) ) ;
    a1.dat = ( ft1 * ) ( a0.dat + a0.n_ctot ) ;
    a1.proxy   =       true;
    a0.n_size += a1. n_size;
    return;
}

////////////////////////////////////////////////////////////
// Setups

template< class prx_T, class derived_T > __host__
void  comm_t < prx_T, derived_T >::set_geo
( holder_c_t & h, const neighbor_t & nb, mod_base_t & mod )
{
    //  Run the full hydro chain first: super_t::set_geo
    // assigns the member `order` ( space order ), which the
    // inherited hydro set_mem uses to size the cross-rank
    // send/recv buffers along `nb[ a ] < 0` axes ( edge and
    // vertex neighbors, and negative faces ). Skipping it
    // ( calling comm_base_t directly ) left `order` = 0 and
    // allocated zero-volume buffers -> OOB in ker_to_buf.
    super_t::set_geo( h, nb, mod );
    same_rank  = h.same_rank(  );

    const auto & ord( mod.space_order(  ) );
    const auto & d  ( * h.p_d );
    const auto & ax ( axis    [ 0  ] );
    const auto & nr ( nb.reverse(  ) );
    const auto & geo( d.     geo(  ) );

    for( int a = 0; a < 3; ++ a )
    {
        const auto n_2( geo.n_ceff[ a ] / 2 );

        offset_es[ a ]  = std::max( nb[ a ], 0 ) * n_2;
        offset_ec[ a ]  = same_rank
                        * std::max( nr[ a ], 0 ) * n_2;
        offset_er[ a ]  = std::max( nb[ a ], 0 ) * n_2;

        offset_bs[ a ]  = std::max( nb[ a ], 0 ) * n_2;
        offset_bc[ a ]  = same_rank
                        * std::max( nr[ a ], 0 ) * n_2;
        offset_br[ a ]  = std::max( nb[ a ], 0 ) * n_2;
    }   // Useful for mode 0 only; mode 1 does not care
    offset_bs[ ax ] -=             ( nb[ ax ] > 1 );
    offset_br[ ax ] -=             ( nr[ ax ] > 1 );
    offset_bc[ ax ] -= same_rank * ( nb[ ax ] > 1 );

    // For hydrodynamic variables
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        const  auto n_2( geo.  n_ceff[ a ] / 2 ) ;
        offset_s[ a ]  = std::max( nb[ a ] , 0 ) * n_2
           - utils::shift( ord, -d_lvl ) * ( nb[ a ] > 0 );
        offset_r[ a ]  = std::max( nb[ a ] , 0 ) * n_2
                       - ord  *  ( nr[ a ] > 0 );
        offset_c[ a ]  = std::max( nr[ a ] , 0 ) * n_2
                       - ord  *  ( nb[ a ] > 0 );
        offset_c[ a ] *= same_rank;
    }
    if( mode == 1 )
    {   // v--- axis[ 1 ] is extension direction for mode 1
        const auto & a1 = axis[ 1 ];
        const auto n_2 ( geo.n_ceff[ a1 ] / 2 );
        offset_s[ a1 ] = nb [ a1 ] * n_2;
        offset_r[ a1 ] = nb [ a1 ] * n_2;
        offset_c[ a1 ] = nr [ a1 ] * n_2 * same_rank;
    } // Some corrections are not applicable to mode 1.
    return;
}

template< class prx_T, class derived_T > __host__
void comm_t   < prx_T, derived_T >::set_mem
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    if( h.same_rank(  ) )
        return;
    super_t::set_mem( h,     nb,      mod );
    const auto & ord( mod.space_order(  ) );
    const auto & d  ( * h.p_d );
    const auto & ax ( axis    [ 0  ] );
    const auto & nr ( nb.reverse(  ) );
    const auto & geo( d.     geo(  ) );

    if( mode == 0 )
    {
        b_ecom. n_int = 2 ;
        b_bsnd. n_int = 3 ;
        b_brcv. n_int = 3 ;
        for( auto * p : { & b_send, & b_recv, & b_fcom } )
            p->free ( h.p_dev->f_free );

        const auto & n( d_lvl > 0 ? nb : nr );
        for( const auto &  a : { axis[ 1 ], axis[ 2 ]  } )
        {
            b_bsnd.n_cell[ a ] = b_send.n_cell [ a ]
                + ( d_lvl >= 0 ? 1 : ( nr[ a ] > 0 ) );
            b_brcv.n_cell[ a ] = b_recv.n_cell [ a ]
                + ( d_lvl <= 0 ? 1 : ( nb[ a ] > 0 ) );

            b_ecom.n_cell[ a ]  = b_fcom. n_cell[ a ]
                                + ( n [ a ] > 0 ) ;
            b_send.n_cell[ a ] += ( d_lvl > 0 ) * ord;
            b_recv.n_cell[ a ] += ( d_lvl < 0 ) * ord;
        }
        b_bsnd.n_cell[ ax ] = 1;
        b_brcv.n_cell[ ax ] = 1;
        b_ecom.n_cell[ ax ] = 1;

        alloc_dual( b_send, b_bsnd, h );
        alloc_dual( b_recv, b_brcv, h );
        if( d_lvl != 0 )
            alloc_dual( b_fcom, b_ecom, h );
    }
    else
    {
        super_t::alloc_one( b_send, h );
        super_t::alloc_one( b_recv, h );
        if( d_lvl != 0 )
        {
            b_ecom.n_int        =                    1;
            b_ecom.n_cell       = idx_t( { 1, 1, 1 } );
            b_ecom.n_cell[ ax ] = geo.n_ceff[ ax ] / 2;
            super_t::alloc_one( b_ecom, h ) ;
        }
        b_fcom = b_ecom;        // For a unified "set_comm"
    }
    return;
}

template< class prx_T, class derived_T >  __host__
void  comm_t  < prx_T,       derived_T >::set_local
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    if( ! h.same_rank(  ) )
        return;
    const auto & bdg
        ( prx_T::ref( mod.data( nb.guest_reg ).d(  ) ) );

    b_send = bdg.hyd. w;
    b_fcom = bdg.hyd. f[ axis[ 0 ] ];
    b_bsnd = bdg.fld.bf;
    b_ecom = bdg.fld.ee;
    return;
}

//  Shared flux / edge pack wiring: the packs write the
//  coarse guest's boundary flux faces and tangential E
//  edges -- slots the guest's own calc_field also reads
//  and writes -- so they wait the DESTINATION block's
//  calc event, not only the source block's.  The kernel
//  launches on h.stream ( ordered after d.event ) so that
//  h.event covers it; the same-rank guest stream must
//  wait before its intg_field reads the edges.
template< class prx_T, class derived_T >
template< class par_T, class str_T, class evt_T >
__host__ void comm_t< prx_T, derived_T >::set_fpck
( holder_c_t & h, const par_T & par,
  const str_T & str_dest, const evt_T & ev_dest )
{
    auto & self = get_self          (       );
    auto &    d ( * h.p_d );
    const int m = mode;
    h.f_fpck = [ &, par, m ] ( const int &   step )
    {
        h.p_dev->event_wait  ( h.stream, d. event );
        if( h.same_rank(  ) && h.d_lvl < 0 )
            h.p_dev->event_wait( h.stream, ev_dest );
        const auto & b_d = prx_T::d_i( d.d(   ) );
        if( m == 0 )
            h.p_dev->launch( hydro::boundary::ker_flux ,
                             par, h.stream ,
                             b_d, self,      step );
        else
            h.p_dev->launch( ker_edge, par , h.stream ,
                             b_d, self,      step );
        h.p_dev->event_record( h. event, h.stream );
        h.p_dev->event_wait  ( d.stream, h. event );
        if( h.same_rank(  ) && h.d_lvl < 0 )
            h.p_dev->event_wait( str_dest, h. event );
    };
    return;
}

template< class prx_T, class derived_T >    __host__
void comm_t   < prx_T, derived_T >::set_launch
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    super_t::set_launch( h, nb, mod );
    auto & self   = get_self          (       );
    auto & d( * h.p_d );
    if( h.same_rank(  ) && h.d_lvl < 0 )
    {
        //  Degenerate blocks ( n_ceff == n_gh ): the
        //  restriction also reads the host's own ghost
        //  band, filled only in the q_same stage AFTER
        //  the q_ghst pack.  Re-run the pack in f_unpk
        //  ( routed to q_plgt.unpk, the last act_bnd
        //  stage ); the restriction is idempotent, so
        //  the first-layer early-pass value stays valid.
        h.f_unpk = h.f_pack;
    }
    if( d_lvl < 0 )
    {
        //  The restriction reads the host's ghost band
        //  that prolongations on other streams may write;
        //  record completion on the host stream so its
        //  later captures follow ( d_lvl > 0 below ).
        const auto par_pack = self.resource_pack( h, nb );
        auto  guest( nb.guest );
        mod.p_mesh-> tree(  ) . regularize  ( guest );
        const auto & str_dest = ( ! h.same_rank(  ) )
                   ? d.stream : mod.data( guest ).stream;
        h.f_pack = [ &, par_pack ] ( const int &   step )
        {
            const auto & b_d  =    prx_T::d_i( d.d(   ) );
            h.p_dev->event_wait    ( h.stream, d. event );
            h.p_dev->launch
                ( dynamics::hydro::boundary::ker_to_buf ,
                  par_pack, h.stream, b_d, self, step   );
            h.p_dev->event_record  ( h.event,  h.stream );
            h.p_dev->event_wait    ( str_dest, h. event );
            h.p_dev->event_wait    ( d.stream, h. event );
        };
    }
    if( d_lvl > 0 )
    {
        //  The slope stencil reads host ghost cells that
        //  this act_bnd's restrictions refill ( queued on
        //  the host stream ); an event recorded here then
        //  follows every restriction wait.  Without it the
        //  prolongation races the restrictions on the GPU.
        const auto par_pack = self.resource_pack( h, nb );
        auto  guest( nb.guest );
        mod.p_mesh-> tree(  ) . regularize  ( guest );
        const auto & str_dest = ( ! h.same_rank(  ) )
                   ? d.stream : mod.data( guest ).stream;
        h.f_pack = [ &, par_pack ] ( const int &   step )
        {
            const auto & b_d  =    prx_T::d_i( d.d(   ) );
            h.p_dev->event_wait    ( h.stream, d. event );
            h.p_dev->event_record  ( h.event,  d.stream );
            h.p_dev->event_wait    ( h.stream, h. event );
            if( h.same_rank(  ) )
            {
                //  Also capture the guest stream: a
                //  restriction may read this block while
                //  the prolongation writes its ghost band.
                h.p_dev->event_record( h.event, str_dest );
                h.p_dev->event_wait  ( h.stream, h.event );
            }
            h.p_dev->launch
                ( dynamics::hydro::boundary::ker_to_buf ,
                  par_pack, h.stream, b_d, self, step   );
            h.p_dev->event_record  ( h.event,  h.stream );
            h.p_dev->event_wait    ( str_dest, h. event );
        };  // The destination blk depends on this
    }
    if( d_lvl != 0 && ( mode == 0 || mode == 1 ) )
    {
        auto  guest( nb.guest );
        mod.p_mesh-> tree(  ) . regularize  ( guest );
        const auto & str_dest = ( ! h.same_rank(  ) )
                   ? d.stream : mod.data( guest ).stream;
        const auto &  ev_dest = mod.data( guest ).event;
        if( mode == 0 )
            set_fpck( h, self.resource_flux( h, nb ),
                      str_dest, ev_dest );
        else
            set_fpck( h, self.resource_edge( h, nb ),
                      str_dest, ev_dest );
    }
    return;
}

////////////////////////////////////////////////////////////
// Resource for device-side computation

template< class prx_T, class derived_T >
__host__ std::tuple< dim3, dim3, int >
comm_t< prx_T, derived_T >::resource_pack
( const holder_c_t & h, const neighbor_t & nb )  const
{
    //  Hydro's mode clamps ( tangent axes -> order slots )
    //  are receiver-band widths, valid for d_lvl >= 0.  A
    //  to_coarser host spans twice the receiver slots, so
    //  the plain division below is already correct; the
    //  clamp would halve the tangent coverage ( leaving
    //  receiver ghost bands unfilled ).
    const auto & b_d  = prx_T::d_i( h.p_d->d(  ) );
    const auto & geo  ( b_d.   geo );
    idx_t n_cell      ( geo.n_ceff );
    if( d_lvl >= 0 )
    {
        if( mode > 0 )
            n_cell[ axis[ 2 ] ] = this -> order;
        if( mode > 1 )
            n_cell[ axis[ 1 ] ] = this -> order;
    }
    for( int a = 0; a < geo.n_dim; ++ a )
        n_cell[ a ] /= 1 + ( d_lvl != 0 );
    if( d_lvl < 0 && mode > 0 )
    {
        //  Banded receivers: the to_coarser gate admits
        //  i_b0 < n_gh slots, while the division yields
        //  n_ceff / 2 -- starving the band for n_ceff <
        //  2 * n_gh.  Widen to the full band; for larger
        //  blocks the gate already caps execution below.
        n_cell[ axis[ 2 ] ] = geo. n_gh;
        if( mode > 1 )
            n_cell[ axis[ 1 ] ] = geo. n_gh;
    }

    const auto [ n_th, n_bl_x ] = resource_thbl( n_cell );
    int n_bl_y ( n_cell[ axis[ 2 ] ] );
    const auto n_gh = h.p_d->d(  ).geo.n_gh;
    if( mode == 0 )
    {
        if( d_lvl > 0 )
            n_bl_y += n_gh / 2;
        else if( d_lvl < 0 ) // For the E-B fields
            n_bl_y += ( nb.reverse( )[ axis[ 2 ] ] > 0 );
        else
            n_bl_y += 1;
    }
    //  else: mode != 0 -- no extra rows; the edge / E
    //  work uses resource_edge.  ( An n_gh override here
    //  once inflated the grid past the receiver ghost
    //  corner. )
    const int s_sh = n_th * b_d.hyd.n_fld( ) * sizeof( f_t )
              * ( d_lvl  > 0 ? 4 : ( d_lvl < 0 ? 1 : 0 ) );
    return std::make_tuple
         ( dim3( n_bl_x, n_bl_y, 1 ), n_th, s_sh );
}

template< class prx_T, class derived_T >
__host__ std::tuple< dim3, dim3, int >
comm_t< prx_T, derived_T >::resource_unpk
( const holder_c_t & h, const neighbor_t & nb )  const
{
    auto    res = super_t::resource_unpk ( h, nb );
    auto & n_bl = std::get< 0 > ( res );
    int d_nby = ( n_bl.y > 1 && mode == 0 );
    if( d_lvl <  0 )
        d_nby *= h.p_d->d(  ).geo.n_gh;
    n_bl.y += d_nby;
    return res;
}

template< class prx_T, class derived_T >
__host__ std::tuple< dim3, dim3, int >
comm_t< prx_T, derived_T >::resource_flux
( const holder_c_t & h, const neighbor_t & nb )  const
{
    const auto & geo( h.p_d->geo(  ) );
    auto n_cell = geo.n_ceff;
    for( int a = 0; a < geo.n_dim; ++ a )
        n_cell[ a ] /= 2;    
    auto [ n_th, n_bl_x ] = resource_thbl( n_cell );

    const auto & n( d_lvl > 0 ? nb  : nb.reverse (  ) );
    const auto n_bl_y = n_cell[ axis[ 2 ] ]
                      + (    n[ axis[ 2 ] ] > 0 );
    return std::make_tuple
         ( dim3( n_bl_x, n_bl_y, 1 ), n_th,  0 );
}

template< class prx_T, class derived_T >
__host__ std::tuple< dim3, dim3, int >
comm_t< prx_T, derived_T >::resource_edge
( const holder_c_t & h, const neighbor_t & nb ) const
{   // v-- mode 1: axis[ 1 ] for extension
    const auto n_edge = h.p_d->geo
                        (  ).n_ceff[ axis[ 1 ] ] / 2;
    const auto n_th   = std::min( n_edge, this->n_th_lim );
    const auto n_bl   = ( n_edge + n_th - 1 ) / n_th;
    return std::make_tuple( n_bl , n_th , 0 ) ;
}

////////////////////////////////////////////////////////////
// Buffer packing/unpacking

//  Per-axis receiver-band predicate: mode >= 1 slabs are
//  `ord` cells wide starting at the band offset; a zero
//  offset ( face side ) admits the whole axis.
__device__ __forceinline__ static bool in_band_axis
( const int & i_b, const int & off, const int & ord )
{
    return off == 0
        || ( i_b - off >= 0 && i_b - off < ord );
}

template < class prx_T , class  derived_T >
template < class bdt_T >
__device__ void comm_t < prx_T, derived_T >::to_same
( const bdt_T & b_d, const int & step ) const
{
    const auto & geo = b_d.geo;
    idx_t i_m,   i_b;
    get_self(  ).dispatch( i_b );

    if( i_b[ axis[ 2 ] ] < geo.n_ceff[ axis[ 2 ] ] )
        super_t::to_same ( b_d, step );
    if( mode != 0 )
        return;
    
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        i_m[ a ] = offset_bs[ a ] + i_b [ a ];
        i_b[ a ] = offset_bc[ a ] + i_b [ a ];
    }
    const int  j_max( ( blockIdx.x == gridDim.x - 1 ) + 1 );
    const int ib_max( geo. n_ceff[ axis[ 1  ] ] + 1 ) ;
    const auto & ib1( i_b[ axis[ 1 ] ] ) ;
    for( int j = 0; j <  j_max && ib1 < ib_max; ++  j )
    {
        auto * b_buff = b_bsnd.at( i_b, same_rank * step );
        auto * b_mesh = b_d.fld.bf.at ( i_m, step );
        b_buff [ axis[ 1 ] ]  = b_mesh[ axis[ 1 ] ];
        b_buff [ axis[ 2 ] ]  = b_mesh[ axis[ 2 ] ];
        i_b[ axis[ 1 ] ] += blockDim.x;
        i_m[ axis[ 1 ] ] += blockDim.x;
    }
    return;
}

template < class prx_T , class  derived_T >
template < class bdt_T >
__device__ void comm_t < prx_T, derived_T >::to_finer
( const bdt_T & b_d, const int & step ) const
{
    const int  &  ax = axis [ 0 ];
    const auto & geo = b_d.   geo;
    const auto &  bf = b_d.fld.bf;
    idx_t i_b,  i_m;
    get_self(  ).dispatch ( i_m );
    //////////////////////////////////////////////////
    //  W-prolongation, MHD-local.  Mode 0 reproduces
    //  hydro's write-set exactly ( full receiver slab:
    //  the grid tiles guest interior + tangent ghost
    //  extensions; a tangent offset_c of -2 / +4 on a
    //  swept axis is normal there, not a band marker ).
    //  The mode >= 1 band clamps ( im_max, in_band )
    //  belong to holders hydro never enrolls; the
    //  hydro module stays untouched.
    {
        const auto & hyd = b_d.hyd;
        const int  n_int = hyd.n_fld(  );

        __dyn_shared__( f_t, var_sh );
        f_t * q_c =   var_sh  + n_int * 4 * threadIdx.x ;
        f_t * dq[ 3 ] = { q_c + n_int,  q_c + 2 * n_int ,
                                        q_c + 3 * n_int };
        idx_t i_m0;
        get_self(  ).dispatch( i_m0 ) ;
        for( int a = 0 ; a < geo.n_dim; ++ a )
        {
            i_b[ a ] = offset_c[ a ] + 2 * i_m0[ a ];
            i_m[ a ] = offset_s[ a ] +     i_m0[ a ];
        }
        //  Band-targeted thread axis, mode >= 1 only:
        //  one coarse slot.  Mode-0 holders sweep the
        //  full extended slab ( n_ceff/2 + order/2
        //  coarse slots ), filling the receiver's
        //  tangent ghost extensions at the interface.
        const int im_max = ( mode > 0 &&
                             offset_c[ axis[ 1 ] ] != 0 )
            ? 1 : ( geo.n_ceff[ axis[ 1 ] ]
                    + this -> order ) / 2;
        const int j_max( ( blockIdx.x == gridDim.x - 1 )
                         + 1 );
        const int  & im1( i_m0[ axis [ 1 ] ] );
        for( int j = 0; j < j_max && im1 < im_max; ++ j )
        {
            for( int k = 0; k < this -> order; ++ k )
            {
                i_m[ ax ] = offset_s[ ax ] + k / 2;
                i_b[ ax ] = offset_c[ ax ] +     k;
                if( ( k & 1 ) == 0 )
                {
                    for( int n = 0; n < n_int; ++ n )
                        q_c[ n ] = hyd.w.at( i_m )[ n ];
                    for( int a = 0; a < geo.n_dim; ++ a )
                    {
                        i_m[ axis[ a ] ] += 1;
                        const auto * q_r  = hyd.w.at( i_m );
                        i_m[ axis[ a ] ] -= 2;
                        const auto * q_l  = hyd.w.at( i_m );
                        i_m[ axis[ a ] ] += 1;
                        for( int n = 0; n < n_int; ++ n )
                            dq[ axis[ a ] ][ n ] =
                                utils::minmod
                                ( q_r[ n ] - q_c[ n ] ,
                                  q_c[ n ] - q_l[ n ] )
                                * 0.25f;
                    }
                }
                map_refine( [ & ]
                    ( const int & d1, const int & d2 )
                {
                    i_b[ axis[ 1 ] ] += d1;
                    i_b[ axis[ 2 ] ] += d2;
                    //  Band guard, mode >= 1: the slab is
                    //  `order` cells wide, [ offset_c,
                    //  offset_c + order ); mode 0: full.
                    const int & i1( i_b[ axis[ 1 ] ] );
                    const int & o1( offset_c[ axis[ 1 ] ] );
                    const int & i2( i_b[ axis[ 2 ] ] );
                    const int & o2( offset_c[ axis[ 2 ] ] );
                    const bool in_band = ( mode == 0 )
                        || ( in_band_axis
                               ( i1, o1, this->order )
                             && in_band_axis
                               ( i2, o2, this->order ) );
                    if( in_band )
                    {
                        auto * q = b_send.at( i_b  );
                        for( int n = 0; n < n_int; ++ n )
                            q[ n ] = q_c[ n ]
                                + dq[ ax ][ n ]
                                * ( 2 * ( k & 1 ) - 1 )
                                + dq[ axis[ 1 ] ][ n ]
                                * ( 2 * d1 - 1 )
                                + dq[ axis[ 2 ] ][ n ]
                                * ( 2 * d2 - 1 );
                    }
                    i_b[ axis[ 1 ] ] -= d1;
                    i_b[ axis[ 2 ] ] -= d2;
                },  geo.n_dim  ) ;
            }
            i_b [ axis[ 1 ] ] += 2 * blockDim.x;
            i_m [ axis[ 1 ] ] +=     blockDim.x;
            i_m0[ axis[ 1 ] ] +=     blockDim.x;
        }
    }
    //  The former super_t::to_finer call ( with the
    //  step < 0 marker for mode 0 ) is replaced above:
    //  the super's j-loop cannot clamp im_max for
    //  band-targeted ( mode >= 1 ) receivers.

    if( mode != 0 )
        return;

    get_self(  ).dispatch( i_m );   // Refresh for B part
    for( int a = 0; a < geo.n_dim; ++  a )
    {
        i_b[ a ] = 2 * i_m[ a ] + offset_bc[ a ] ;
        i_m[ a ] =     i_m[ a ] + offset_br[ a ] ;
    }
    if( i_b[ axis[ 2 ] ] > geo.n_ceff[ axis[ 2 ] ] - 2 ||
        i_b[ axis[ 1 ] ] > geo.n_ceff[ axis[ 1 ] ] - 2  )
        return;

    //////////////////////////////////////////////////
    //  Toth & Roe (2002) prolongation for FC fields
    //  b_cache[ ax_field ][ stage ][ ax_1 ][ ax_2 ]
    f_t b[ 3 ][ 3 ] [ 2 ][ 2 ];
    f_t db_1  [ 2 ], db_2[ 3 ], db_3[ 3 ] ;
    using utils::minmod;
    //  Write target along the interface axis.  Same-rank:
    //  b_bsnd aliases the guest's bf, so write its GHOST
    //  column ( read by the transverse normal-B loads );
    //  cross-rank: layer 0 of the send slab, from_buf
    //  places it.  rhs: guest on the high side -> its
    //  interface face is the low one -> ghost at -1.
    //  ( The old column-0 write starved the low-side
    //  ghost and clobbered the high-side far interior. )
    i_b [ ax ] = ( same_rank
        ? ( rhs ? -1 : geo.n_ceff[ ax ] ) : 0 );
    i_m [ ax ] = ( rhs ? geo.n_ceff[ ax ] - 1 : 0 ) ;
    for( int a = 0; a < 3; ++ a )
    {
        const int a_p[ 2 ] = { utils::mod3( a + 1 ) ,
                               utils::mod3( a + 2 ) };
        for( int d = 0; d < 2; ++ d )
        {
            i_m[ a ] += d ;
            const auto bc = bf.at( i_m, step )[ a ] ;
            auto  bs = b[ a ] [ 2 * d ] ;
            for( int n = 0; n < 2; ++ n )
            {
                i_m[ a_p[ n ] ]  -= 1;
                const auto bl = bf.at( i_m, step )[ a ];
                i_m[ a_p[ n ] ]  += 2;
                const auto br = bf.at( i_m, step )[ a ];
                i_m[ a_p[ n ] ]  -= 1;
                db_1[ n ] = minmod
                    ( br - bc, bc - bl ) / 4 ;
            }
            bs[ 0 ][ 0 ] = bc - db_1[ 0 ] - db_1[ 1 ];
            bs[ 0 ][ 1 ] = bc - db_1[ 0 ] + db_1[ 1 ];
            bs[ 1 ][ 0 ] = bc + db_1[ 0 ] - db_1[ 1 ];
            bs[ 1 ][ 1 ] = bc + db_1[ 0 ] + db_1[ 1 ];
            i_m [ a ] -= d ;
        }
    }
    for( int a = 0; a < 3; ++ a )
    {
        db_2[ a ] = 0 ;
        const int a_1 = utils::mod3( a + 1 );
        const int a_2 = utils::mod3( a + 2 );
        for( int l = 0; l < 2; ++ l )
            for( int m = 0; m < 2; ++ m )
                db_2[ a ] +=  ( 2 * m - 1 ) *
                    ( b[ a_1 ][ 2 ][ l ][ m ] -
                      b[ a_1 ][ 0 ][ l ][ m ] )
                          +   ( 2 * l - 1 ) *
                    ( b[ a_2 ][ 2 ][ l ][ m ] -
                      b[ a_2 ][ 0 ][ l ][ m ] ) / 8;
        db_3 [ a ] = 0;
        for( int k = 0; k < 2; ++ k )
            for( int l = 0; l < 2; ++ l )
                for( int m = 0; m < 2; ++ m )
                    db_3[ a ] +=  ( 2 * k - 1 ) /  16.f
                        * ( 2 * l - 1 ) * ( 2 * m - 1 )
                        * b [ a ][ 2 * k ][ l ] [ m ] ;
    }
    for( int a = 0; a < 3; ++ a )
    {
        const int a_1 = utils::mod3( a + 1 );
        const int a_2 = utils::mod3( a + 2 );
        auto  bs = b [ a ][ 1 ];
        for( int l = 0 ; l < 2; ++  l )
            for( int m = 0 ; m < 2; ++  m )
                bs[ l ][ m ]  =  db_2 [ a ] +
                    ( b[ a ][ 2 ][ l ][ m ] +
                      b[ a ][ 0 ][ l ][ m ] ) / 2
                    + db_3[ a_1 ] * ( 2 * l - 1 )
                    + db_3[ a_2 ] * ( 2 * m - 1 );
    }
    // i_b[ a ] shall not reach n_ceff[ a ] - 2,
    // except for the "boundary-touching"  case.
    const int l_max = ( i_b   [ axis[ 1 ] ] == geo.
                        n_ceff[ axis[ 1 ] ] - 2 ) + 2;
    const int m_max = ( i_b   [ axis[ 2 ] ] == geo.
                        n_ceff[ axis[ 2 ] ] - 2 ) + 2;
    for( int l = 0; l < l_max; ++ l )
        for( int m = 0; m < m_max; ++ m )
        {
            i_b[  axis [ 1 ] ] += l ;
            i_b[  axis [ 2 ] ] += m ;
            auto  * bt = b_bsnd.at( i_b, same_rank * step );
            if( m < 2 )
                bt [ axis[ 1 ] ]
                    =  b[ axis[ 1 ] ][ l ][ m ][ rhs ];
            if( l < 2 )
                bt [ axis[ 2 ] ]
                    =  b[ axis[ 2 ] ][ m ][ rhs ][ l ];
            i_b[  axis[ 1 ] ] -= l;
            i_b[  axis[ 2 ] ] -= m;
        }
    return;
}

template < class prx_T , class  derived_T >
template < class bdt_T >
__device__ void comm_t < prx_T, derived_T >::to_coarser
( const bdt_T & b_d, const int & step ) const
{
    const int     ax = axis [ 0 ];
    const auto &  bf = b_d.fld.bf;
    const auto & geo = b_d.   geo;

    idx_t i_m, i_b, i_b0;
    get_self(   ).dispatch( i_b0 );
    //  mode 0: tangent footprints span n_ceff / 2
    //  receiver slots; mode 1 / 2: only the n_gh-wide
    //  ghost band ( wider reads leave the host extent ).
    const int lim2 = ( mode > 0 ? geo.n_gh
                     : geo.n_ceff[ axis[ 2 ] ] / 2 );
    const int lim1 = ( mode > 1 ? geo.n_gh
                     : geo.n_ceff[ axis[ 1 ] ] / 2 );
    if( i_b0[ axis[ 2 ] ] < lim2
     && i_b0[ axis[ 1 ] ] < lim1 )
        super_t::to_coarser( b_d, step );
    if( mode != 0 )
        return;

    for( int a = 0; a < 3; ++ a )
    {
         i_m [ a ] = 2 * i_b0[ a ] + offset_bs[ a ];
         i_b [ a ] =     i_b0[ a ] + offset_bc[ a ];
    }
    //  The tangential-component footprint must cover the
    //  two fine cells adjacent to the interface; on the
    //  right side offset_bs points at the last interior
    //  row, so shift the base one row inward ( otherwise
    //  the depth loop reads the ghost row, and overruns
    //  the unpadded geometry tables ).
    if( offset_bs[ ax ] > 0 )
        -- i_m[ ax ];
    const int j_max( ( blockIdx.x == gridDim.x - 1 ) + 1 );
    int ib_max  = b_bsnd.n_cell[ axis[ 1 ] ];
    if( same_rank ) // ^-- ( ! same_rank ) only!
        ib_max = geo.n_ceff[ axis[ 1 ] ] / 2
             + ( offset_bc [ axis[ 1 ] ] > 1 );
    // ^-- To the right side along axis[ 1 ]
    const auto & ib1( i_b0 [ axis[ 1 ] ] ) ;
    for( int j = 0; j < j_max && ib1 < ib_max; ++ j )
    {
        i_m[ axis[ 1 ] ] = 2 *  i_b0[ axis[ 1 ] ]
                         + offset_bs[ axis[ 1 ] ];
        auto * b_t  = b_bsnd.at( i_b, same_rank * step );
        for( int l = 0; l < 2; ++ l )
        {
            const  int & a  ( axis[ 1 + l ] );
            const  int & a_p( axis[ 2 - l ] );
            //  Faces of comp a have no boundary row
            //  along a_p; at that receiver row only
            //  the other comp exists.
            if( i_m[ a_p ] >= geo.n_ceff[ a_p ] )
                continue;

            f_t b( 0 ) , s( 0 ) ;
            const int depth = 2;
            for( int k = 0; k < depth; ++ k )
            {
                auto ds = geo.surface( i_m,         a );
                b  += ds * bf .at( i_m, step )[ a ];
                s  += ds ;
                ++ i_m[ a_p ];
                ds   = geo.surface( i_m,         a );
                b  += ds * bf .at( i_m, step )[ a ];
                s  += ds ;
                -- i_m[ a_p ];
                ++ i_m[ ax  ];
            }
            i_m[ ax ] -= depth;
            //  Zero-area footprint rows (e.g. the pole
            //  theta-face row, whose surface vanishes
            //  exactly) receive zero - the pole boundary
            //  functor pins those faces anyway.
            b_t [ a ]  = ( s > 0 ? b / s : 0 );
        }
        i_b0[ axis[ 1 ] ] +=      blockDim. x;
        i_b [ axis[ 1 ] ] +=      blockDim. x;
    }
    return;
}

////////////////////////////////////////////////////////////
// Extract from buffer

template < class prx_T , class  derived_T >
template < class bdt_T >
__device__ void comm_t < prx_T, derived_T >::from_buf
( const bdt_T & b_d, const int & step ) const
{
    idx_t i_m, i_b;
    get_self(  ).dispatch( i_b );

    if( i_b[ axis[ 2 ] ] <  b_recv.n_cell[ axis[ 2 ] ] )
        super_t::from_buf( b_d, step );
    if( mode  != 0 ||
        i_b[ axis[ 2 ] ] >= b_brcv.n_cell[ axis[ 2 ] ] )
        return;
    for( int a = 0; a < b_d.geo.n_dim; ++ a )
        i_m[ a ] = i_b[ a ] + offset_br [ a ];

    const int  j_max( ( blockIdx.x == gridDim.x - 1 ) + 1 );
    const int ib_max( b_brcv.n_cell[ axis[ 1 ] ] );
    const auto & ib1( i_b          [ axis[ 1 ] ] );
    for( int j = 0; j < j_max && ib1 < ib_max; ++ j )
    {
        auto * b_buff  = b_brcv    .at( i_b       );
        auto * b_mesh  = b_d.fld.bf.at( i_m, step );
        b_mesh [ axis[ 1 ] ]  = b_buff[ axis[ 1 ] ];
        b_mesh [ axis[ 2 ] ]  = b_buff[ axis[ 2 ] ];
        i_b    [ axis[ 1 ] ] +=         blockDim. x;
        i_m    [ axis[ 1 ] ] +=         blockDim. x;
    }
    return;
}

////////////////////////////////////////////////////////////
// Fluxes (including electric field) communication

template < class prx_T , class  derived_T >
template < class bdt_T >
__device__ void comm_t < prx_T, derived_T >::flux_to_buf
( const bdt_T & b_d, const int & step ) const
{
    const auto & geo = b_d . geo;
    const auto &  ax = axis[ 0 ];
    idx_t i_m,   i_b, i_b0;
    get_self(  ).dispatch( i_b0 );
    if( i_b0[ axis[ 2 ] ] < geo.n_ceff[ axis[ 2 ] ] / 2 )
    {
        super_t::flux_to_buf( b_d, step );
    }

    for( int a = 0; a < b_d.geo.n_dim; ++ a )
    {
        i_m [ a ] = 2 * i_b0[ a ] + offset_es[ a ];
        i_b [ a ] =     i_b0[ a ] + offset_ec[ a ];
    }

    const int j_max( ( blockIdx.x == gridDim.x - 1 ) + 1 );
    int ib_max  = b_ecom.n_cell[ axis[ 1 ] ];
    if( same_rank ) // ^-- ( ! same_rank ) only!
        ib_max = geo.n_ceff[ axis[ 1 ] ] / 2
             + ( offset_ec [ axis[ 1 ] ] > 1 );
    // ^-- To the right side along axis[ 1 ]
    const auto & ib1( i_b0 [ axis[ 1 ] ] ) ;
    for( int j = 0; j <  j_max && ib1 < ib_max; ++ j )
    {
        i_m[ axis[ 1 ] ] = 2 *  i_b0[ axis[ 1 ] ]
                         + offset_es[ axis[ 1 ] ];

        auto * e_b = b_ecom    .at( i_b );
        auto * e_m = b_d.fld.ee.at( i_m );
        for( int a = 0; a < 3; ++ a )
        {
            const int & am( axis[ utils::mod3( ax + a ) ] );
            if( am == ax )
            //  ^-- interface-normal: never packed ( its
            //  source rows are ghosts ); the receiver gets
            //  its own from interior fluxes.
                continue;
            const int   ab( same_rank ? am
                : ( am > ax ? am - 1 : am ) );

            const auto  dl_l = geo.dx_f( am, i_m );
            const auto   e_l = e_m[ am ]  * dl_l;
            ++ i_m[ am ] ;
            auto       * e_p = b_d.fld.ee.at( i_m );
            const auto  dl_r = geo.dx_f( am,  i_m );
            const auto   e_r = e_p[ am ] * dl_r;
            -- i_m[ am ] ;
            e_b[ ab ]  = ( e_l + e_r ) / ( dl_l + dl_r );
            //  Write-back onto the fine boundary edges: the
            //  fine boundary-row E comes from never-written
            //  transverse flux ghost rows; this keeps the
            //  shared face consistent fine vs coarse.
            e_m[ am ]  = e_b[ ab ];
            e_p[ am ]  = e_b[ ab ];
        }
        i_b [ axis[ 1 ] ] += blockDim.x;
        i_b0[ axis[ 1 ] ] += blockDim.x;
    }
    return ;
}

template < class prx_T , class  derived_T >
template < class bdt_T >
__device__ void comm_t < prx_T, derived_T >::flux_from_buf
( const bdt_T & b_d, const int & step ) const
{
    idx_t i_m, i_b;
    get_self(  ).dispatch( i_b );
    const auto &  ax = axis[ 0 ];
    if( i_b[ axis[ 2 ] ] < b_fcom.n_cell[ axis[ 2 ] ] )
        super_t::flux_from_buf( b_d, step );

    for( int a = 0 ; a < b_d.geo.n_dim; ++ a )
         i_m [ a ] = i_b[ a ] + offset_er[ a ];

    const int  j_max( ( blockIdx.x == gridDim.x - 1 ) + 1 );
    const int ib_max( b_ecom.n_cell[ axis[ 1 ] ] );
    const auto & ib1( i_b          [ axis[ 1 ] ] );
    for( int j = 0; j < j_max && ib1 < ib_max; ++ j )
    {
        auto * e_b = b_ecom    .at( i_b );
        auto * e_m = b_d.fld.ee.at( i_m );
        for( int a = 0; a < 3; ++ a )
        {
            const int am( axis[ utils::mod3( ax + a ) ] );
            if( am == ax )  //  ^-- interface-normal: absent
                continue;
            e_m[ am ] = same_rank ? e_b[ am ]
                : e_b[ am > ax ? am - 1 : am ];
        }
        i_b[ axis[ 1 ] ] += blockDim.x;
        i_m[ axis[ 1 ] ] += blockDim.x;
    }
    return;
}

template < class prx_T , class  derived_T >
template < class bdt_T >
__device__ void comm_t < prx_T, derived_T >::edge_efield
( const bdt_T & b_d, const int & step ) const
{
    idx_t i_b, i_m;  // v-- mode 1: axis[ 1 ] for extension
    const auto & a1( axis[ 1 ] );
    get_self(  ).dispatch( i_b );
    const auto & ee = b_d.fld.ee;

    if( d_lvl > 0 )                // Receiving
    {
        for( int a = 0; a < 3; ++  a )
            i_m[ a ] = offset_er [ a ] + i_b  [ a ];
        ee.at( i_m )[ a1 ] = b_ecom.at ( i_b )[ 0 ];
        return;
    }

    for( int a = 0; a < 3; ++ a )  // Packing and sending
    {
        i_m[ a ] = offset_es[ a ] + 2 * i_b[ a ];
        i_b[ a ] = offset_ec[ a ] +     i_b[ a ];
    }
    auto       * e_l = b_d.fld.ee.at( i_m );
    const auto & dl_l = b_d.geo.dx_f( a1, i_m );
    auto e_tgt = e_l[ a1 ] * dl_l;
    ++ i_m[ a1 ] ;
    auto       * e_r = b_d.fld.ee.at( i_m );
    const auto & dl_r = b_d.geo.dx_f( a1, i_m );
    e_tgt     += e_r[ a1 ] * dl_r;
    const auto   e_avg = e_tgt / ( dl_l + dl_r );
    b_ecom.at( i_b )[ same_rank ? a1 : 0 ] = e_avg;
    e_l[ a1 ] = e_avg;   // Write-back: the fine boundary
    e_r[ a1 ] = e_avg;   // edge == the coarse side.
    return;
}

};                    // namespace dynamics::mhd::boundary
