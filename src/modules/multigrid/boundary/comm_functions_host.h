#pragma once

#include "../../../utilities/functions.h"
#include "comm.h"

namespace multigrid::boundary
{
////////////////////////////////////////////////////////////
// Memory (de-)allocation

template< class prx_T, class derived_T >
template< class d_T > __host__  void comm_t
< prx_T, derived_T >::alloc( d_T & a, const holder_c_t & h )
{
    a.init( h.p_dev->f_malloc, a.n_cell,
            idx_t::null(    ), a.n_int );
    return h.p_dev->f_mset( a.dat, 0, a.n_size, 0 );
}

template< class  prx_T,  class derived_T > __host__ void
comm_t  < prx_T, derived_T >::free( const holder_c_t & h )
{
    auto f_free = [ & ] ( auto * & f )
    {  h.p_dev->free_device( f ) ;   };
    b_send.free ( f_free );     // Proxy delete avoided;
    b_recv.free ( f_free );
    return;
}

template< class prx_T, class  derived_T >
__host__  void comm_t< prx_T, derived_T >::set_mem
( holder_c_t & h, const neighbor_t & nb, mod_base_t & mod )
{
    auto  & d( prx_T::dual( * h.p_d ) );
    auto  & geo( d.b_w. get(  ).geo );
    m_dim = utils::shift( 1, geo.n_dim );
    n_fld = prx_T::d_i( d.h(  ) ).mgd.n_fld (  ) ;
    for( auto * p : { & b_send, & b_recv } )
        p->n_int = n_fld;
    order  = d.order;
    n_ceff_s = idx_t( 1 ) ;

    if( h.same_rank(  ) )
        return;

    for( int a = 0; a < geo.n_dim; ++ a )
    {
        if( nb[ a ] < 0 || nb[ a ] >  1 )
        {
            b_send.n_cell[ a ] = order;
            b_recv.n_cell[ a ] = order;
            continue;
        }
        b_send.n_cell[ a ] = geo.n_ceff[ a ];
        b_recv.n_cell[ a ] = geo.n_ceff[ a ];
    }
    alloc( b_send, h );
    alloc( b_recv, h );
    return;
}


template< class prx_T, class derived_T > __host__
void    comm_t< prx_T, derived_T >:: set_launch
( holder_c_t & h, const neighbor_t & nb, mod_base_t & mod )
{
    auto & self = get_self_c(  );
    const auto & d ( prx_T::dual( * h.p_d ) );
    auto  *  q_g   ( & d );
    if( h.same_rank(   ) )
        q_g  = & prx_T::dual( mod.data ( nb.guest_reg ) );
    auto & dev ( *  h.p_dev );
    h.f_pack = [ &, q_g ] ( const int & l )
    {
        self.adjust_lvl ( l,   h,  nb,  mod );
        auto par = self.resource_pack( h, l );
        dev.event_wait  ( h.stream, d.event );
        const auto & bl ( d.d_mg( l ) );
        if( d_lvl == 0 )
            dev.launch( ker_same, par, h.stream, bl, self );
        else if( d_lvl > 0 )
        {
            if( l == 0 )
                dev.launch( ker_finer_finest, par,
                            h.stream, bl, self );
            else
                dev.launch( ker_finer,        par,
                            h.stream, bl, self );
        }
        else if( d_lvl < 0 )
            dev.launch( ker_coarser, par,
                        h.stream, bl, self );
        if( ! same_rank )
        {
            h.p_com->isend_d( b_send.dat, b_send.n_size,  h.
                              rank, h.tag_s - l, h.stream );
            h.p_com->irecv_d( b_recv.dat, b_recv.n_size,  h.
                              rank, h.tag_r - l, h.stream );
            dev.stream_record ( d   .stream,     h.stream );
        }
        else
            dev.stream_record ( q_g->stream,     h.stream );
    };
    if( ! same_rank || d_lvl < 0 )
    {
        h.f_unpk = [ & ] ( const int & l )
        {
            if( l != 0 && same_rank )
                return;
            self.adjust_lvl ( l, h, nb, mod );
            const auto & bdt( d.d_mg  ( l ) );
            const auto par = self.resource_unpk( h, l );
            dev.sync_stream( d.stream );
            if( l == 0 )
                dev.launch( ker_from_buf_finest, par,
                            d.stream,    bdt,    self );
            else
                dev.launch( ker_from_buf,        par,
                            d.stream,    bdt,    self );
        };
    }
    return;
}

////////////////////////////////////////////////////////////
// Job dispatching and resources

template< class prx_T, class derived_T > __host__
std::tuple< int, int > comm_t< prx_T, derived_T >::
resource_thbl( const idx_t & n_cell )  const
{
    const int n = n_cell[ axis[ 1 ] ] * n_cell[ axis[ 2 ] ];
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( ( n + t - 1 ) / t, t );
}   // Along the direction of extension

template< class prx_T, class derived_T > __host__
std::tuple< dim3, dim3, int > comm_t< prx_T, derived_T >::
resource_pack( const holder_c_t & h, const int & l )
{
    const auto & d =  prx_T::dual( * h.p_d );
    const auto & geo  ( d.b_w.get(   ).geo );
    const auto [ n_bl , n_th ] = resource_thbl( n_ceff_s );
    const int s_fld =   n_fld * sizeof( f_t );
    const int s_sh  = ( d_lvl != 0 )  * s_fld;
    return std::make_tuple
         ( dim3( n_bl ), dim3( n_th ) , s_sh * n_th );
}

template< class prx_T, class derived_T > __host__
std::tuple< dim3, dim3, int > comm_t< prx_T, derived_T >::
resource_unpk( const holder_c_t & h, const int & l )
{
    if( ! same_rank )
        n_ceff_r = b_recv.n_cell;
    else if( l == 0 )
        n_ceff_r = prx_T::dual( * h.p_d )
                 . b_w.get(  ).geo.n_ceff;
    else
        throw std::runtime_error( "Should not reach here" );
    const auto & [ n_bl, n_th ] = resource_thbl( n_ceff_r );
    return std::make_tuple( n_bl, dim3 ( n_th ), 0 );
}

template< class prx_T, class  derived_T >
__host__  void comm_t< prx_T, derived_T >::adjust_lvl
( const        int &  l, holder_c_t &   h ,
  const neighbor_t & nb, mod_base_t & mod )
{
    const auto &   d = prx_T::dual( * h.p_d );
    const auto & geo = d.b_w.get(  ).geo;
    const int    fac = 1 << std::max( -l, 0 );
    this->lvl        = l ;
    const int  ord_2 = order + ( order & 1 );
    const bool  stag = ( l != 0 && d_lvl > 0 );
    for( int a = 0 ; a < geo.n_dim; ++ a )
    {
        offset_s[ a ] = geo.n_ceff[ a ] / fac / 2
            * std::max( nb[ a ] , 0 ) - ( nb[ a ] == 2 )
            * ( stag ? ord_2 / 2 : order
                * ( d_lvl < 0 ? 2 : 1 ) );

        n_ceff_s[ a ] = geo.n_ceff[ a ] / fac
                      / ( d_lvl !=  0 ? 2 : 1 );
    }
    const auto & nb_r( nb.reverse(  ) );
    if( h.same_rank(  ) )
    {
        auto & d_g ( prx_T::dual
                   ( mod.data( nb.guest_reg ) ) );
        b_send = d_g.d_mg( l ).mgd.fld;
        for( int a  = 0 ; a < 3; ++ a )
            offset_c[ a ] = geo.n_ceff[ a ] / fac
                * std::max( nb_r[ a ] , 0 ) / 2
                - ( stag ? ord_2 : order )
                * ( nb[ a ] == 2 );
    }
    else
    {
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            offset_r[ a ] = geo.n_ceff[ a ] / fac / 2
                          * std::max ( nb[ a ], 0 )
                          - order * ( nb[ a ] < 0 );
            if( nb[ a ] < 0 || nb[ a ] >  1 )
            {
                b_send.n_cell[ a ] = order;
                b_recv.n_cell[ a ] = order;
                continue;
            }
            b_send.n_cell[ a ] = geo.n_ceff[ a ] / fac
                               / ( ( d_lvl < 0 ) + 1 );
            b_recv.n_cell[ a ] = geo.n_ceff[ a ] / fac
                               / ( ( d_lvl > 0 ) + 1 );
        }
        offset_c[ axis[ 0 ] ] = stag
            * ( order - ord_2 ) * ( nb[ axis[ 0 ] ] >  0 );

        for( auto * p_b : { & b_send, & b_recv } )
        {
            p_b->n_size = p_b->n_cell.prod(  ) * p_b->n_int;
            if( p_b->n_size  >  p_b->n_ctot )
                throw std::runtime_error ( "adjust_lvl" );
            p_b->n_size *= type::size( p_b->dat );
        }
    }
    return;
}

};                   // namespace multigrid::boundary
