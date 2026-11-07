#pragma once

#include "comm.h"

namespace dynamics::hydro::boundary
{
////////////////////////////////////////////////////////////
// Memory (de-)allocation

template< class prx_T , class derived_T >
template< class  d3_T > __host__ void comm_t
< prx_T, derived_T >::alloc_one
( d3_T & a, const holder_c_t & h )
{
    a.init( h.p_dev->f_malloc, a.n_cell ,
            idx_t::null(    ), a.n_int );
    return;
}

template< class prx_T, class derived_T > __host__ void
comm_t< prx_T, derived_T >::free( const holder_c_t & h )
{
    auto f_free = [ & ]  ( f_t * & f )
    {  h.p_dev->free_device( f ) ;   };
    b_send.free ( f_free );     // Proxy delete avoided;
    b_fcom.free ( f_free );     // For details see dat_3d.h
    b_recv.free ( f_free );
    return;
}

////////////////////////////////////////////////////////////
// Initialization

template< class prx_T, class derived_T > __host__
void  comm_t  < prx_T, derived_T >::set_geo
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    mesh::boundary::comm_base_t::set_geo( h, nb, mod );
    order  = mod.space_order(  );
    same_rank  = h.same_rank(  );    
    
    const auto & geo( h.p_d->geo(  ) );
    const auto & ax ( axis[ 0 ] );

    for( int a = 0; a < geo.n_dim; ++ a )
        offset_s[ a ] = std::max( 0, nb[ a ] )
                      * ( geo.n_ceff [ a ] / 2 );
    // Specifically for odd-number spatial order
    offset_s[ ax ] -=   ( nb[ ax ] > 0 )
        * ( ( order & 1 ) && d_lvl > 0 ? ( order + 1 ) / 2
            : utils::shift( order, -d_lvl ) );

    if( h.same_rank(  ) )
    {
        const auto & nr ( nb.reverse (  ) ) ;
        for( int a = 0; a < geo.n_dim; ++ a )
            offset_c[ a ] = std::max( 0,  nr[ a ] )
                          * ( geo.n_ceff[ a ] / 2 );
        offset_c[ ax ]
            = nr[ ax ] < 0 ? -order : geo.n_ceff[ ax ];
    }
    for( int a = 0; a < geo.n_dim; ++ a )
        offset_r[ a ] = std::max( 0,  nb[ a ] )
                      * ( geo.n_ceff[ a ] / 2 );
    offset_r[ ax ] =
        ( nb[ ax ] < 0 ? -order : geo.n_ceff[ ax ] );
    return;
}

template< class prx_T, class derived_T > __host__
void  comm_t  < prx_T, derived_T >::set_mem
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    if( h.same_rank(  ) )
        return;

    const auto & b_d  = prx_T::d_i( h.p_d->d(  ) ) ;
    for( auto * p : { & b_send, & b_recv, & b_fcom } )
         p->n_int = b_d.hyd.n_fld (  );

    const auto & geo( h.p_d->geo(  ) );
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        if( nb[ a ] < 0 || nb[ a ] >  1 )
        {
            b_send.n_cell[ a ] = order;
            b_recv.n_cell[ a ] = order;
            b_fcom.n_cell[ a ] =     1;
            continue;
        }
        const auto & n( geo.n_ceff[ a ] );
        b_recv.n_cell[ a ] = n / ( d_lvl >  0 ? 2 : 1 );
        b_send.n_cell[ a ] = n / ( d_lvl <  0 ? 2 : 1 );
        b_fcom.n_cell[ a ] = n / ( d_lvl != 0 ? 2 : 1 );
    }
    alloc_one( b_send, h ) ;
    alloc_one( b_recv, h ) ;
    alloc_one( b_fcom, h ) ;
    return;
}

template< class prx_T, class derived_T > __host__
void  comm_t  < prx_T, derived_T >::set_local
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    if( ! h.same_rank(  ) )
        return;
    const auto & bdg
        ( prx_T::d_i( mod.data( nb.guest_reg ).d(  ) ) );
    b_send = bdg.hyd.w;
    b_fcom = bdg.hyd.f[ axis[ 0 ] ];
    return ;
}

template< class prx_T, class derived_T > __host__
void  comm_t  < prx_T, derived_T >::set_comm
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    if( h.same_rank(  ) )
        return;
    if( h.d_lvl < 0 && mode == 0 )
        h.f_fcom = [ & ] (  )
        {
            h.p_com->isend_d( b_fcom.dat, b_fcom.n_size ,
                              h.rank, h.tag_f, h.stream );
        };
    else if( h.d_lvl > 0 && mode == 0 )
        h.f_fcom = [ & ] (  )
        {
            h.p_com->irecv_d( b_fcom.dat, b_fcom.n_size ,
                              h.rank, h.tag_f, h.stream );
        };
    h.f_recv = [ & ] (  )
    {
        h.p_com->irecv_d( b_recv.dat, b_recv.n_size ,
                          h.rank, h.tag_r, h.stream );
    };
    h.f_send = [ & ] (  )
    {
        h.p_com->isend_d( b_send.dat, b_send.n_size ,
                          h.rank, h.tag_s, h.stream );
    };
    return;
}

template< class prx_T, class derived_T >    __host__
void  comm_t  < prx_T, derived_T >::set_launch
( holder_c_t  & h, const neighbor_t & nb, mod_base_t & mod )
{
    auto & self   = get_self          (       );
    auto par_pack = self.resource_pack( h, nb );
    auto & d( * h.p_d );

    auto  guest( nb.guest );
    mod.p_mesh-> tree(  ) . regularize  ( guest );
    const auto & str_dest = ( ! h.same_rank(  ) )
               ? d.stream : mod.data( guest ).stream;
    h.f_pack = [ &, par_pack ] ( const int &   step )
    {
        const auto & b_d  =    prx_T::d_i( d.d(   ) );
        h.p_dev->event_wait    ( h.stream, d. event );
        h.p_dev->launch ( ker_to_buf,      par_pack  ,
                          h.stream, b_d, self, step );
        h.p_dev->event_record  ( h.event,  h.stream );
        h.p_dev->event_wait    ( str_dest, h. event );
    };  // The destination blk depends on this
    if( ! h.same_rank(  ) )
    {
        const auto par_unpk = self.resource_unpk( h,   nb );
        h.f_unpk = [ &, par_unpk ]   ( const  int &  step )
        {
            const auto      & b_d =  prx_T::d_i( d.d(   ) );
            h.p_dev->launch ( ker_from_buf,      par_unpk  ,
                              h.stream, b_d,  self,  step );
            h.p_dev->event_record    ( h. event, h.stream );
            h.p_dev->event_wait      ( d.stream, h. event );
            h.p_dev->event_record    ( d. event, d.stream );
        };
    }
    if( d_lvl != 0 && mode == 0 )
    {
        const auto par_flux = self.resource_flux ( h,  nb );
        h.f_fpck = [ &, par_flux ]   ( const int &   step )
        {
            h.p_dev->event_wait      ( h.stream, d. event );
            const auto     & b_d =   prx_T::d_i( d.d(   ) );
            h.p_dev->launch( ker_flux, par_flux, h.stream  ,
                             b_d,             self,  step );
            h.p_dev->event_record    ( h. event, h.stream );
            h.p_dev->event_wait      ( d.stream, h. event );
            if( h.same_rank(  ) && h.d_lvl < 0 )
                h.p_dev->event_wait  ( str_dest, h. event );
        };
    }
    return;
}

////////////////////////////////////////////////////////////
// Job dispatching

template< class prx_T, class derived_T > __device__
void comm_t< prx_T, derived_T >::dispatch( idx_t & i ) const
{
    i[ axis[ 0 ] ] = 0;
    i[ axis[ 1 ] ] = blockIdx.x * blockDim.x + threadIdx.x;
    i[ axis[ 2 ] ] = blockIdx.y ;
    return;
}

template< class prx_T, class derived_T >
template< class     fun_T >  __device__ __forceinline__
void    comm_t< prx_T, derived_T > ::map_refine
( const fun_T & f, const int & n_dim ) const
{
    if     ( n_dim == 3 )
        for( const auto & da1 : { 0, 1 } )
            for( const auto & da2 : { 0, 1 } )
                f( da1, da2 );
    else if( n_dim == 2 )
        for( const auto & da1 : { 0, 1 } )
            f( da1, 0 );
    else
        f( 0, 0 );
    return;
}

////////////////////////////////////////////////////////////
// Resources

template  < class prx_T, class derived_T > __host__
std::tuple< int, int > comm_t< prx_T, derived_T >::
resource_thbl( const idx_t  & n_cell )  const
{
    const int n_th = std::min
            ( n_cell[ axis[ 1 ] ], this ->n_th_lim );
    return std::make_tuple
        ( n_th, ( n_th - 1 + n_cell[ axis[ 1 ] ] ) / n_th );
}   // Along the direction of extension

template  < class prx_T, class derived_T >
__host__ std::tuple< dim3, dim3, int >
comm_t< prx_T, derived_T >::resource_pack
( const holder_c_t & h, const neighbor_t & nb )  const
{
    // #warning "TODO: pass b_d from launch to here"
    const auto & b_d  = prx_T::d_i ( h.p_d->d(  ) );
    const auto & geo  ( b_d.   geo );
    idx_t n_cell      ( geo.n_ceff );
    if( mode > 0 )
        n_cell[ axis[ 2 ] ] = order;
    if( mode > 1 )
        n_cell[ axis[ 1 ] ] = order;
    for( int a = 0; a < geo.n_dim; ++ a )
        n_cell[ a ] /= 1 + ( d_lvl != 0 );

    const auto [ n_th, n_bl_x ] = resource_thbl( n_cell ) ;
    // const int n_bl_y( mode == 0 ? n_cell[ axis[ 2 ] ] :
    const int n_bl_y( n_cell[ axis[ 2 ] ] );
    
    const int s_sh = n_th * b_d.hyd.n_fld( ) * sizeof( f_t )
              * ( d_lvl  > 0 ? 4 : ( d_lvl < 0 ? 1 : 0 ) ) ;
    return std::make_tuple
         ( dim3( n_bl_x, n_bl_y, 1 ), n_th, s_sh );
}

template  < class prx_T, class derived_T >
__host__ std::tuple< dim3, dim3, int >
comm_t< prx_T, derived_T >::resource_unpk
( const holder_c_t & h, const neighbor_t & nb )  const
{
    const auto & geo = h.p_d->geo(  );
    auto n_cell = geo.n_ceff;
    if( mode > 0 )
        n_cell[ axis[ 2 ] ] = order;
    if( mode > 1 )
        n_cell[ axis[ 1 ] ] = order;    
    for( int a = 0; a < geo.n_dim; ++ a )
        n_cell[ a ] /= 1 + ( d_lvl > 0 );

    const  auto [ n_th, n_bl_x ] = resource_thbl( n_cell );
    const  int    n_bl_y ( n_cell[ axis[ 2 ] ] );
    return std::make_tuple
         ( dim3( n_bl_x, n_bl_y, 1 ), n_th,  0 );
}

template  < class prx_T, class derived_T >
__host__ std::tuple< dim3, dim3, int >
comm_t< prx_T, derived_T >::resource_flux
( const holder_c_t & h, const neighbor_t & nb )  const
{
    const auto & geo = h.p_d->geo(  );
    auto n_cell = geo.n_ceff;
    for( int a = 0; a < geo.n_dim; ++ a )
        n_cell[ a ] /= 2;
    const auto [ n_th, n_bl_x ] = resource_thbl( n_cell );
    int n_bl_y ( n_cell [ axis[ 2 ] ] );
    return std::make_tuple
         ( dim3( n_bl_x, n_bl_y, 1 ), n_th,  0 );
}

////////////////////////////////////////////////////////////
// Actions of communication boundary

template< class prx_T , class derived_T >
template< class bdt_T > __device__
void   comm_t < prx_T, derived_T >::to_same
( const bdt_T &   b_d, const int & step ) const
{
    const auto & geo( b_d.geo );
    idx_t i_m, i_b;
    get_self(  ).dispatch ( i_b );
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        i_m[ a ]  = offset_s[ a ] + i_b[ a ];
        i_b[ a ] += offset_c[ a ] ;
    }
    for( int i = 0; i < b_d.hyd.order(  ); ++ i )
    {   //   Special offset conventions
        i_m[ axis[ 0 ] ]  = i + offset_s[ axis[ 0 ] ];
        i_b[ axis[ 0 ] ]  = i + offset_c[ axis[ 0 ] ];
        auto       * p_b  = b_send   .at( i_b ) ;
        const auto * p_m  = b_d.hyd.w.at( i_m ) ;
        for( int n = 0; n < b_d.hyd.n_fld(  ); ++ n )
            p_b[ n ] =  p_m [ n ];
    }
    return;
}

template< class prx_T , class derived_T >
template< class bdt_T > __device__
void   comm_t < prx_T, derived_T >::to_coarser
( const bdt_T &   b_d, const int & step ) const
{
    const auto &   geo = b_d.geo;
    const auto &   hyd = b_d.hyd;
    const auto & n_int = b_d.hyd.n_fld(  );
    idx_t i_m,     i_b ;
    get_self(  ).dispatch( i_b );
    for( int a = 0 ; a < geo.n_dim; ++ a )
    {
        i_m[ a ]  = offset_s[ a ] + 2 * i_b[ a ];
        i_b[ a ] += offset_c[ a ] ;
    }
    __dyn_shared__( f_t , var_sh );
    f_t * p_b =  var_sh + threadIdx.x * n_int;
    f_t  norm ;
    const int depth = 2 * order;
    for(  int k = 0 ; k < depth; ++ k )
    {
        i_m[ axis[ 0 ] ] = k + offset_s[ axis[ 0 ] ];
        if ( ( k & 1 )  == 0 )
        {
            i_b [ axis [ 0 ] ]
                = k  / 2 + offset_c[ axis[ 0 ] ];
            norm = 0 ;
            for( int n = 0; n < n_int; ++ n )
                p_b[ n ] =  0 ;
        }
        idx_t i_m0  = i_m ;
        map_refine( [ & ]
                    ( const int & d1, const int & d2 )
        {
            i_m[ axis[ 1 ] ]  = i_m0[ axis[ 1 ] ] + d1;
            i_m[ axis[ 2 ] ]  = i_m0[ axis[ 2 ] ] + d2;
            auto * p_m = hyd. w . at  ( i_m );
            const auto dv = geo.volume( i_m );
            for( int n = 0; n < n_int; ++ n )
                p_b[ n ] += p_m[ n ] * dv;
            norm += dv ;
        },  geo. n_dim );
        i_m = i_m0;
        if( ( k & 1 ) != 0 )
        {
            auto * p_dest = b_send.at(  i_b );
            for( int n = 0; n < n_int; ++ n )
                p_dest[ n ] = p_b[ n ] / norm;
        }
    }
    return;
}

template< class prx_T , class derived_T >
template< class bdt_T > __device__
void   comm_t < prx_T, derived_T >::to_finer
( const bdt_T &   b_d, const int & step ) const
{
    const int     ax = axis[ 0 ];
    const auto & geo = b_d . geo;
    const auto & hyd = b_d . hyd;
    const int  n_int = hyd.n_fld(  );

    __dyn_shared__ ( f_t, var_sh );
    f_t * q_c =   var_sh  + n_int * 4 * threadIdx.x ;
    f_t * dq[ 3 ] = { q_c + n_int,  q_c + 2 * n_int ,
                                    q_c + 3 * n_int };
    idx_t i_b, i_m, i_m0;
    get_self(  ).dispatch( i_m0 ) ;
    for( int a = 0 ; a < geo.n_dim; ++ a )
    {
        i_b[ a ] = offset_c[ a ] + 2 * i_m0[ a ];
        i_m[ a ] = offset_s[ a ] +     i_m0[ a ];
    }

    // v-- For re-using in MHD; step < 0 --> MHD
    const int im_max = ( geo.n_ceff [ axis[ 1 ] ]
                       + ( step < 0 ) * order ) / 2;
    const int  j_max( ( blockIdx.x == gridDim.x - 1 ) + 1 );
    const int  & im1( i_m0[ axis [ 1 ] ] );
    for( int j = 0; j < j_max && im1 < im_max; ++ j )
    {
        for( int k = 0; k < order; ++ k )
        {
            i_m[ ax ] = offset_s[ ax ] + k / 2;
            i_b[ ax ] = offset_c[ ax ] +     k;
            if( ( k & 1 ) == 0 )
            {
                for( int n = 0; n  <  n_int; ++ n )
                    q_c[ n ] = hyd.w.at( i_m )  [ n ];
                for( int a = 0; a < geo.n_dim; ++ a )
                {
                    i_m[ axis[ a ] ] += 1;
                    const auto * q_r  = hyd.w.at( i_m );
                    i_m[ axis[ a ] ] -= 2;
                    const auto * q_l  = hyd.w.at( i_m );
                    i_m[ axis[ a ] ] += 1;
                    for( int n = 0; n < n_int; ++ n )
                        dq[ axis[ a ] ][ n ] = utils::minmod
                            ( q_r[ n ] - q_c[ n ] ,
                              q_c[ n ] - q_l[ n ] ) * 0.25f;
                }
            }
            map_refine( [ & ]
                        ( const int & d1, const int & d2 )
            {
                i_b[ axis[ 1 ] ] += d1;
                i_b[ axis[ 2 ] ] += d2;
                auto * q = b_send.at( i_b  );
                for( int n = 0; n < n_int; ++ n )
                    q[ n ] = q_c[ n ]
                    + dq[ ax ][ n ] * ( 2 * ( k & 1 ) - 1 )
                    + dq[ axis[ 1 ] ][ n ] * ( 2 * d1 - 1 )
                    + dq[ axis[ 2 ] ][ n ] * ( 2 * d2 - 1 );
                i_b[ axis[ 1 ] ] -= d1;
                i_b[ axis[ 2 ] ] -= d2;
            },  geo.n_dim  ) ;
        }
        i_b [ axis[ 1 ] ] += 2 * blockDim.x;
        i_m [ axis[ 1 ] ] +=     blockDim.x;
        i_m0[ axis[ 1 ] ] +=     blockDim.x;
    }
    return;
}

template< class prx_T , class derived_T >
template< class bdt_T > __device__
void   comm_t < prx_T, derived_T >::from_buf
( const bdt_T &   b_d, const int & step ) const
{
    idx_t i_m, i_b;
    get_self(  ).dispatch( i_b );
    for( int a = 0 ; a  < i_m.size(  ); ++ a )
        i_m[ a ] =   i_b[ a ] + offset_r [ a ] ;

    // v-- For re-using in MHD
    const int ib_max( ( b_recv.n_cell [ axis[ 1 ] ] ) );
    const int  j_max( ( blockIdx.x == gridDim.x - 1 ) + 1 );
    const int  & ib1( i_b[ axis [ 1 ] ] );
    for( int j = 0; j < j_max && ib1 < ib_max; ++ j )
    {    
        for( int k = 0; k < b_d.hyd.order(  ); ++ k )
        {
            i_m[ axis[ 0 ] ]  = k + offset_r[ axis[ 0 ] ];
            i_b[ axis[ 0 ] ]  = k ;
            const auto * p_buff = b_recv   .at( i_b );
            auto       * p_mesh = b_d.hyd.w.at( i_m );
            for( int n = 0; n < b_d.hyd.n_fld(  ); ++ n )
                p_mesh[ n ] =   p_buff  [ n ];
        }
        i_m[ axis[ 1 ] ] += blockDim.x;
        i_b[ axis[ 1 ] ] += blockDim.x;
    }
    return;
}

////////////////////////////////////////////////////////////
// Fluxes

template< class prx_T , class derived_T >
template< class bdt_T > __device__
void   comm_t < prx_T, derived_T >::flux_to_buf
( const bdt_T &   b_d, const int & step ) const
{
    const int  n_int = b_d.hyd.w.n_int;
    const auto & geo = b_d.  geo;
    const int     ax = axis[ 0 ];
    idx_t i_m,   i_b , i_b0;
    get_self(  ).dispatch( i_b0 );
    for( int a = 0; a < geo.n_dim; ++ a )
        i_b[ a ] = i_b0[ a ] + offset_c [ a ];
    i_b [ ax ] = ( ! same_rank ? 0 :
                 ( rhs ? 0 : b_d.geo.n_ceff[ ax ] ) );
    i_m [ ax ] = ( rhs ? b_d.geo.n_ceff[ ax ] : 0 )  ;

    auto * f_b = b_fcom.at( i_b );
    for( int n = 0 ; n < n_int; ++ n )
        f_b[ n ] = 0 ;
    f_t norm ( 0 ) ;
    map_refine( [ & ] ( const int & d1, const int & d2 )
    {
        i_m [ axis[ 1 ] ] = 2 * i_b0[ axis[ 1 ] ] + d1;
        i_m [ axis[ 2 ] ] = 2 * i_b0[ axis[ 2 ] ] + d2;
        auto * f_m = b_d.hyd.f  [ ax ].at  ( i_m );
        auto ds    = geo.surface( i_m, axis[ 0 ] );
        for( int n = 0;  n < n_int; ++ n )
            f_b[ n ] += f_m [ n ] * ds;
        norm += ds ;
    },  geo.n_dim );
    for( int n = 0 ; n < n_int; ++ n )
        f_b[ n ] /= norm;


    return ;
}


template< class prx_T , class derived_T >
template< class bdt_T > __device__
void   comm_t < prx_T, derived_T >::flux_from_buf
( const bdt_T &   b_d, const int & step ) const
{
    const int  &  ax = axis[ 0 ];
    const auto & geo = b_d.  geo;
    idx_t i_m, i_b;
    get_self(  ).dispatch ( i_b ) ;
    for( int a = 0; a < geo.n_dim; ++ a )
        i_m[ a ] =  i_b [ a ] + offset_r[ a  ] ;
    i_m[ ax ]  = (  rhs ? b_d.geo.n_ceff[ ax ] : 0 ) ;
    i_b[ ax ]  = 0;
    auto * f_b = b_fcom.at( i_b );
    auto * f_m = b_d.hyd.f[ axis[ 0 ] ].at( i_m );
    for( int n = 0 ; n  < b_d.hyd.w.n_int; ++ n )
        f_m[ n ] = f_b[ n ];
    return;
}

};  // namespace boundary::hydro::dynamics
