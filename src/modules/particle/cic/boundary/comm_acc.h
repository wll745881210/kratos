#pragma once

#include "../../../../mesh/boundary/boundary.h"
#include "comm_kernel.h"

namespace particle::cic::boundary::acc
{
////////////////////////////////////////////////////////////
// Communication boundary for fields

template< class prx_T, class derived_T = crtp::dummy_t >
struct    comm_t  : public mesh::boundary::comm_base_t
{
    ////////// Types //////////
    __crtp_def_self__( comm_t, derived_T, prx_T );
    using     f_t = type::              float_t;
    using super_t = mesh::boundary::comm_base_t;

    ////////// Data //////////
    using   super_t::order;
    dat_3d_t< f_t > b_send;
    dat_3d_t< f_t > b_recv;

    ////////// Host-side functions //////////
    __host__ virtual void free ( const holder_c_t & h )
    {
        auto f_free = [ & ]  ( f_t * & f )
        {  h.p_dev->free_device( f ) ;   };
        b_send.free ( f_free );
        b_recv.free ( f_free );
        return;
    };

    __host__ virtual void set_geo
    ( holder_c_t &   h, const neighbor_t & nb,
      mod_base_t & mod )
    {
        mesh::boundary::comm_base_t::set_geo( h, nb, mod );

        const auto & geo( h.p_d->geo(  ) );
        same_rank  = h.same_rank(  );
        order      = mod.space_order(  );
        const auto & nr( nb.reverse (  ) );

        const auto ord_2( order + ( order & 1 ) );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            const  auto n_2 ( geo.n_ceff[ a ] / 2 ) ;
            offset_s[ a ] = std::max( nb[ a ] , 0 ) * n_2;
            offset_r[ a ] = std::max( nb[ a ] , 0 ) * n_2;
            offset_c[ a ] = std::max( nr[ a ] , 0 ) * n_2;

            offset_s[ a ]  -= ( nb[ a ] > 0 ) 
                * ( d_lvl < 0 ?   order * 2 :
                  ( d_lvl > 0 ?   ord_2 / 2 : order ) ) ;
            offset_c[ a ]  -= ( nb[ a ] > 0 )
                * ( d_lvl > 0 ? ( ord_2 / 2 ) * 2 : order );
            offset_r[ a ]  -= ( nr[ a ] > 0 )
                * ( d_lvl < 0 ? ( ord_2 / 2 ) * 2 : order );
            offset_c[ a ] *= same_rank;
        }
        return;
    };

    __host__ virtual void set_mem
    ( holder_c_t &   h, const neighbor_t & nb,
      mod_base_t & mod )
    {
        if( h.same_rank(  ) )
            return;

        const auto & b_d
            = prx_T::d_i( h.p_d->d(  ) );
        for( auto * p : { & b_send, & b_recv } )
            p->n_int = b_d.cic.acc.n_int;

        const auto & geo( h.p_d->geo(  ) );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            if( nb[ a ] < 0 || nb[ a ] >  1 )
            {
                b_send.n_cell[ a ] = order;
                b_recv.n_cell[ a ] = order;
                continue;
            }
            const auto & n( geo.n_ceff[ a ] );
            b_recv.n_cell[ a ] = n / ( d_lvl > 0 ? 2 :  1 )
                                   + ( d_lvl < 0 ) * order;
            b_send.n_cell[ a ] = n / ( d_lvl < 0 ? 2 :  1 )
                                   + ( d_lvl > 0 ) * order;
        }
        for( auto * p : { & b_send, & b_recv } )
            p->init( h.p_dev->f_malloc, p->n_cell ,
                     idx_t::null(    ), p->n_int );
        return;
    };

    __host__ virtual  void  set_comm
    ( holder_c_t & h, const neighbor_t & nb,
      mod_base_t & mod )
    {
        if( h.same_rank(  ) )
            return;
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
    };

    __host__ virtual  void  set_local
    ( holder_c_t & h, const neighbor_t & nb,
      mod_base_t & mod )
    {
        if( ! h.same_rank(  ) )
            return;
        const auto & bdg
        ( prx_T::d_i( mod.data( nb.guest_reg ).d(  ) ) );
        b_send = bdg.cic.acc;
        return ;
    };

    __host__ virtual  void  set_launch
    ( holder_c_t & h, const neighbor_t & nb,
      mod_base_t & mod )
    {
        auto & self = get_self_c(  );
        auto & d    ( * h.p_d );
        auto par_pack = self.resource_pack( h, nb );

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
        if( h.same_rank(  ) )
            return;

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
        return;
    };

    __host__ virtual std::tuple< dim3, dim3, int >
    resource_pack( const holder_c_t & h,
                   const neighbor_t & nb )
    {
        const auto & b_d = prx_T::d_i( h.p_d->d(  ) );
        const auto & geo  ( b_d.   geo );
        n_ceff_s = geo.n_ceff ;
        for( int a = 0; a < geo.n_dim; ++ a )
            n_ceff_s[ a ] /= 1 + ( d_lvl != 0 );

        const auto & nr( nb.reverse(  ) );
        if( mode == 0 && d_lvl > 0 )
            for( const auto & i_a : { 1, 2 } )
                n_ceff_s[ axis[ i_a ] ]
                     += ( order + 1 ) / 2;
        else if( mode == 1 )
        {
            n_ceff_s[ axis [ 1 ] ]
                 += ( order + 1 ) / 2 * ( d_lvl > 0 );
            n_ceff_s[ axis [ 2 ] ]
                   = order  / ( d_lvl > 0 ? 2 : 1 );
        }
        else if( mode == 2 )
            for( int a = 0; a < 3; ++ a )
                n_ceff_s[ a ] = order
                              / ( d_lvl > 0 ? 2 : 1 );
        
        auto [ n_th, n_bl_x ] = resource_thbl( n_ceff_s );
        const int n_bl_y ( n_ceff_s[ axis[ 2 ] ] );
        const int s_sh = n_th * b_d.cic.acc.n_int
                * sizeof( f_t ) * ( d_lvl != 0  );
        return std::make_tuple
            ( dim3( n_bl_x, n_bl_y, 1 ), n_th, s_sh );
    };

    __host__ virtual std::tuple< dim3, dim3, int >
    resource_unpk( const holder_c_t & h,
                   const neighbor_t & nb )
    {
        const auto & geo = h.p_d->geo(  );
        n_ceff_r = geo.n_ceff;
        for( int a = 0; a < geo.n_dim; ++ a )
             n_ceff_r[ a ] /= 1 + ( d_lvl > 0 );

        const auto & nr( nb.reverse(  ) );
        if( mode == 0 && d_lvl < 0 )
            for( const auto & i_a : { 1, 2 } )
                n_ceff_r[ axis[ i_a ] ] += order;
        else if( mode == 1 )
        {
            n_ceff_r[ axis[ 1 ] ] += order * ( d_lvl < 0 );
            n_ceff_r[ axis[ 2 ] ]  = order ;
        }
        else if( mode == 2 )
            for( int a = 0; a < 3; ++ a )
                n_ceff_r[ a ] = order;

        auto [ n_th, n_bl_x ] = resource_thbl( n_ceff_r );
        const  int n_bl_y( n_ceff_r[ axis[ 2 ] ] );
        return std::make_tuple
            ( dim3( n_bl_x, n_bl_y, 1 ), n_th, 0 );
    };

    ////////// Device-side functions //////////
    __device__ bool dispatch
    ( idx_t & i, const idx_t & n_ceff ) const
    {
        i[ axis[ 0 ] ] =  0;
        i[ axis[ 1 ] ] =  blockIdx.x * blockDim.x
                       + threadIdx.x ;
        i[ axis[ 2 ] ] =  blockIdx.y ;
        return i[ axis[ 1 ] ] < n_ceff[ axis[ 1 ] ];
    };

    template< class bdt_T >  __device__ void
    to_same ( const bdt_T & b_d, const int & step ) const
    {
        idx_t i_m, i_b;
        if( ! get_self(  ).dispatch ( i_b, n_ceff_s ) )
            return;
        const auto &   geo( b_d.geo );
        const auto & n_int( b_d.cic.acc.n_int );        
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            i_m[ a ]  = offset_s[ a ] + i_b[ a ];
            i_b[ a ] += offset_c[ a ] ;
        }
        for( int i = 0; i < order; ++ i )
        {   //   Special offset conventions
            i_m[ axis[ 0 ] ]  = i + offset_s[ axis[ 0 ] ];
            i_b[ axis[ 0 ] ]  = i + offset_c[ axis[ 0 ] ];
            auto       * p_b  = b_send     .at( i_b ) ;
            const auto * p_m  = b_d.cic.acc.at( i_m ) ;
            for( int n = 0; n < n_int; ++ n )
                p_b[ n ] = p_m[ n ];
        }
        return;
    };

    template< class bdt_T > __device__ void
    to_coarser( const bdt_T & b_d, const int & step ) const
    {
        idx_t i_m, i_m0, i_b, di ;
        if( ! get_self(  ).dispatch( i_b, n_ceff_s ) )
            return;
        const auto &   geo = b_d.geo;
        const auto &   cic = b_d.cic;
        const auto & n_int = b_d.cic.acc.n_int;        
        for( int a = 0 ; a < geo.n_dim; ++ a )
        {
            i_m0[ a ]  = offset_s[ a ] + 2 * i_b[ a ];
            i_b [ a ] += offset_c[ a ] ;
        }
        __dyn_shared__( f_t , var_sh );
        f_t * p_b =  var_sh + threadIdx.x * n_int;
        f_t  norm ;

        const int m_dim( 1 << geo.n_dim );
        for( int k = 0; k < order; ++ k )
        {
            norm = 0;
            for( int n = 0; n < n_int; ++ n )
                p_b[ n ] =  0 ;
            for( int l = 0; l < m_dim; ++ l )
            {
                utils::muldim( di, l, geo.n_dim );
                for( int a = 0 ; a <  geo.n_dim; ++ a )
                    i_m[ a ] = di[ a ] + i_m0[ a ];
                i_m[ axis[ 0 ] ] +=  k * 2 ;
                auto *    p_m = cic.acc.at ( i_m );
                const auto dv = geo.volume ( i_m );
                for( int n = 0; n < n_int;  ++ n )
                    p_b[ n ] += p_m[ n ] *  dv ;
                norm += dv;
            }
            i_b[ axis[ 0 ] ] = k + offset_c[ axis[ 0 ] ];
            auto * p_dest = b_send.at(  i_b );
            for( int n = 0; n < n_int; ++ n )
                p_dest[ n ] = p_b[ n ] / norm;
        }
        return;
    };

    template < class bdt_T > __device__ void to_finer
    ( const bdt_T  & b_d,  const  int & step )  const
    {
        idx_t i_b, i_m, i_b0,  di;
        if( ! get_self(  ).dispatch( i_b0, n_ceff_s ) )
            return;
            
        const int       ax = axis[ 0 ];
        const auto &   geo = b_d.geo;
        const auto &   cic = b_d.cic;
        const auto & n_int = b_d.cic.acc.n_int;

        __dyn_shared__( f_t, var_sh );
        f_t * q_c = var_sh + n_int * threadIdx.x;
        for( int a = 0 ; a < geo.n_dim; ++ a )
        {
            i_m [ a ] = offset_s[ a ] +     i_b0[ a ];
            i_b0[ a ] = offset_c[ a ] + 2 * i_b0[ a ];
        }
        const int depth( ( order + 1 ) / 2 );
        const int m_dim( 1 <<  geo.n_dim ) ;
        for ( int k = 0; k < depth; ++ k )
        {
            i_m [ ax ] = offset_s[ ax ]  +     k;
            i_b0[ ax ] = offset_c[ ax ]  + 2 * k;
            const auto * p_m = cic.acc.at( i_m );
            for( int n = 0; n < n_int; ++ n )
                q_c[ n ] = p_m[ n ];

            for( int l = 0; l < m_dim; ++ l )
            {
                utils::muldim( di, l, geo.n_dim );
                bool skip( false );
                for( int a = 0 ; a <  geo.n_dim; ++ a )
                {
                    i_b[ a ] = di[ a ] + i_b0[ a ];
                    if( i_b[ a ] <  -b_send.n_gh  [ a ] ||
                        i_b[ a ] >= -b_send.n_gh  [ a ] +
                                     b_send.n_cell[ a ] )
                        skip = true;
                }
                if( skip )
                    continue;
                auto * q = b_send.at( i_b );
                for( int n = 0; n < n_int; ++ n )
                    q[ n ] = q_c[ n ];
            }
        }
        return;
    };

    template< class bdt_T > __device__ void from_buf
    ( const bdt_T & b_d , const  int & step )  const
    {
        idx_t i_m, i_b;
        if( ! get_self(  ).dispatch( i_b, n_ceff_r ) )
            return;
            
        const auto & n_int = b_d.cic.acc.n_int;

        for( int a = 0 ; a  < i_m.size(  ); ++ a )
            i_m[ a ] =   i_b[ a ] + offset_r [ a ];

        for( int k = 0; k < order; ++ k )
        {
            i_m[ axis[ 0 ] ]  = k + offset_r[ axis[ 0 ] ];
            i_b[ axis[ 0 ] ]  = k ;
            const auto * p_buff = b_recv     .at( i_b );
            auto       * p_mesh = b_d.cic.acc.at( i_m );
            for( int n = 0; n < n_int; ++ n )
                 p_mesh[ n ]  = p_buff  [ n ];
        }
        return;
    };
};                 // struct    comm_t
};                // namespace particle::cic::boundary::acc
