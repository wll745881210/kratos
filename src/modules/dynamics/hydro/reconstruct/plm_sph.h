#pragma once

#include "plm.h"

namespace dynamics::hydro::reconstruct
{
////////////////////////////////////////////////////////////
// Reconstruction with 

template< class derived_T = crtp::dummy_t >
struct plm_sph_t : plm_t
                 < crtp::helper< plm_sph_t, derived_T > >
{
    ////////// Types //////////
    __crtp_def_self__( plm_sph_t, derived_T );
    using super_t
        = plm_t< crtp::helper< plm_sph_t, derived_T > >;
    using super_t::q_ll, super_t::q_l;
    using super_t::q_rr, super_t::q_r;
    using super_t::        is_uniform;

    ////////// Functions //////////
    template< class dat_T, class idx_T, class bdt_T >
    __device__ void  pre_proc
    ( dat_T & d, const int & i, const int & n,
      const idx_T & j, const bdt_T & bdt ) const        
    {
        const auto & geo( bdt.geo );
        if( n == idx::mom + 2 )
        {
            if( d.axis == 0 )
            {
                const auto s = __sin1
                          ( geo.x_cc( 1, j[ 1 ] )   );
                q_ll /= s * geo.x_cc( 0, j[ 0 ] - 2 );
                q_l  /= s * geo.x_cc( 0, j[ 0 ] - 1 );
                q_r  /= s * geo.x_cc( 0, j[ 0 ]     );
                q_rr /= s * geo.x_cc( 0, j[ 0 ] + 1 );
            }
            else if( d.axis == 1 )
            {
                const auto r = geo.x_cc( 0, j[ 0 ] );
                q_ll /= r * __abs1( __sin1
                     ( geo.x_cc( 1, j[ 1 ] - 2 ) ) );
                q_l  /= r * __abs1( __sin1
                     ( geo.x_cc( 1, j[ 1 ] - 1 ) ) );
                q_r  /= r * __abs1( __sin1
                     ( geo.x_cc( 1, j[ 1 ]     ) ) );
                q_rr /= r * __abs1( __sin1
                     ( geo.x_cc( 1, j[ 1 ] + 1 ) ) );
            }
        }   // Reconstruct omega ( except ax == 2 )
        return;
    };

    template< class dat_T, class idx_T, class bdt_T >
    __device__ void post_proc
    ( dat_T & d, const int & i, const int & n,
      const idx_T & j, const bdt_T & bdt ) const
    {
        if( n != idx::mom + 2 )
            return;
        const auto & geo( bdt.geo );
        type::float_t R_f( 1 ); // Default: ax == 2
        if( d.axis == 0 )
            R_f =         geo.x_fc( 0, j[ 0 ] )
                * __sin1( geo.x_cc( 1, j[ 1 ] ) );
        else if( d.axis == 1 )
            R_f =         geo.x_cc( 0, j[ 0 ] )
                * __sin1( geo.x_fc( 1, j[ 1 ] ) );
        d.ql.at( i )[ idx::mom + 2 ] *= R_f;
        d.qr.at( i )[ idx::mom + 2 ] *= R_f;
        return;
    };

    template< class dat_T, class bdt_T > __device__
    void general( dat_T     & d, const int   &   i ,
                  const int & n, const bdt_T & bdt ) const
    {
        type::idx_t j;
        d.dispatch( j, bdt );
        j[ d.axis ]   = ( i + d.blk_shift );                
        pre_proc        ( d, i, n, j, bdt );        
        super_t::general( d, i, n,    bdt );
        return post_proc( d, i, n, j, bdt );
    };
    
    template< class dat_T, class bdt_T > __device__
    void uniform( dat_T     & d, const int   &   i ,
                  const int & n, const bdt_T & bdt ) const
    {
        type::idx_t j;
        d.dispatch( j, bdt );
        j[ d.axis ]   = ( i + d.blk_shift );                
        pre_proc        ( d, i, n, j, bdt );        
        super_t::uniform( d, i, n,    bdt );
        return post_proc( d, i, n, j, bdt );        
    };

    template < class dat_T, class bdt_T > __device__
    void calc( dat_T     & d , const bdt_T & b_d,
               const int & i ) const
    {
        const  auto & self = get_self(  );
        is_uniform = d.is_uniform;
        if( ! is_uniform )
            self.load_geo( d.xf.at( i ), d.xv.at( i ) );

        for( int n = 0; n < d.q.n_int; ++ n )
        {
            q_ll = d.q( i - 2, n );
            q_l  = d.q( i - 1, n );
            q_r  = d.q( i    , n );
            q_rr = d.q( i + 1, n );
            if( is_uniform )
                get_self(  ).uniform( d, i, n, b_d );
            else
                get_self(  ).general( d, i, n, b_d );
        }
        return;
    };
};                              // class plm_sph_t
};                 // namespace dynamics::hydro::reconstruct
