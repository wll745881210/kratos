#pragma once

#include "../../../utilities/mapping/reduction.h"
#include "../types/particle_base.h"

namespace particle::cic
{
////////////////////////////////////////////////////////////
// Types

using type::  idx_t;
using type::float_t;
using type::coord_t;

////////////////////////////////////////////////////////////
// Basic photon type in Cartesian coordinates


template< class derived_T = crtp    ::   dummy_t ,
          class    base_T = particle::par_base_t >
struct particle_t : public base_T
{
    ////////// Types //////////
    __crtp_def_self__( particle_t, derived_T, base_T );
    using geo_loc_t  = particle::geo_loc_t;
    using base_T::     i;
    using base_T::     x;
    using base_T::   dir;
    using base_T::  ib_l;

    ////////// Functions //////////
    __host__ __device__ __forceinline__
    void load( const particle_t & tgt )
    {
        ( * this ) = tgt;
        return;
    };
    __host__ __device__ __forceinline__
    void save( particle_t & tgt ) const
    {
        tgt = ( * this );
        return;
    };
    __host__ __device__ __forceinline__
    void move( particle_t & tgt ) const
    {
        tgt = ( * this );
        return;
    };

    __device__ __host__ __forceinline__ auto & v(  )
    {
        return dir;  // Using direciton for velocity
    };
    __device__ __host__ __forceinline__
    const auto & v(  ) const
    {
        return dir;  // Using direciton for velocity
    };

    __device__ __forceinline__ float_t weight
    ( idx_t &  di , const  int & i_dim, const int & n_dim,
      const coord_t & x_c, const coord_t & dx ) const
    {
        float_t res( 1 );
        for( int a = 0; a < n_dim ; ++ a )
        {
            di[ a ]  = ( i_dim >> a ) & 1 ;
            const auto ddx = __abs1( x[ a ] - x_c[ a ] );
            res     *= ( di[ a ] ? ddx : dx[ a ] - ddx );
            di[ a ] *= ( x [ a ] > x_c [ a ] ? 1 :  -1 );
        }
        return res;
    };

    template< class acc_T, class prx_T, class idx_T >
    __device__ __forceinline__ void load_acc_raw
    ( acc_T & ac, const prx_T & prx, const idx_T & i ) const
    {
        const auto * acc_src = prx.cic.acc.at( i );
        for( int a = 0; a < prx.geo.n_dim; ++ a )
            ac[ a ] = acc_src[ a ];
        return;
    };

    template< class prx_T > __device__
    void get_acc( coord_t & acc, const prx_T & prx )
    {
        coord_t   x_c, dx, a_src;
        auto  i = this->i ;
        const auto & geo( prx.geo );
        for( int a = 0 ; a < geo.n_dim; ++ a )
        {
            dx [ a ] = geo.dx_f( a, i );
            acc[ a ] = 0;
        }
        geo.x_cc( x_c, i );
        auto dv_i = 1 / geo.volume( i );
        idx_t di;
        const auto m_dim( 1 << geo.n_dim );
        decltype( dv_i ) norm( 0 );
        bool skip;
        for( int l = 0; l <  m_dim; ++ l )
        {
            const auto w = get_self(  ).weight
                ( di, l, geo.n_dim, x_c, dx ) * dv_i;
            skip  = false;
            for( int a = 0 ; a < geo.n_dim; ++ a )
            {
                i[ a ] += di[ a ];
                if( i[ a ] <  0 )
                    skip |= geo.phybnd_flag[ a ][ 0 ];
                else if( i[ a ] >= geo.n_ceff[  a ] )
                    skip |= geo.phybnd_flag[ a ][ 1 ];
            }
            if( skip )
                continue;
            norm += w;
            get_self(  ).load_acc_raw( a_src, prx, i );
            for( int a = 0 ; a < geo.n_dim; ++ a )
            {
                acc[ a ] += w * a_src[ a ];
                i  [ a ] -= di       [ a ];
            }
        }
        for( int a = 0 ; a < geo.n_dim; ++ a )
            acc[ a ] /= norm;
        return;
    };
    template< class prx_T, class itg_T > __device__ void
    deposite( const prx_T & prx, const itg_T itg ) const
    {
        coord_t x_c, dx;
        const auto & geo( prx.geo );
        geo.x_cc( x_c, i );
        for( int a = 0 ; a < geo.n_dim; ++ a )
            dx [ a ] = geo.dx_f( a, i );
        const auto dv      = geo.volume( i );
        const auto cpl_dv2 = itg.couple(  ) / ( dv * dv );

        idx_t j;
        const auto m_dim( 1 << geo.n_dim );
        for( int l = 0; l <  m_dim; ++ l )
        {
            const auto w = cpl_dv2 * base_T::proper *
            get_self(  ).weight( j, l, geo.n_dim, x_c, dx );
            for( int a = 0 ; a < geo.n_dim; ++ a )
                j[ a ] += i[ a ];
            atomicAdd( prx.cic.den.at( j ), w );
        }
        return;
    };

    __device__ __host__
    void find_cell_local( const geo_t & geo )
    {
        for( int a = 0; a < 3; ++ a )
        {
            if( geo.is_uniform[ a ] )
            {
                const auto ii = ( x[ a ] - geo.xf0[ a ] )
                                         / geo.dx0[ a ] ;
                i[ a ] = int( ii ) - ( ii < 0 );
                continue;
            }
            while( i[ a ] >= -geo.n_gh &&
                   i[ a ] <   geo.n_gh + geo.n_ceff[ a ] )
            {
                const auto x_l = geo.x_fc( a, i[ a ]     );
                const auto x_r = geo.x_fc( a, i[ a ] + 1 );
                if( x_l <= x [ a ] &&  x_r >= x[ a ] )
                    break;
                i [ a ] += ( x[ a ] < x_l ? -1 : 1 );
            }
        }
        return;
    };

    template< class map_T > __device__ __host__
    bool find_cell( const map_T & bmap )
    {
        const auto  & geo( bmap[ ib_l ].geo );
        get_self_c(  ).find_cell_local( geo );
        for( int a = 0; a < 3; ++ a )
            if( i[ a ] < 0 || i[ a ] >= geo.n_ceff[ a ] )
                return false;
        return true;
    };

    template   < class map_T, class itg_T > __device__
    void evolve( const map_T & bmap, const itg_T & itg )
    {
        auto &    self( get_self_c(    ) );
        const  auto & prx(  bmap[ ib_l ] );
        type  ::coord_t  acc,  dx , ddx;
        const auto & dt( bmap. dt );
        self.get_acc   ( acc, prx );
        auto & v = self.v(  );
        auto dt_eff( dt );
        for( int a = 0; a < prx.geo.n_dim; ++ a )
        {
            auto v_new = v[ a ] + acc[ a ] * dt; // Kick
            if( v_new == 0 )
                continue;
            dx[ a ]   = prx.geo.dx_f( a, i ) ;
            const auto & dt_s
                   = __abs1( dx[ a ] / v_new );
            dt_eff = __min1( dt_s,    dt_eff );
        }
        for( int a = 0; a < prx.geo.n_dim; ++ a )
        {
            v[ a ] += acc[ a ] * dt_eff;
            x[ a ] += v  [ a ] * dt_eff;
        }
        type::float_t dt_tgt ( FLT_MAX ), dt_fac( 1 );
        self.find_cell( bmap ) ;
        self.regulate ( bmap ) ;
        if( prx .shift_blk( self, dt_fac ) )
            self.find_cell( bmap );
        for( int a = 0; a < prx.geo.n_dim; ++ a )
        {
            auto dt_s( v [ a ] ==  0 ? dt_tgt :
                       dx[ a ] / __abs1( v[ a ] ) );
            dt_tgt = utils::min( dt_s, dt_tgt );
            if( acc[ a ] !=  0 )
            {
                const auto  v02  =  v[ a ] * v[ a ];
                auto  det = v02  + 2 * dx[ a ] * acc[ a ];
                if( det < 0 ) // det < 0: dx_final * v < 0
                    continue;
                det = __sqrt1( det );
                const auto dt_ap = __abs1
                    ( ( -v[ a ]  + det ) / acc[ a ] );
                const auto dt_am = __abs1
                    ( ( -v[ a ]  - det ) / acc[ a ] );
                const auto dt_a  = __min1( dt_ap , dt_am );
                dt_s   = ( dt_a  >  0 ? dt_a : dt_s );
                dt_tgt = utils::min ( dt_s,  dt_tgt );
            }
        }
        dt_tgt *= dt_fac;
        if( dt_tgt < bmap.dt_lim )
            utils::atomic_min ( bmap.p_dt, dt_tgt );
        return;
    };

    template < class map_T, class itg_T > __device__
    int proc( const map_T & bmap, const itg_T & itg )
    {
        auto & self( get_self_c(  ) );
        self.evolve( bmap, itg )  ;
        if( self.dest.todo ==  to_keep )
            self.deposite( bmap [ ib_l ], itg );
        return self.dest.i_rank ;
    };
};
};                // namespace particle::cic
