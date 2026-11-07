#pragma once

#include "../../../utilities/mapping/reduction.h"

namespace multigrid::operation
{
////////////////////////////////////////////////////////////
// Dispatcher

template < class          derived_T >
template < class idx_T, class bdt_T > __forceinline__
__device__ bool operation_t  < derived_T >:: dispatch_lvlchg
( idx_T & idx, const bdt_T & b_c                   ) const
{
    return get_self(  ).dispatch
         ( idx, b_c.geo.n_ceff, utils::th_id(  ) );
}

template < class          derived_T >
template < class idx_T, class bdt_T > __forceinline__
__device__ bool operation_t  < derived_T >:: dispatch_smooth
( idx_T & i, const bdt_T & bdt, const int & stage ) const
{
    const auto res =  get_self(  ).dispatch
        ( i, bdt.geo.n_ceff, 2 * utils::th_id(  ) );
    i[ 0 ] += ( stage + i[ 1 ] + i[ 2 ] ) & 1 ;
    return res;
}

template < class derived_T >
template < class     idx_T > __device__ __forceinline__
bool    operation_t < derived_T >::dispatch
( idx_T & i, const idx_T & n_c, const int & i_t ) const
{
    utils::divmod( i  [ 2 ] , i   [ 1 ] , i_t,
                   n_c[ 1 ] * n_c [ 0 ] ) ;
    const int i1 = i[ 1 ];
    utils::divmod( i[ 1 ], i[ 0 ], i1, n_c[ 0 ] );
    return ( i[ 0 ] < n_c[ 0 ] &&  i[ 1 ] < n_c[ 1 ] &&
             i[ 2 ] < n_c[ 2 ] );
}

template < class derived_T >
template < class     idx_T > __device__ __forceinline__
bool    operation_t < derived_T >::muldim
( idx_T & j, const int & i_t, const int & n_dim ) const
{
    bool rb( j [ 0 ] = i_t & 1 ) ;
    for( int a = 1 ; a < n_dim ; ++ a )
         rb ^= ( j [ a ] = ( i_t >> a ) & 1 );
    return  rb ;
}

template< class derived_T >
template< class   f_T, class dat_T ,
          class bdt_T, class idx_T >
__device__ __forceinline__
void  operation_t< derived_T >::weight
( f_T & w_c, f_T w [  ][ 2 ], const dat_T & dat,
  const bdt_T  & bdt,         const idx_T & i  ) const
{
    w_c = 0 ;
    const auto & n_ceff( bdt.geo.n_ceff );
    for( int a = 0 ; a < bdt.geo.n_dim; ++ a )
    {
        const auto l_bnd = bdt.geo.phybnd_flag[ a ][ 0 ]
            * ( i[ a ]  == 0 );
        const auto r_bnd = bdt.geo.phybnd_flag[ a ][ 1 ]
            * ( i[ a ]  == n_ceff[ a ] - 1 );

        const auto dx_l = dat. dx[ a ][ 0 ] / ( 1 + l_bnd );
        const auto dx_r = dat. dx[ a ][ 1 ] / ( 1 + r_bnd );
        w_c        -= 2 / ( dx_l *   dx_r ) ;
        w[ a ][ 0 ] = 2 / ( dx_l * ( dx_l + dx_r ) );
        w[ a ][ 1 ] = 2 / ( dx_r * ( dx_l + dx_r ) );

        if( phy_bnd != fixed )
        {
            if( l_bnd )
            {
                w_c += w[ a ][ 0 ];
                w[ a ][ 0 ] =  0  ;
            }
            if( r_bnd )
            {
                w_c += w[ a ][ 1 ];
                w[ a ][ 1 ] =  0  ;
            }
        }
    }
    return;
}

////////////////////////////////////////////////////////////
// Device-side operations

template  < class derived_T >
template  < class     bdt_T > __device__
void operation_t< derived_T >::coarsen
( const bdt_T & b_c, const bdt_T & b_f ) const
{
    using f_t = typename decltype( b_c.mgd )::f_t;
    type::idx_t i;
    if( ! get_self(  ).dispatch_lvlchg( i, b_c ) )
        return;
    const auto & geo( b_f. geo             );
    const auto m_dim( 1 << geo.      n_dim );
    const auto dv_c ( b_c. geo.volume( i ) );
    type::idx_t j, dj;

    __dyn_shared__( f_t, p_sh );
    auto  * f_c = p_sh + n_sh_coarse( b_c ) * threadIdx.x;
    auto  * s_c = f_c  + n_fld ;
    data_t< f_t > dat  ( n_fld + s_c, b_c, n_fld );

    for ( int n = 0; n < n_fld; ++ n )
        s_c[ n ] = f_c[ n ] = 0;
    for ( int l = 0; l < m_dim; ++ l )
    {
        muldim( dj, l, geo.n_dim );
        for( int a = 0; a < geo. n_dim; ++ a )
             j [ a ] =  2 * i[ a ] + dj  [ a ];
        const auto * f_f = b_f.mgd.fld.at( j );
        const auto v_rat = geo.volume( j ) / dv_c;        
        
        dat.load ( b_f, j ) ;
        f_t w[ 3 ][ 2 ], w_c;
        get_self(  ).weight( w_c,  w, dat, b_f, j );
        for ( int n = 0; n < n_fld; ++ n )
        {
            auto r_t = dat.src[ n ] - w_c * f_f[ n ];
            for( int a = 0; a < geo.n_dim ; ++ a )
                for( int k = 0; k < 2; ++ k )
                     r_t -= w [ a ][ k ]  *
                            dat.fld[ a ][ k ][ n ];
            s_c[ n ] += r_t      * v_rat;
            f_c[ n ] += f_f[ n ] * v_rat;
        }
    }
    auto * s_tgt = b_c.mgd.src_dat.at( i );
    auto * f_tgt = b_c.mgd.    fld.at( i );
    auto * f_ori = f_tgt  +  n_fld;
    for( int n = 0; n < n_fld; ++ n )
    {
        s_tgt[ n ] = s_c[ n ];
        f_tgt[ n ] = f_c[ n ];
        f_ori[ n ] = f_c[ n ];
    }
    return;
}

template  < class derived_T >
template  < class     bdt_T > __device__
void operation_t< derived_T > ::refine
( const bdt_T & b_c,  const bdt_T & b_f ) const
{
    using f_t = typename decltype( b_c.mgd )::f_t;
    type::idx_t i ;
    if( ! get_self(  ).dispatch_lvlchg( i, b_c ) )
        return;
    const auto & geo( b_f.  geo );
    __dyn_shared__  ( f_t, p_sh );
    f_t * df[ 3 ];
    f_t * f_c = p_sh + 4 * n_fld * threadIdx.x;

    const auto m_dim ( 1 << geo.n_dim );
    type::idx_t j, dj, dk;

    for( int a = 0; a < geo.n_dim; ++ a )
    {
        df[ a ] = f_c;
        f_c  += n_fld;
        ++ i[ a ];
        const auto * qrn = b_c.mgd.fld.at( i );
        const auto * qro = qrn + n_fld;
        i[ a ] -= 2;
        const auto * qln = b_c.mgd.fld.at( i );
        const auto * qlo = qln + n_fld;        
        for( int n = 0; n < n_fld; ++ n )
            df[ a ][ n ] = ( ( qrn[ n ] - qro[ n ] ) -
                             ( qln[ n ] - qlo[ n ] ) ) / 8;
        ++ i[ a ];
    }
    const auto * qcn = b_c.mgd.fld.at( i );
    const auto * qco = qcn + n_fld;
    for( int n = 0; n < n_fld; ++ n )
        f_c[ n ] = ( qcn[ n ] - qco[ n ] );

    const bool sub_fld( flag & opt_flag_t::subtract_fld );
    for( int l = 0; l < m_dim; ++ l )
    {
        if( muldim( dj, l, geo.n_dim ) == rb_t::red )
            continue;
        for( int a = 0; a < geo.n_dim; ++ a )
            j[ a ] = 2  * i[ a ] + dj[ a ];
        auto * f_f = b_f. mgd.fld. at( j );
        for( int n = 0; n < n_fld; ++  n )
        {
            f_t d( f_c[ n ] );
            for( int a = 0; a < geo.n_dim; ++ a )
                d += ( 2 * dj[ a ] - 1 ) * df[ a ][ n ];
            if( sub_fld )
                d -= src_ave_c[ n_fld + n ];
            f_f[ n ] += d;
        }
    }
    return;
};

template< class derived_T >
template< class     bdt_T >
__device__ void operation_t< derived_T >::smooth
( const bdt_T & bdt, const rb_t  & stg )   const
{
    using f_t = typename decltype( bdt.mgd )::f_t;
    __dyn_shared__   ( f_t, p_sh );
    const int n_sh = n_sh_smooth( bdt );
    data_t< f_t > dat ( p_sh + threadIdx.x * n_sh,
                        bdt,  n_fld );
    auto * f_tmp( dat.fld[ 0 ][ 0 ] );
    const bool to_defect = ( stg == rb_t::black ) &
        ( flag & opt_flag_t::output_defect ) ;

    const auto  & geo = bdt.geo;
    auto  *  flag_eff
          = ( bool * )( p_sh + n_sh * blockDim.x );
    auto  &       eff ( flag_eff [ threadIdx.x ] );
    type::idx_t i ;
    eff  = get_self(  ).dispatch_smooth( i, bdt, stg );

    if( eff )
    {
        if( flag & opt_flag_t::subtract_src )
        {
            auto * s = bdt.mgd.src_dat.at( i );
            for( int n  = 0; n < n_fld; ++ n )
                s[ n ] -= src_ave_c      [ n ];
        }
        dat.load( bdt, i );        
        f_t  w[ 3 ][ 2 ],    w_c, f_0( 0 );
        get_self(  ).weight( w_c, w, dat, bdt, i );
        auto * f = bdt.mgd.fld.at( i );
        for( int n = 0; n < n_fld; ++ n )
        {
            f_t df( 0 );
            for( int a = 0 ; a < geo.n_dim; ++  a )
                for( int j = 0 ; j <   2;   ++  j )
                    df -= w[ a ] [ j ]
                        * dat.fld[ a ][ j ][ n ];
            if( to_defect )
                f_0 = f[ n ];
            const auto f_new = ( dat.src[ n ] + df ) / w_c;
            if( to_defect )
            {
                f_tmp  [ n ] = -__abs1( f_new + f_0 ) / 2;
                dat.src[ n ] = -__abs1( f_new - f_0 );
            }
            f[ n ] = f_new ;
        }
    }
    __syncthreads(  );
    if( to_defect && threadIdx.x == 0 )
        for( int n = 0; n < n_fld; ++ n )
        {
            if( ! eff )
                dat.src[ n  ] = 0;
            for( int j = 1; j < blockDim.x; ++ j )
            {
                if( ! flag_eff[ j ] )
                    continue;
                dat.src[ n ] = __min1
                ( dat.src[ n_sh * j + n ], dat.src[ n ] );
                f_tmp  [ n ] = __min1
                ( f_tmp  [ n_sh * j + n ], f_tmp  [ n ] );
            }
            utils::atomic_min
                (    defect_d + n, dat.src[ n ] );
            utils::atomic_min
                ( field_max_d + n,   f_tmp[ n ] );
        }
    return;
}

template< class derived_T >
template< class     bdt_T > __device__ void
operation_t< derived_T >::residual( const bdt_T & bdt )const
{
    type::idx_t i;
    if( ! get_self(  ).dispatch_lvlchg( i, bdt ) )
        return;
    using f_t = typename decltype( bdt.mgd )::f_t;
    __dyn_shared__( f_t, p_sh );
    data_t< f_t > dat( p_sh + n_sh_coarse( bdt ) *
                       threadIdx.x, bdt, n_fld ) ;
    dat.load   ( bdt, i, false );
    f_t  w[ 3 ][ 2 ],    w_c;
    get_self(  ).weight( w_c, w, dat, bdt, i );    
    const auto * f = bdt.mgd.fld    . at ( i );
    auto     * s_c = bdt.mgd.src_dat. at ( i );
    for( int n = 0; n < n_fld; ++ n )
    {
        auto ds = w_c * f[ n ];
        for( int a = 0; a < bdt.geo.n_dim; ++ a )
            for( int k = 0; k < 2; ++ k )
                ds += w[ a ][ k ] * dat.fld[ a ][ k ][ n ];
        s_c[ n ] += ds;
    }
    return;
}

template< class derived_T >
template< class     bdt_T > __device__ void
operation_t< derived_T >::calc_ave
( const bdt_T & bdt ) const
{
    type::idx_t i;
    i[ 2 ] =  blockIdx.x;
    i[ 1 ] = threadIdx.x;
    const auto & n_fld = this->n_fld;
    __dyn_shared__( type::float_t, p_sh );

    auto  n_sh =  ( 2 * n_fld + 1 );
    auto * src = p_sh + n_sh  * threadIdx.x;
    auto * fld = src  + n_fld ;
    for( int n = 0; n < n_sh  ; ++ n )
        src[ n ] =  0 ;
    for( int k = 0; k < bdt.geo.n_ceff[ 0 ]; ++ k )
    {
        i[ 0 ] = k;
        const auto  dv  = bdt.geo.volume    ( i );
        const auto * s  = bdt.mgd.src_dat.at( i );
        const auto * f  = bdt.mgd.fld    .at( i );
        for( int n = 0; n < n_fld; ++ n )
        {
            src[ n ] += s[ n ] * dv;
            fld[ n ] += f[ n ] * dv;
        }
        src[ n_sh - 1 ] += dv;
    }
    __syncthreads(  );
    if( threadIdx.x == 0 )
        for( int n = 0; n < n_sh; ++ n )
        {
            for( int j = 1; j < blockDim.x; ++ j )
                src[ n ] += src [ n_sh * j + n ]  ;
            atomicAdd(  src_ave_d +  n, src[ n ] );
        }        
    return;
}

template< class derived_T >
template< class     bdt_T > __device__   void
operation_t< derived_T >::defect ( const bdt_T & bdt ) const
{
    type::idx_t i;    
    i[ 0 ] = threadIdx.x;
    i[ 1 ] =  blockIdx.x;
    i[ 2 ] =  blockIdx.y;
    
    using f_t = typename decltype( bdt.mgd )::f_t;
    __dyn_shared__( f_t, p_sh );
    const auto n_sh  = n_sh_smooth ( bdt );
    data_t< f_t > dat( p_sh + n_sh * threadIdx.x,
                       bdt, n_fld );
    dat.load( bdt, i );
    f_t  w[ 3 ][ 2 ], w_c;
    get_self(  ).weight( w_c, w, dat, bdt, i );    
    const auto * f  = bdt.mgd.fld   . at ( i );
    auto       * s  = bdt.mgd.src_dat.at ( i );    
    for( int n = 0; n < n_fld; ++ n )
    {
        const auto s0 = __abs1( dat.src[ n ] );
        f_t   ddf     = - w_c   *     f[ n ]  ;
        for( int a = 0; a < bdt.geo.n_dim ; ++ a )
            for( int k = 0 ; k < 2; ++ k )
                ddf -= w[ a ][ k ] * dat.fld[ a ][ k ][ n ];
        s[ n ] = __abs1( dat.src[ n ] + ddf );
    }
    return;
}
};                         // namespace multigrid::operation
