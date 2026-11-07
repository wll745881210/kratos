#pragma once

#include "../../../src/utilities/functions.h"
#include "weight.h"

namespace extension::immersed
{
////////////////////////////////////////////////////////////
// Small dense matrix on the device

template< class f_T = float_t >
struct mat_t
{
    int    m;
    f_T    dat[ 64 ];
    __host__ __device__ mat_t( const int & n ) { this->m = n; };
    __host__ __device__ f_T & operator(  )
    ( const int & i, const int & j ) const
    {
        return const_cast< f_T * >( dat )[ size_t( i ) * m
                                         + size_t( j ) ];
    };
};

////////////////////////////////////////////////////////////
// LU decomposition and back substitution ( Numerical
// Recipes, in-kernel, size <= 8 )

template< class mat_T, class vi_T, class vf_T > __device__
void lu_decomp( mat_T & a, vi_T & indx, vf_T & vv, vf_T & t )
{
    const int n = a.m;
    float_t big;
    for( int i = 0; i < n; ++ i )
    {
        big     = 0;
        for( int j = 0; j < n; ++ j )
            big = __max1( big, __abs1( a( i, j ) ) );
        if( big == 0 )
            big = utils::__epsilon_fl;
        vv[ i ] = 1 / big;
    }
    for( int j = 0; j < n; ++ j )
    {
        for( int i = 0; i < j; ++ i )
        {
            auto sum = a( i, j );
            for( int k = 0; k < i; ++ k )
                sum -= a( i, k ) * a( k, j );
            a( i, j ) = sum;
        }
        big = 0;
        int imax = j;
        for( int i = j; i < n; ++ i )
        {
            auto sum = a( i, j );
            for( int k = 0; k < j; ++ k )
                sum -= a( i, k ) * a( k, j );
            a( i, j ) = sum;
            const auto dum = vv[ i ] * __abs1( sum );
            if( dum >= big )
            {
                big  = dum;
                imax = i;
            }
        }
        if( j != imax )
        {
            for( int k = 0; k < n; ++ k )
            {
                const auto dum = a( imax, k );
                a( imax, k ) = a( j, k );
                a( j, k )    = dum;
            }
            vv[ imax ] = vv[ j ];
        }
        indx[ j ] = imax;
        if( a( j, j ) == 0 )
            a( j, j ) = utils::__epsilon_fl;
        if( j != n - 1 )
        {
            const auto dum = 1 / a( j, j );
            for( int i = j + 1; i < n; ++ i )
                a( i, j ) *= dum;
        }
    }
    return;
};

template< class mat_T, class vi_T, class vf_T > __device__
void backst( vf_T & b, const mat_T & a, const vi_T & indx )
{
    const int n = a.m;
    int ii = -1;
    for( int i = 0; i < n; ++ i )
    {
        const int  ip = indx[ i ];
        auto      sum = b[ ip ];
        b[ ip ] = b[ i ];
        if( ii >= 0 )
            for( int j = ii; j <= i - 1; ++ j )
                sum -= a( i, j ) * b[ j ];
        else if( sum != 0 )
            ii = i;
        b[ i ] = sum;
    }
    for( int i = n - 1; i >= 0; -- i )
    {
        auto sum = b[ i ];
        for( int j = i + 1; j < n; ++ j )
            sum -= a( i, j ) * b[ j ];
        b[ i ] = sum / a( i, i );
    }
    return;
};

//////////////////////////////////////////////////////////
// Truncated-SVD solve of a small square system ( in-kernel,
// size <= 8 ).
// One-sided Jacobi rotations orthogonalise the columns of M
// ( the caller passes M = V^T so that this solves V^T w = b ),
// accumulating the rotations in J:  M J = A  with orthogonal
// columns, sigma_i = |A(:,i)|, and hence  M = A J^T.  Then
//     w = M^{-1} b = J diag( 1/sigma_i^2 ) A^T b,
// with every singular value below rcond*sigma_max dropped.
// For a rank-deficient M ( e.g. immersed constraint rows that
// share one normal and span <= 5 dimensions ) this is the
// minimum-norm least-squares solution of the retained
// subspace: the weights stay bounded instead of blowing up to
// ~1/epsilon as an LU solve would.

template< class mat_T, class vf_T > __device__
void svd_solve( mat_T & M, mat_T & J, const vf_T & b, vf_T & w,
                const int & n, const float_t & rcond )
{
    const float_t eps = utils::__epsilon_fl;
    for( int sweep = 0; sweep < 30; ++ sweep )
    {
        bool rot( false );
        for( int p = 0; p < n - 1; ++ p )
            for( int q = p + 1; q < n; ++ q )
            {
                float_t alp( 0 ), bet( 0 ), gam( 0 );
                for( int k = 0; k < n; ++ k )
                {
                    alp += M( k, p ) * M( k, p );
                    bet += M( k, q ) * M( k, q );
                    gam += M( k, p ) * M( k, q );
                }
                if( __abs1( gam ) <= eps * __sqrt1( alp * bet ) )
                    continue;
                const float_t zet = ( bet - alp ) / ( 2 * gam );
                const float_t t
                    = ( zet >= 0 ? 1 : -1 )
                    / ( __abs1( zet ) + __sqrt1( 1 + zet * zet ) );
                const float_t c = 1 / __sqrt1( 1 + t * t );
                const float_t s = c * t;
                for( int k = 0; k < n; ++ k )
                {
                    const float_t mp = M( k, p ), mq = M( k, q );
                    M( k, p ) = c * mp - s * mq;
                    M( k, q ) = s * mp + c * mq;
                    const float_t jp = J( k, p ), jq = J( k, q );
                    J( k, p ) = c * jp - s * jq;
                    J( k, q ) = s * jp + c * jq;
                }
                rot = true;
            }
        if( ! rot )
            break;
    }
    float_t sig[ 8 ], smax( 0 );
    for( int i = 0; i < n; ++ i )
    {
        float_t s2( 0 );
        for( int k = 0; k < n; ++ k )
            s2 += M( k, i ) * M( k, i );
        sig[ i ] = __sqrt1( s2 );
        smax = __max1( smax, sig[ i ] );
    }
    for( int m = 0; m < n; ++ m )
        w[ m ] = 0;
    if( smax <= 0 )
        return;
    for( int i = 0; i < n; ++ i )
    {
        if( sig[ i ] <= rcond * smax )
            continue;
        float_t c( 0 );
        for( int k = 0; k < n; ++ k )
            c += M( k, i ) * b[ k ];
        const float_t e = c / ( sig[ i ] * sig[ i ] );
        for( int m = 0; m < n; ++ m )
            w[ m ] += J( m, i ) * e;
    }
    return;
};

};                              // namespace extension::immersed
