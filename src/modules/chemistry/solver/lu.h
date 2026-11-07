#pragma once

#include "../types.h"

namespace chemistry::solver
{
////////////////////////////////////////////////////////////
// Multi-threading LU decomposition

template< class f_T >
struct lu_t
{
    ////////// Data //////////
    f_T    *   vv;
    f_T    *   tt;
    f_T    *  dat;
    int    * indx;
    int       dim;    

    ////////// Functions //////////
    __host__ __device__ __forceinline__
    static int size( const int & dim )
    {
        return ( sizeof( f_T )   * dim +
                 sizeof( int ) ) * dim;
    };
    __device__ __forceinline__
    void * setup( char * dat, const int & dim )
    {
        this->dim  =              dim ;
        this->dat  = ( f_T * )    dat ;
        this->indx = ( int * ) ( this->dat + dim * dim );
        return indx + dim;
    };
    __device__ __forceinline__
    void set_aux( void * vv, void * tt )
    {
        this->  vv = ( f_T    * )   vv;
        this->  tt = ( f_T    * )   tt;
        return;
    };

    __host__ __device__ __forceinline__
    f_T & operator(  ) ( const int & i, const int & j )
    {
#ifdef __CPU_DEBUG__
        if( i < 0 || i >= dim || j < 0 || j >= dim )
            throw std::runtime_error( "LU  matrix" );
#endif
        return dat[ j + i * dim ];
    };

    __device__ __forceinline__ int max_idx( f_T * x, int m )
    {
        int imax(    0   );
        f_T xmax( x[ 0 ] );
        for( int j = 1; j < m; ++ j )
            if( x[ j ] > xmax )
            {
                xmax = x[ j ];
                imax =      j;
            }
        return imax;
    };
    template< class g_T, class h_T >
    __device__ __forceinline__ void swap( g_T & a, h_T & b )
    {
        const auto c = a;
        a            = b;
        b            = c;
        return;
    };
    __device__ __forceinline__ void decompose(  )
    {
        auto & a ( *  this );

        const int   i_th( threadIdx.x );
        const int   n_th(  blockDim.x );
        const auto & dim(  this-> dim );
        
        __syncthreads(  );
        for( int i = i_th; i < dim; i += n_th )
        {
            indx [ i ] = i;
            f_T dum( 0 );
            for( int j = 0; j < dim; ++ j )
                dum = __max1( dum, __abs1( a( i, j ) ) );
            vv[ i ] = 1 / dum;
        }
        for( int j = 0; j < dim; ++ j )
        {
            for( int k = 0; k < j; ++ k )
            {
                __syncthreads(  );
                for( int i = i_th; i < dim; i += n_th )
                    if( k < i )
                        a ( i, j ) -= a( i, k ) * a( k, j );
            }
            __syncthreads(  );
            for( int i = i_th; i < dim; i += n_th )
                if( i >= j )
                    tt[ i ] = vv[ i ] * __abs1( a( i, j ) );
            __syncthreads(  );
            int im = max_idx( tt + j, dim - j ) + j;
            if( j !=  im )
            {
                for( int i = i_th; i < dim; i += n_th )
                    swap( a( im, i ), a( j, i ) );
                if( i_th == 0 ) // Swap only once!
                {
                    vv  [ im ] = vv[ j ];
                    indx[ j  ] =    im;
                }
            }
            __syncthreads(  );
            for( int i = i_th; i < dim; i += n_th )
                if( j < i )
                    a ( i, j ) /= a ( j, j );
        }
        return;
    };

    template < class vec_T >
    __device__ void back_sub( vec_T & b )
    {
        auto  & a     ( *      this );
        const int i_th( threadIdx.x );
        const int n_th(  blockDim.x );

        __syncthreads(  );

        if( i_th ==  0 )
            for( int k = 0; k < this->dim; ++ k )
            {
                const int & kp( indx[ k ] );
                if( kp != k )
                    swap( b[ k ], b[ kp ] );
            }
        for( int j = 0; j < this->dim; ++ j )
        {
            __syncthreads(  );
            for( int i = i_th; i < this->dim; i += n_th )
                 if( j < i )
                     b [ i ] -= a( i, j )  * b[ j ];
        }
        __syncthreads(  );
        for( int i = i_th; i < this->dim; i += n_th )
            b[ i ] /= a ( i, i );
        for( int j  = this->dim - 1; j > 0; -- j )
        {
            __syncthreads(  );
            for( int i = i_th; i < this->dim; i +=  n_th )
                 if( j > i )
                     b [ i ] -= a( i, j ) * b[ j ]
                             /  a( i, i );
        }
        __syncthreads(  );
        return;
    };
};

};                       // namespace chemistry::solver
