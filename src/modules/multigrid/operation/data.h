#pragma once

#include "../../../utilities/functions.h"

namespace multigrid::operation
{
////////////////////////////////////////////////////////////
// Local data buffer for the operations

template < class f_T >
struct data_t
{
    ////////// Data //////////
    using f_t     = f_T;
    int           n_fld;
    f_T *           src;
    f_T             vol;
    f_T         xc[ 3 ];
    f_T    dx[ 3 ][ 2 ];    
    f_T * fld[ 3 ][ 2 ];

    ////////// Functions //////////
    template < class bdt_T > __device__
    data_t( f_T * p , const bdt_T & bdt, const int & n_fld )
    {
        this->n_fld = n_fld;
        for( int a = 0; a < bdt.geo.n_dim; ++ a )
            for( int j = 0; j < 2; ++ j )
            { 
                fld[ a ][ j ] = p;
                p += n_fld;
            }
        src   = p;
        return;
    };
    template < class bdt_T > __forceinline__ __device__
    void load( const bdt_T & bdt, type::idx_t & i,
               const bool  & load_src = true )
    {
        const auto & geo = bdt.geo;
        const auto & mgd = bdt.mgd;
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            xc[ a ] = geo.x_cc ( a, i[ a ] );
            --   i[ a ] ;
            for( int  j = 0; j < 2; ++ j )
            {
                const auto * p = mgd .fld.at( i );
                for( int n = 0; n < n_fld; ++ n )
                    fld[ a ][ j ] [ n ]  = p[ n ];
                dx[ a ][ j ] = __abs1
                    ( geo.x_cc( a , i[ a ] ) - xc[ a ] );
                i [ a ] += 2;
            }
            i[ a ] -= 3;
        }
        if( load_src )
            for( int  n = 0 ; n < n_fld;  ++  n )
                this->src [ n ] = mgd.src( i, n );
        return;
    };
};
};                         // namespace multigrid::operation
