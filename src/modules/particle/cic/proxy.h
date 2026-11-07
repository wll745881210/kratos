#pragma once

#include "../particle.h"

namespace particle::cic
{
////////////////////////////////////////////////////////////
// Types for block access

template < class bdt_t = block_data_t >
struct proxy_t : mesh::block::proxy_t< bdt_t >
{
    struct nb_t
    {
        bool   shift[ 3 ];        
        int            id;
        int         d_lvl;
        int        i_rank;
    };
    nb_t        neighbors[ 4 ][ 4 ][ 4 ];
    mesh::  geo_t     geo;

    __host__ virtual void setup
    ( const mesh::block::dual_t   &      d ,
      const std ::map< int, int > & m_rank ,
      particle  ::base_t          &    mod )    
    {
        d.d(  ).geo.cp_shlw( geo );
        for( int i = 0; i < 4; ++ i )
            for( int j = 0; j < 4; ++ j )
                for( int k = 0; k < 4; ++ k )
                {
                    auto & n = neighbors[ i ][ j ][ k ];
                    n.    id = -1;
                    n. d_lvl =  0;
                    n.i_rank = -1;
                }
        
        const auto &  b ( d.b_w.get(  ) );
        for( const auto & [ nb, reg ] : b.neighbors )
        {
            const auto & g = nb.guest_info;
            auto & n = neighbors[ nb[ 0 ] + 1 ]
                 [ nb[ 1 ] + 1 ][ nb[ 2 ] + 1 ];
            if( g.rank  != b.rank )   // Skip same-rank
                n.i_rank = m_rank.at( g .rank );
            n.   id = g.  id_l;
            n.d_lvl = nb.d_lvl;
            const auto & nb_r ( nb.reverse(  ) );
            for( int a = 0; a < 3; ++ a )
                 n.shift[ a ] = nb_r[ a ];
        }
        return;
    };
    
    template< class par_T, class f_T > __device__
    bool shift_blk( par_T & par, f_T & dt_fac ) const
    {
        dt_fac = 1;
        par.dest.todo = particle::to_keep;
        const auto & n_ceff( geo.n_ceff );
        auto  & i ( par.i );
        type::idx_t      ib;
        bool inside( true );
        for( int a = 0; a < 3; ++ a )
        {
            ib[ a ] = ( i [ a ] >= n_ceff[ a ] ? 2 :
                      ( i [ a ] <  0 ?  -1 : 0 ) ) ;
            inside &= ( ib[ a ] == 0 )   ;
        }
        if( inside )
            return false;

        nb_t * p_nb = ( nb_t * )( & neighbors[ ib[ 0 ] + 1 ]
                          [ ib[ 1 ] + 1 ][ ib[ 2 ] + 1 ] ) ;
        if( p_nb ->d_lvl > 0 )
        {
            for( int a = 0; a < 3; ++ a )
            {
                ib[ a ] =   i [ a ] / ( n_ceff [ a ] / 2 );
                ib[ a ] = ( ib[ a ] >= 2 ?  2 :
                          ( i [ a ] <  0 ? -1 : ib[ a ] ) );
            }
            p_nb = ( nb_t * )( & neighbors[ ib[ 0 ] + 1 ]
                          [ ib [ 1 ] + 1 ][ ib[ 2 ] + 1 ] );
        }
        const auto &  nb( * p_nb );

        dt_fac = ( nb.d_lvl > 0 ?
                   0.5f : nb.d_lvl < 0 ? 2 : 1 );
        par.ib_l = nb.id;  // All-purpose!
        if( nb .i_rank >= 0 )
            par.dest.i_rank = nb.i_rank;
        else if( nb.id <  0 )
        {
            par.dest.todo = particle::to_rm;
            return false;
        }
        if( nb.d_lvl >= 0 )
            for( int a = 0; a < 3; ++ a )
            {
                i[ a ] -= ( ib  [ a ] < 0 ? -2 : ib[ a ] )
                       *  n_ceff[ a ] / 2 ;
                if( ib[ a ] >= 0 && ib[ a ] < 2 )
                    i [ a ] *= ( nb.d_lvl > 0 ? 2 : 1 );
            }   // Not exact, but next cycle shall fix it
        else
            for( int a = 0; a < 3; ++ a )
                i[ a ] = ( i[ a ] + nb.shift[ a ] *
                           n_ceff [ a ] ) / 2;
        return  par.dest.todo == to_keep;
    };  
};                       // struct proxy_t
};                       // namespace particle::cic
