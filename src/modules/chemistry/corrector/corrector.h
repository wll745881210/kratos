#pragma once

#include "../../../mesh/mesh.h"
#include "../../../utilities/mapping/reduction.h"
#include "corrector_decomp.h"
#include "stoichiometry.h"

namespace chemistry
{
class base_t;
};

namespace chemistry::solver::corrector
{
////////////////////////////////////////////////////////////
// Correction of the4

struct base_t
{
    ////////// Data and modules //////////
    sto_t                  stoi; // Stoichiometric
    int                    n_sp;
    int                    n_el;
    int                i_elec_e; // For charge    (as elem)
    int                i_elec_s; // For electrons (as sps.)
    decomp_ls_t          decomp;
    type::float_t         y_min; // Floor for Y values

    ////////// Host-side functions //////////
    __host__ virtual void  init
    ( const  input & args, ::chemistry::base_t & mod );
    __host__ virtual  void  set_decomp
    ( ::chemistry::base_t &, const std::map< int, int > &,
      const mat_int_t & );

    ////////// Device-side functions //////////
    template < class   f_T, class dat_T ,
               class bdt_T, class eos_T > __device__ void
    fix( dat_T &, const bdt_T &, const eos_T &,
         const bool & is_final ) const;
};

////////////////////////////////////////////////////////////
// Fix the negative abundances

template< class f_T, class dat_T, class bdt_T, class eos_T >
__device__ void base_t::fix
( dat_T &  d , const bdt_T & bdt, const eos_T & eos,
  const bool & is_final ) const
{
    __syncthreads(  );
    const auto & i_th( threadIdx.x );        
    if( ! decomp )
        return;
    f_T * d_y  = ( f_T * )( d.x[ 2 ] );
    f_T * d_ye = ( f_T * )( d.x[ 4 ] );
    f_T * y0   = ( f_T * )( d.y      );
    if( is_final )
    {   // "final fix" deserves a full-precision abund. fix
        y0 = ( f_T * )( d.y_2 );
        for( int i = i_th; i < n_sp; i += blockDim.x )
            y0[ i ] = d.y[ i ];
        __syncthreads(  );
    }
    if( stoi.dat != nullptr )
        stoi( d );
    else
    {
        for( int i = i_th; i < n_sp; i += blockDim.x )
        {
            if( y0[ i ] < y_min || __isnan( y0[ i ] )
                                || __isinf( y0[ i ] ) )
                y0[ i ] = y_min;
            d_y [ i ] = y0[ i ] - f_T( d.y_fid[ i ] );
            d_ye[ i ] = 0;
        }
        __syncthreads(  );
        for( int i = i_th; i < n_sp; i += blockDim.x )
        {
            const auto & dec_s = decomp.s( i );
            for( int j = 0;  j < dec_s.size; ++  j )
                atomicAdd( & d_ye[ dec_s.idx [ j ] ] ,
                           - d_y[ i ]  * dec_s.num [ j ] );
        }
        __syncthreads(  );
    }
    if( i_th == 0 )
    {
        for( int i = 0; i < n_el; ++ i )
        {
            if( i == i_elec_e )
                continue;
            const auto & dec_e = decomp.e( i );        
            y0[ dec_e.i_map ] += d_ye[ dec_e.i_map ];
            auto dy  = y0 [ dec_e.i_map ];
            if(  dy >= 0 || dec_e.size == 0 )
                continue;
        
            int  s_max( -1 ), j_max( -1 );
            decltype ( d.h )  n_max(  0 );
            for( int j = 0; j  < dec_e.size; ++ j )
                if( y0[ dec_e.idx[ j ] ] > n_max )
                {
                    j_max =              j;
                    s_max = dec_e.idx[ j ];
                    n_max = y0   [ s_max ];
                }
            dy /= dec_e.num[ j_max ];
            y0[ s_max ] += dy;
            const auto & dec_s ( decomp.s( s_max ) );
            for( int j = 0;  j < dec_s.size; ++  j )
                y0[ dec_s.idx[ j ] ] -= dy * dec_s.num[ j ];
            y0[ dec_e.i_map ] = 0;
        }
        if( i_elec_s >= 0 )
        {
            const auto & dec_e = decomp.e( i_elec_e );
            y0[ i_elec_s ] = 0;
            for( int j = 0; j < dec_e.size ; ++ j )
                y0[ i_elec_s ] += dec_e.num[ j ]
                                * y0[ dec_e.idx[ j ] ];
        }
    }
    __syncthreads(  );
    return;
}

};                 // namespace chemistry::solver::corrector
