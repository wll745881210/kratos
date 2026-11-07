#pragma once

#include "../../../utilities/functions.h"
#include "../reaction/reaction.h"
#include "../types.h"

namespace chemistry
{
class base_t;
};

namespace chemistry::nuclear::corrector
{
////////////////////////////////////////////////////////////
// Species abundance corrector for nuclear reactions

struct base_t
{
    ////////// Data and modules //////////
    int n_sp;                   // Total number of species 
    int *  a;                   // A of each species

    ////////// Functions //////////
    __host__ virtual void  init
    ( const  input & args, ::chemistry::base_t & mod );

    template< class   f_T, class dat_T ,
              class bdt_T, class eos_T > __device__ void
    fix( dat_T & d, const bdt_T &, const eos_T &,
         const bool & ) const
    {        
        __syncthreads(  );
        const auto & i_th( threadIdx.x );
        for( int i = i_th; i < n_sp; i +=  blockDim.x )
            d.y[ i ] = __max1( d.y[ i ], 0.f );
        
        __syncthreads(  );
        typename dat_T::f_t y ( 0 ), y_old( 0 );
        for( int i = 0; i < n_sp; ++ i )
        {
            y     += d.y    [ i ] * a[ i ];
            y_old += d.y_old[ i ] * a[ i ];
        }        
        __syncthreads(  );
        const auto  ratio( y_old / y );
        for( int i = i_th; i < n_sp; i +=  blockDim.x )
            d.y[ i ] *= ratio;
        __syncthreads(  );
        return;
    };
};

};             // namespace chemistry::nuclear::corrector
