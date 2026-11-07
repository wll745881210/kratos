#pragma once

#include "../../../utilities/math/svd.h"
#include "../../../device/device.h"
#include "../types.h"
#include <numeric>

namespace chemistry
{
////////////////////////////////////////////////////////////
// Stoichiometric projection

using mat_int_t = utils::math::matrix_t< int >;

struct    sto_t
{
    ////////// Data //////////
    using  f_t = type::float_t;
    int      n ;
    f_t *  dat ;

    ////////// Host-side Functions //////////
    __host__ sto_t(  )  : dat( nullptr ) {  } ;

    __host__ virtual void init( device::base_t  &   f_dev ,
                                const mat_int_t & sp_elem );
    __host__ virtual void init( device::base_t  &   f_dev ,
                                const  sp_vec_t & species );

    ////////// Device-side Functions //////////
    __device__ __forceinline__ operator bool (  ) const
    {
        return dat != nullptr;
    };
    
    template < class f_T >
    __device__ void proj( f_T * dn, f_T * aux ) const
    {
        for( int i = 0; i < n; ++ i )
        {
            aux[ i ] = 0;
            for( int j = 0; j < n; ++ j )
                aux[ i ] += dat[ i * n + j ] * dn[ j ];
        }
        for( int i = 0; i < n; ++ i )            
            dn[ i ] = aux[ i ];
        return;
    };
    
    template < class dat_T >
    __device__  void operator (  ) ( dat_T & d ) const
    {
        const int n_th( blockDim.x );
        auto & aux( d.x[ 0 ] );
        for( int i = threadIdx.x; i < n; i += n_th )
            aux[ i ] = d.y[ i ] - d.y_old[ i ];
        __syncthreads(  );
        for( int i = threadIdx.x; i < n; i += n_th )
        {
            d.y[ i ] = d.y_old[ i ];
            for( int j = 0; j < n; ++ j )
                d.y[ i ] += dat[ i * n + j ] * aux[ j ];
        }
        return ;
    };
};

};                              // namespace chemistry
