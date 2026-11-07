#pragma once

#include "../../../../src/modules/particle/types/particle_pool.h"

namespace extension::radiation
{
////////////////////////////////////////////////////////////
//

template< class par_T, class int_T = int >
struct pool_t : particle::pool_t
              < par_T, int, pool_t< par_T, int_T > >
{
    ////////// Data //////////
    int n_hv;
    int n_sp;
    using float_t = type::float_t;

    ////////// Functions //////////
    __host__ void set_n_int
    ( const int & n_hv, const int & n_sp )
    {
        this->n_hv = n_hv;
        this->n_sp = n_sp;
        return;
    };

    __host__ __device__ size_t unit_size(  ) const
    {
        constexpr auto s_f( sizeof( float_t ) );
        constexpr auto s_p
                = utils::align_c( sizeof( par_T ),
                                  sizeof( float2_t ) );
        return  utils::align
            ( s_f * ( n_hv + n_sp ) + s_p,
              sizeof( float2_t ) );
    };
};
};                      // namespace extension::radiation
