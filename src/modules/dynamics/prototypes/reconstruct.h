#pragma once

#include "../../../io/args/input.h"
#include "../../../utilities/types/crtp.h"

namespace dynamics::reconstruct
{
////////////////////////////////////////////////////////////
// Abstract base class for reconstruction; using CRTP

struct base_t
{
    __host__ virtual void init( const input & args ) {  };
    __host__ virtual int  space_order(  ) { return  -1; };
};

template< class derived_T = crtp::dummy_t >
struct interface_t : public base_t
{
    __crtp_def_self__( interface_t, derived_T );    
    ////////// Interface //////////
    template< class dat_T, class bdt_T > __device__
    void operator(  ) ( dat_T & d, const bdt_T & b_d ) const
    {
        __syncthreads(  );
        for( int i = threadIdx.x ; i < d.n_feff;
             i    +=  blockDim.x )
            get_self(  ).calc( d, b_d, i );
        return;
    };
};
};                        // namespace dynamics::reconstruct
