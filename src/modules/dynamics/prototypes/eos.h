#pragma once

#include "../../../types.h"
#include "../../../device/device.h"
#include "../../../io/args/input.h"
#include "../../../mesh/mesh.h"
#include "../../../utilities/types/crtp.h"
#include "../../../utilities/phys/unit.h"
#include <memory>

namespace dynamics::eos
{
////////////////////////////////////////////////////////////
// Types

using type::   idx_t;
using type::float2_t;

////////////////////////////////////////////////////////////
// Generic kernel function

template< class bdt_T, class eos_T >
__global__ void c2p_kernel
( const bdt_T b_d, const eos_T f_eos, const int step )
{
    return f_eos( b_d, step );
}

////////////////////////////////////////////////////////////
// Abstract base class for EoS

struct base_t
{
    ////////// Basic data //////////
    phys::unit_t< type::float_t > unit;
    int                       n_th_lim;
    type::float_t              d_floor;
    type::float_t              t_floor;
    type::float_t                gamma; // Default value
    type::float_t                  cfl;

    ////////// Initialization //////////
    __host__ virtual void read( const input & args )
    {
        gamma    = args.get< type::float_t >
                 ( "dynamics", "gamma",   1.66667 );
        cfl      = args.get< type::float_t >
                 ( "dynamics", "cfl",         0.3 );
        n_th_lim = args.get< int     >
                 ( "dynamics", "thread_limit", 64 );
        d_floor  = args.get< type::float_t >
                 ( "dynamics", "rho_floor",  1e-9 );
        t_floor  = args.get< type::float_t >
                 ( "dynamics", "tmp_floor",  1e-9 );
        return unit.init( args );
    };

    ////////// Interface //////////
    __host__ virtual int n_hvar(  ) const{ return 2; };

    __host__ virtual type::float_t cvkb_xn_T_h
    ( type::float2_t * x, const type::float2_t & T ) const
    {
        return 1 / ( gamma - 1 );
    };  // In units of ( kb / molecule )

    __host__ virtual void free( device::base_t & f ) {  };
};

////////////////////////////////////////////////////////////
// CRTP interface

template< class derived_T = crtp::dummy_t >
struct interface_t : public base_t
{
    ////////// Types //////////
    __crtp_def_self__( interface_t, derived_T );

    ////////// Host-side interface //////////
    using  dev_t = device     ::base_t;
    using dual_t = mesh::block::dual_t;    
    
    template < class bdt_T > __host__ void
    cons2prim( const bdt_T & b_d, const dual_t &  dd ,
               const int   & stp, const  dev_t & dev ) const
    {
        const auto & self = get_self(  );
        dev.launch( c2p_kernel, self.resource( b_d ),
                    dd .stream, b_d, self,     stp );
        return;
    };
};
};                              // namespace dynamics::eos
