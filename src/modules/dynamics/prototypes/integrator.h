#pragma once

#include "../../../io/args/input.h"

namespace dynamics
{
class base_t;

namespace integrator
{
////////////////////////////////////////////////////////////
// Generic kernel function

template< class dat_T, class bdt_T, class rec_T,
          class rie_T, class eos_T, class itg_T >
__global__ void flux_kernel
( const bdt_T   b_d, const int    axis, const int    step,
  const rec_T f_rec, const rie_T f_rie, const eos_T f_eos,
  const itg_T f_itg )
{
    //////////////////////////////////////////////////
    // Data holder for the Godunov modules
    __dyn_shared__    ( char, var_sh        );
    dat_T d( b_d, axis, step, var_sh, f_eos );

    //////////////////////////////////////////////////
    // Proceduresd of Godunov solver
    f_rec( d, b_d        );   // Reconstruction
    f_rie( d, b_d, f_eos );   // Riemann solver
    f_itg( d, b_d, f_eos );   // Add fluxes & additionals
    return;
}

////////////////////////////////////////////////////////////
// Abstract base class for integrator

struct base_t
{
    ////////// Resource dispatching //////////
    int n_th_lim;

    ////////// Con-/destructor //////////
    __host__ base_t(  ) {  };

    ////////// Initialization //////////
    __host__ virtual void init
    ( const input & args, const::dynamics::base_t & mod )
    {
        n_th_lim = args.get< int >
                 ( "dynamics", "thread_limit", 64 );
        return;
    };

    __host__ virtual void finalize
    ( const::dynamics::base_t & mod ) {  };

    ////////// Temporal order //////////
    __host__ virtual int time_order(  ) const { return 1; };
};

};                              // namespace integrator
};                              // namespace dynamic
