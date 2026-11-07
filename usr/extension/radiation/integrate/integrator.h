#pragma once

#include "../../../../src/modules/particle/evolve/integrator.h"

namespace extension::radiation
{
////////////////////////////////////////////////////////////
// Integrator

struct itg_t : particle::integrate::base_t< itg_t >
{
    ////////// Type and Data //////////
    using super_t = particle::integrate::base_t< itg_t >;
    int   n_stride;
    //  The  VET  chi - moment  gate  (  2026 - 10 - 15
    // fld-ppd-integration-plan.md § 3.19 ):  1  =  the
    // photon  deposit  also  accumulates  the  direction
    // moments  into  rad.chi_m.   Default  0  =  the
    // legacy  deposit  only  (  bit - identical  for  the
    // existing  users ).   A  subclass  (  e.g.  the  ppd
    // bench_itg_t )  reads  its  par  into  this:
    int   vet_gate;

    __host__ itg_t(  )
    {
        vet_gate = 0;
        return;
    };

    ////////// Functions //////////
    __host__ virtual void init
    ( const input &, particle::base_t & ) override;

    template < class pol_T >
    __host__ std::tuple< dim3, dim3, int >
    resource( const pol_T & pool ) const
    {
        dim3 n_bl(( pool.n_par_eff( ) + n_th - 1 ) / n_th );
        //  the  shared  slot  covers  the  ALIGNED  per -
        // photon  dat  (  the  photon.h  load  strides  by
        // size_dat / sizeof( float_t )  --  see  the  2026
        // - 10 - 15  note  there ):
        const auto s_f( sizeof( type::float_t ) );
        const auto n_slot( utils::align
            ( n_stride * s_f, sizeof( type::float2_t ) )
            / s_f );
        const auto s_sh = n_th * n_slot * s_f;
        return std::make_tuple ( n_bl, dim3( n_th ), s_sh );
    };
};

};                     // namespace extension::radiation
