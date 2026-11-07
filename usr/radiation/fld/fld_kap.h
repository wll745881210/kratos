#pragma once

#include "fld_base.h"

namespace rad::fld
{

////////////////////////////////////////////////////////////
//  The IR opacity provider contract  (  the nonideal
// const_eta_t pattern ):  a provider evaluates the gray IR
// mass opacity  kappa_IR  at a cell from the local state;
// the provider object is carried by value into the freeze
// kernels,  so it must stay trivially copyable.   The ppd
// module plugs in here  (  the stage - C species version
// evaluates  y_Gr * sigma_IR  from the dust abundance ).
//  Units:  kappa [ cm^2 / g ] -> code  kappa * rho0 * l0
// ( dimensionless,  the kap_of_x convention ).

//  Constant opacity  (  test / stage - B default ).
struct kap_const_t
{
    type::float_t   kap0;   //  code kappa

    __host__ kap_const_t(  )
    {
        kap0 = 0;
        return;
    };

    template< class bdt_T, class idx_T >
    __device__ __forceinline__ type::float_t eval
    ( const bdt_T &, const idx_T & ) const
    {
        return kap0;
    };
};

};                          // namespace rad::fld
