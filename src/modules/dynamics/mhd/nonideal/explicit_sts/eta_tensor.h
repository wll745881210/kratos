#pragma once

#include "../../../../../../src/types.h"
#include "../../../../../../src/utilities/functions.h"

////////////////////////////////////////////////////////////
//  THE eta TENSOR OF ONE CELL -- the anisotropic
//  diffusivity
//  of the implicit ( edge-centred ) diffusion leg,  as the
//  operator consumes it.
//
//      eta = ( eo + ea ) I  -  ea b b ,      b = B / | B |
//
//  i.e. Ohmic ( eo ) plus ambipolar ( ea ), the SYMMETRIC
//  part
//  of the diffusivity.  The Hall term is antisymmetric and
//  cannot enter a symmetric operator; it stays on the
//  explicit
//  emf leg.
//
//  AMBIENT BRANCH. When |B| = 0 the direction is undefined,
//  so
//  b = 0 and eta = eo I.  It must stay finite and SPD:  an
//  unnormalised 0/0 here poisons every later eta . J, and a
//  controlled ambient region ( a dead zone, a vacuum halo )
//  reaches |B| = 0 by construction -- the polar axis is not
//  the
//  only place where this branch is taken.
//
//  WHY IT IS A FUNCTION AND NOT A KERNEL BODY.  The kernel
//  ( ni_eta_fill_k,  nonideal.h ) and the gate
//  ( usr_ext/mhd_ee/gate_eta_fill.cpp ) must evaluate the
//  SAME
//  algebra: a second copy is exactly where an I - b b gets
//  transposed,  or where the ambient branch survives in one
//  copy and not in the other.  It is `inline` because a
//  non-inline __host__ __device__ free function is emitted
//  once
//  per TU under -rdc=true, i.e. an nvlink multiple
//  definition
//  ( handover section 5,  trap 11 ).
//
//  Layout of `et` is the module's Voigt order,
//      { xx, yy, zz, xy, xz, yz }
//  which is what edge_mg::assemble_cell / cell_blk_t
//  expect.
//  On return bh holds the UNIT direction ( or zero, in the
//  ambient branch ), so a caller that also needs b does not
//  normalise twice.

namespace mhd_ee::ni
{
using type::float_t;


template< class f2_T >
__host__ __device__ inline void ni_eta_tensor
( const f2_T & eo, const f2_T & ea, f2_T ( & bh )[ 3 ],
  f2_T ( & et )[ 6 ] )
{
    f2_T b2( 0 );
    for( int a = 0; a < 3; ++ a )
    {
        b2 += bh[ a ] * bh[ a ];
    }
    f2_T ea_iso( 0 );
    if( b2 > f2_T( 0 ) )
    {
        ea_iso = ea;
        //  __sqrt1 is a DEVICE intrinsic ( __dsqrt_rn /
        //  __fsqrt_rn ), and this
        //  function must also compile for the HOST pass --
        //  that is what lets
        //  the gate call it at all. sqrt() is the same
        //  instruction in
        //  round-to-nearest, so the device bits do not
        //  change; hoisting it
        //  out of the loop is likewise exact ( the
        //  intrinsic is deterministic,
        //  so three identical divisions stay three
        //  identical divisions ).
#if defined( __CUDA_ARCH__ )
        const f2_T sn = __sqrt1( b2 );
#else
        const f2_T sn = sqrt( b2 );
#endif
        for( int a = 0; a < 3; ++ a )
        {
            bh[ a ] /= sn;
        }
    }
    const f2_T dg( eo + ea_iso );
    et[ 0 ] = dg - ea_iso * bh[ 0 ] * bh[ 0 ];
    et[ 1 ] = dg - ea_iso * bh[ 1 ] * bh[ 1 ];
    et[ 2 ] = dg - ea_iso * bh[ 2 ] * bh[ 2 ];
    et[ 3 ] =    - ea_iso * bh[ 0 ] * bh[ 1 ];
    et[ 4 ] =    - ea_iso * bh[ 0 ] * bh[ 2 ];
    et[ 5 ] =    - ea_iso * bh[ 1 ] * bh[ 2 ];
    return;
};

};  // namespace nonideal
