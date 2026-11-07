#pragma once

#include <cmath>
#include <vector>

#include "edge_mg_cplx.h"
#include "edge_mg_geo.h"

//  the __global__ layer of this file lives in the kernels/ subdirectory here.
#include "kernels/ni_ie_kernels.h"

////////////////////////////////////////////////////////////
//  THE SUB-CELL eta GRADIENT --  one implementation,  host
//  and device.
//
//  The operator wants eta at every QUADRATURE POINT, not at
//  the cell centre:
//
//      eta( x ) = eta_c + ( grad eta )_c . ( x - x_c )
//
//  and a single tensor per cell is only FIRST order, for any
//  spatially varying eta, through two independent mechanisms:
//
//    (1) K  drops to first order,  because curl phi_e flips
//  sign across the six face patches of a cell, so the
//        linear part of ( eta - eta_h ) does NOT integrate
//        away;
//  (2) the SOURCE loses the whole ( grad eta ) x J term,
//        which is zero order,  not a loss of order.
//
//  The gradient is NOT stored: the caller keeps 6 numbers a
//  cell and derives 18 here,  from a neighbour stencil.
//
//  THE FIVE THINGS THIS FILE ENCODES ( each is a trap that
//  costs accuracy or reads the wrong node ):
//
//  * the cell centre is the MU midpoint of the theta cell,
//  theta_c = arccos( ( cos th_lo + cos th_hi ) / 2 ), NOT
//      the arithmetic midpoint ( 17.8% of the gradient );
//  * the two step sizes of a one-sided stencil are the TWO
//  DISTANCES TO THE TWO NEAREST NODES. Reusing the first
//  one is exact on a uniform grid and wrong by 31% at the
//      pole,  where the mu-parametrised centres go like
//      sqrt( 2 ( j + 1/2 ) dmu )  and  h2/h1 = 0.688
//  INDEPENDENT of dmu -- that is what stalls the polar row;
//    * at the HI end the nodes are  0, -a, -(a+b)  and the
//  Lagrange weights are the NEGATED ones ( 144% of error
//      on the outer wall when they are not );
//    * the one-sided stencil is ANCHORED AT THE NODE.  A
//  3-point forward stencil needs i + 2 <= n - 1, so the
//  SECOND-TO-LAST row is also inside the last two and must
//  go backward -- special-casing only the last row reads
//      the next row's data and returns ANOTHER node's
//      derivative;
//  * the rotation is d/dx_a = sum_q ( e_q )_a ( 1 / C_qq )
//  d/dq with C = diag( 1, r, r sin theta ): the 1/C sits
//      on the INNER index q,  not on a ( swapping them is a
//      6.3e-2 error against a 1.4e-9 correct one ).
//
//  The GATHER ( which three lattice values to mix ) is the
//  caller's: on the host it is a std::vector, in a kernel
//  it
//  is a strided read. The STENCIL -- where all five traps
//  live -- is here,  once.

namespace mhd_ee::ni::edge_mg
{
using type::float_t;


//////////////////////////////////////////////////////////
//  The lattice geometry the gradient differentiates.

//  theta of the cell centre,   in the MU parameterisation



//  The Cartesian centre of one cell of the eta lattice: the
//  point a
//  positional provider must sample at, and the point the
//  Taylor expansion
//  is taken about.
__host__ __device__ inline void eta_centre
( const geo_sph_t & g, const idx_t & i, float2_t ( & x )[ 3 ] )
{
    const float2_t r  = eta_r_centre ( g, i[ 0 ] );
    const float2_t th = eta_theta_centre ( g, i[ 1 ] );
    const float2_t ph = eta_ph_centre ( g, i[ 2 ] );
    const float2_t st = sin( th );
    const float2_t ct = cos( th );
    x[ 0 ] = r * st * cos( ph );
    x[ 1 ] = r * st * sin( ph );
    x[ 2 ] = r * ct;
    return;
};

//////////////////////////////////////////////////////////
//  THE 1-D STENCIL of the first derivative at node `i` of a
//  lattice with `n` nodes.
//
//  The caller passes the neighbour coordinates a branch may
//  need -- cm2, cm, c0, cp, cp2 = nodes i-2 .. i+2. A
//  branch
//  that does not use one of them never reads it, so a
//  caller
//  at a wall may pass anything for the outside node.
//
//  On return:
//      d = w[ 0 ] * v[ id[ 0 ] ] + w[ 1 ] * v[ id[ 1 ] ]
//        + w[ 2 ] * v[ id[ 2 ] ]
//
//  Centred inside, one-sided within two nodes of either
//  end,
//  and every step size taken from the coordinates
//  themselves.

//////////////////////////////////////////////////////////
//  The derivative at one node of a full 1-D line ( host
//  gather ).

__host__ float_t _d1_line
( const std::vector< float_t > & cc, const std::vector<
    float_t > & v,
  const int & i );

//////////////////////////////////////////////////////////
//  The HOST gradient of a whole block.
//
//  lat in : n_cell * 6, the per-cell tensor ( Voigt )
//      gr   out :  n_cell * 18,  gr[ ic * 18 + a * 6 + q ]
//  = d eta_q / d x_a in CARTESIAN coordinates,
//  i.e. exactly assemble_cell's p_deta layout.
//
//  The cell centre used here is eta_centre -- so a provider
//  that samples
//  eta on this module's lattice needs to do nothing extra,
//  and one that
//  samples elsewhere must pass its own centres ( see the
//  device twin ).
__host__ void eta_grad_lat
( const geo_sph_t & g, const cplx_t & c,
  const std::vector< float_t > & lat, std::vector< float_t >
      & gr );

//////////////////////////////////////////////////////////
//  The DEVICE gradient: the same stencil, a strided gather.
//
//  One thread per cell over the flat cell order ( i fastest
//  ), so the
//  neighbours are +-1, +-nr, +-nr*nt ( wrapped in phi ).
//  Output layout
//  is assemble_cell's p_deta:  gr[ ic * 18 + a * 6 + q ].
//
//  This is NOT bit-identical to the host path: the device
//  compiler may contract the rotation's sum with FMA. The two
//  are therefore compared to a tolerance ( 1e-13 relative ),
//  not bitwise -- the stencil weights themselves are the same
//  expressions in the same order.
//  STATIC: a header-defined __global__ that two TUs include is
//  emitted once per TU and nvlink refuses the second
//  definition.

};  // namespace edge_mg
