#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>

#include "../../../../../../src/device/device.h"
#include "../../../../../../src/mesh/geometry/geometry.h"
#include "../../../../../../src/mesh/types/dat_3d.h"
#include "../../../../../../src/types.h"
#include "edge_mg_grad.h"

//  the __global__ layer of this file lives in the kernels/ subdirectory here.
#include "kernels/ni_ie_kernels.h"

////////////////////////////////////////////////////////////
//  THE eta LATTICE -- PACKED WHERE eta ALREADY IS.
//
//  ( P5 / K1.  Before this file,  every implicit step did )
//
//      (1) read the whole eta array back to the host
//          ( 6 numbers a cell + a ghost ring ),
//      (2) repack it on the host into the per-cell
//          float2_t[ 6 ] lattice the assembly reads,
//      (3) upload the lattice again.
//
//  That is one of the four per-grid read-backs and one of
//  the three per-grid writes the P5 gate counts,  and it is
//  PURE DATA MOVEMENT:  the lattice is a REARRANGEMENT of
//  eta,  component for component,  nothing else.  Packing
//  it on the device deletes both transfers and the host
//  loop.
//
//  THE THREE THINGS THAT MAKE THIS FILE CORRECT ( and the
//  reason a gate exists even for a copy ):
//
//  * the CELL BOX is  geo.n_ceff,  NOT  eta.n_cell:  the
//    production eta carries one ghost ring ( n_gh = 1 ),
//    while the lattice holds exactly
//    nc = n_ceff[0] * n_ceff[1] * n_ceff[2]  cells.
//    `at( i )` is ghost-aware,  so a cell index inside the
//    box lands on the interior slot -- the very slot the
//    host repack reads;
//
//  * axis 0 ( i ) is the FAST index of the packing and
//    lat[ ic * 6 + q ]  is the storage order the assembly
//    expects ( q = xx yy zz xy xz yz );
//
//  * the arithmetic is a copy of float_t -- no
//    transcendentals,  no interpolation -- so the device
//    answer is BIT identical to the host one,  not merely
//    close.  The gate demands  rel == 0.
//
//  The eta array reaches the kernel BY VALUE as the
//  dat_3d_t itself ( trivially copyable:  one pointer plus
//  six scalars ).  That keeps `at(  )` -- and therefore the
//  ghost offset -- in ONE implementation shared with the
//  host code.  A raw pointer parameter here would mean a
//  SECOND copy of the offset arithmetic,  i.e. a second
//  thing to get wrong,  and getting it wrong is silent (
//  trap M63:  a probe that reads the wrong array does not
//  crash ).

namespace mhd_ee::ni::edge_mg
{

using type::  idx_t;
using type::float_t;

////////////////////////////////////////////////////////////
//  op:  eta ( ghost-aware ) -> the per-cell lattice
////////////////////////////////////////////////////////////

struct eta_lat_t
{
    mesh::dat_3d_t< float_t > eta;
    idx_t                     n_ceff;
    float2_t *                p_lat  { nullptr };
    int                       n_cell { 0 };

    __device__ void apply( const int & ic ) const
    {
        //  the cell box,  axis 0 fastest -- the SAME i the
        //  host repack builds;  `_eta_lat_host` ( edge_mg.cpp
        //  ) is the one authority on this order
        const idx_t i( ic % int( n_ceff[ 0 ] ),
                      ( ic / int( n_ceff[ 0 ] ) )
                          % int( n_ceff[ 1 ] ),
                       ic / int( n_ceff[ 0 ] * n_ceff[ 1 ] ) );
        //  ghost-aware read:  at(  ) adds n_gh and walks the
        //  padded box,  so the ghost ring is never touched
        //  from inside the cell box
        const float_t * et = eta.at( i );
        const std::size_t o = std::size_t( ic ) * 6;
        for( int q = 0; q < 6; ++ q )
        {
            p_lat[ o + std::size_t( q ) ] = float2_t( et[ q ] );
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


////////////////////////////////////////////////////////////
//  P5 / K1:  the eta lattice,  packed on the device.
//
//  Replaces   ( read back eta -> host repack -> upload the
//  lattice )   with one kernel:  no per-cell host traffic
//  inside the implicit step.
//
//  The sync is here because every probe_* launcher in
//  edge_mg.cpp ends with one;  a production caller that
//  keeps its own barrier can drop it later.
////////////////////////////////////////////////////////////

__host__ void ni_ie_eta_lat_dev
( device::base_t & dev, const mesh::geo_t & geo,
  const mesh::dat_3d_t< float_t > & eta,
  float2_t * p_lat, const int & n_th_lim );

////////////////////////////////////////////////////////////
//  THE SUB-CELL STRETCH ( P5 / K1,  optional ).
//
//  `par.subcell` builds one more per-cell lattice on the
//  host:  the CELL CENTRE x_c  ( 3 float2_t a cell ),  fed
//  to  eta( x ) = eta_c + grad . ( x - x_c ).  That loop
//  runs  nc  transcendentals on the host -- sin,  cos and
//  an  arccos  -- and then uploads the result.
//
//  `eta_centre` is already  __host__ __device__  ( it and
//  its three helpers live in edge_mg_grad.h ),  so the loop
//  moves to the device as it stands.  What does NOT move
//  bit-exactly is  sin/cos/acos  themselves:  the CUDA and
//  the libm versions round differently,  so this one is
//  compared with a tolerance,  NOT with rel == 0 -- and the
//  arccos at the theta midpoint is ill-conditioned near the
//  pole,  which is why the gate reports both the max
//  absolute and the norm-wise relative error.
////////////////////////////////////////////////////////////

struct eta_xc_t
{
    geo_sph_t  g;
    idx_t      n_ceff;
    float2_t * p_xc   { nullptr };
    int        n_cell { 0 };

    __device__ void apply( const int & ic ) const
    {
        const idx_t i( ic % int( n_ceff[ 0 ] ),
                      ( ic / int( n_ceff[ 0 ] ) )
                          % int( n_ceff[ 1 ] ),
                       ic / int( n_ceff[ 0 ] * n_ceff[ 1 ] ) );
        float2_t xx[ 3 ];
        eta_centre( g, i, xx );
        const std::size_t o = std::size_t( ic ) * 3;
        for( int q = 0; q < 3; ++ q )
        {
            p_xc[ o + std::size_t( q ) ] = xx[ q ];
        }
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim ) const;
};


__host__ void ni_ie_eta_xc_dev
( device::base_t & dev, const geo_sph_t & g,
  const idx_t & n_ceff, float2_t * p_xc, const int & n_th_lim );

};  // namespace edge_mg