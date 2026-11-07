#pragma once

#include "../../../../../../src/types.h"
#include "../../../../../../src/mesh/geometry/geometry.h"
#include "../../../../../../src/utilities/functions.h"

////////////////////////////////////////////////////////////
//  Geometry policies for the assembly.
//
//  A policy answers ONE question,  in the exact form the
//  weak form needs:
//
//      at cell i and local coordinate xi in [0,1]^3,
//      give  x,  J = [ dx/dxi_0 | dx/dxi_1 | dx/dxi_2 ]
//      ( ALREADY multiplied by the cell's spans )  and
//      det J.
//
//  The Jacobian must be ANALYTIC,  never a trilinear map of
//  the eight corners:  on a sphere that map is a chord in
//  every curved direction,  which is singular exactly where
//  the sphere is special -- det J = 0 at n_phi = 1,  and the
//  boundary load is far too large at small n_phi.
//
//  Policies are plain POD structs and are passed to the
//  kernels BY VALUE,  so no device-side pointer chasing and
//  no virtual dispatch.
//
//  Only the cartesian policy is implemented here.  The
//  spherical one ( mu = cos(theta) parameterisation ) needs
//  only its own `map` / `node_x`,  because the assembly
//  kernel is templated on the policy.

namespace mhd_ee::ni::edge_mg
{

using type::   idx_t;
using type:: float_t;
using type::float2_t;
using mesh::   geo_t;

////////////////////////////////////////////////////////////
//  THE POLE TEST -- and the only place it is written down.
//
//  Never test theta == 0: a mesh that starts at 1e-14, or a
//  theta
//  grid built from a mu grid, gives sin(theta) ~ 1e-16
//  rather than 0,
//  and then the phi edges on that ring have a length that
//  is small but
//  NOT zero -- the matrix is singular in exact arithmetic
//  and merely
//  awful in floating point. The criterion is relative to
//  the ring's
//  own circumference: sin(theta) < tol means "this ring is
//  the axis",
//  with tol = 1e-8 ( the shell's own metric ratio, not a
//  magic
//  constant: 1e-8 of a radian of circumference is far below
//  any
//  resolved cell, and far above the 1e-16 round-off of sin
//  ).
//
//  TWO consumers,  and they must never disagree:
//  geo_sph_from writes the flags into the policy,  and every
//  cplx_t of a block takes them from pole_flags ( below ).  A
//  complex built WITHOUT them keeps the zero-length phi edges
//  on the axis as ordinary degrees of freedom,  so
//  `M1 + dt K` is near singular there and the CG silently
//  returns without converging.

__host__ bool sph_is_axis( const float2_t & th );

//  The pole flags of ONE BLOCK, decided from its geometry
//  alone -- not from
//  the caller's opinion and not from the resolution.
__host__ void pole_flags
( const mesh::geo_t & geo, const idx_t & n_ceff, bool & lo, bool & hi );

//////////////////////////////////////////////////////////
//  Cartesian ( uniform or stretched along each axis ).
//
//  Built on the host from the block's geo_t: the coordinate
//  table gives the cell faces, so a stretched grid needs no
//  extra data -- x_fc is reproduced from x0 + i * dx only
//  when the axis is uniform,  which setup() checks.

struct geo_cart_t
{
    ////////// Data //////////
    float2_t  x0[ 3 ];      // coordinate of the first cell face
    float2_t  dx[ 3 ];      // cell size ( of THIS block )

    ////////// Functions //////////
    __host__ __device__ void map
    ( const idx_t & i, const float2_t ( & xi )[ 3 ],
      float2_t ( & x )[ 3 ], float2_t ( & J )[ 3 ][ 3 ],
      float2_t & det ) const
    {
        for( int a = 0; a < 3; ++ a )
        {
            x[ a ]      = x0[ a ] + ( float2_t( i[ a ] ) + xi[ a ] )
                        * dx[ a ];
            for( int q = 0; q < 3; ++ q )
            {
                J[ a ][ q ] = ( a == q ) ? dx[ a ] : float2_t( 0 );
            }
        }
        det = dx[ 0 ] * dx[ 1 ] * dx[ 2 ];
        return;
    };

    //  A NODE ( unwrapped ) to a point: edge vectors are
    //  built from it, and on a periodic axis the node index
    //  n_a is a real, distinct coordinate,  so two nodes one
    //  period apart are NOT the same point.
    __host__ __device__ void node_x
    ( const idx_t & nd, float2_t ( & x )[ 3 ] ) const
    {
        for( int a = 0; a < 3; ++ a )
        {
            x[ a ] = x0[ a ] + float2_t( nd[ a ] ) * dx[ a ];
        }
        return;
    };
};

////////////////////////////////////////////////////////////
//  Spherical ( r, theta, phi ) with the theta direction
//  parametrised uniformly in  mu = cos( theta ),  NOT in
//  theta.
//
//  This single choice is what makes the pole converge:
//  with a theta parametrisation the
//  polar cell's energy integrand is (d_theta w)^2 / sin
//  theta
//  with  d_theta w  constant,  i.e. ~ 1/theta -- a log
//  divergence, rate 1.00. In mu the poloidal flux function
//  psi = r sin(theta) A_phi  is a polynomial near the axis,
//  and  d(theta)/d(xi) = dmu / sin(theta)  cancels the
//  sin(theta) of the volume element:
//
//  det J = dr * dmu * dphi * r^2 ( finite at the pole )
//
//  and no basis function carries a 1/sin(theta).
//
//  A rate near 1.0 on the global mesh is the signature of a
//  theta-uniform policy running instead of this one.

struct geo_sph_t
{
    //  set by geo_sph_from with a tolerance on sin(theta);
    //  see there.
    bool pole_lo { false };
    bool pole_hi { false };

    ////////// Data //////////
    float2_t   r0,  dr;      // r = r0 + i0 * dr   (+ xi0)
    float2_t  th0, dth;      // theta at the cell faces
    float2_t  ph0, dph;      // phi at the cell faces ( dph = 2 pi
                             // when the phi axis is thin )

    ////////// Functions //////////
    __host__ __device__ void map
    ( const idx_t & i, const float2_t ( & xi )[ 3 ],
      float2_t ( & x )[ 3 ], float2_t ( & J )[ 3 ][ 3 ],
      float2_t & det ) const
    {
        const float2_t r  = r0 + ( float2_t( i[ 0 ] ) + xi[ 0 ] ) * dr;
        const float2_t ta = th0 + float2_t( i[ 1 ] ) * dth;
        const float2_t ma = cos( ta );
        const float2_t mb = cos( ta + dth );
        const float2_t mu = ma + xi[ 1 ] * ( mb - ma );
        //  theta from mu,  numerically stable near mu = 1:
        //  arccos( 1 - eps ) loses ~ sqrt( eps_machine/eps
        //  )
        const float2_t th = float2_t( 2 )
            * asin( sqrt( __max1( float2_t( 0.5 )
                                * ( float2_t( 1 ) - mu ),
                                float2_t( 0 ) ) ) );
        const float2_t st = sin( th );
        const float2_t ct = cos( th );
        const float2_t dtdxi = -( mb - ma )
                             / __max1( st, float2_t( 1.e-30 ) );
        const float2_t ph = ph0
            + ( float2_t( i[ 2 ] ) + xi[ 2 ] ) * dph;
        const float2_t sp = sin( ph );
        const float2_t cp = cos( ph );
        x[ 0 ] = r * st * cp;
        x[ 1 ] = r * st * sp;
        x[ 2 ] = r * ct;
        //  J = [ dx/dxi_0 | dx/dxi_1 | dx/dxi_2 ]
        J[ 0 ][ 0 ] =  dr * st * cp;
        J[ 1 ][ 0 ] =  dr * st * sp;
        J[ 2 ][ 0 ] =  dr * ct;
        J[ 0 ][ 1 ] =  r * ct * cp * dtdxi;
        J[ 1 ][ 1 ] =  r * ct * sp * dtdxi;
        J[ 2 ][ 1 ] = -r * st      * dtdxi;
        J[ 0 ][ 2 ] = -r * st * sp * dph;
        J[ 1 ][ 2 ] =  r * st * cp * dph;
        J[ 2 ][ 2 ] =  float2_t( 0 );
        //  orthogonal columns,  right-handed for dtdxi > 0
        det = fabs( dr * dtdxi * r * r * st * dph );
        return;
    };

    __host__ __device__ void node_x
    ( const idx_t & nd, float2_t ( & x )[ 3 ] ) const
    {
        const float2_t r  = r0  + float2_t( nd[ 0 ] ) * dr;
        const float2_t th = th0 + float2_t( nd[ 1 ] ) * dth;
        const float2_t ph = ph0 + float2_t( nd[ 2 ] ) * dph;
        x[ 0 ] = r * sin( th ) * cos( ph );
        x[ 1 ] = r * sin( th ) * sin( ph );
        x[ 2 ] = r * cos( th );
        return;
    };
};

////////////////////////////////////////////////////////////
//  Build the policy of one block from its geo_t.
//
//  Returns false ( and the caller reports which axis ) when
//  the block is not a uniform cartesian box -- the stretched
//  variant needs the x_fc table on the device and is not
//  implemented yet.

__host__ bool geo_cart_from
( const geo_t & geo, geo_cart_t & g );

//  The spherical policy needs a uniform ( r, theta ) grid:
//  the mu parameterisation is per cell and the tables are
//  not carried on the device yet.  phi is uniform by
//  construction ( stretched phi does not exist here ).

__host__ bool geo_sph_from
( const geo_t & geo, geo_sph_t & g );

};  // namespace edge_mg
