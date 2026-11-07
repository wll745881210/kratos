#pragma once

#include "phys.h"

namespace dynamics::mhd::boundary
{
////////////////////////////////////////////////////////////
// Functors for most commonly used hydrodynamic boundaries

template< class prx_T = dynamics::mhd::proxy_t >
struct free_t : phys_base_t< prx_T, free_t< prx_T > >
{
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
        return;
    };
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void  field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int n_gh = bc.n_spa / 2;
        for( int i = -n_gh; i < 0; ++ i )
            for( int n = 0; n < 3; ++ n )
            {
                bc( i, n ) = bc( n );
                bf( i, n ) = bf( n );
            }
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct fixed_t : phys_base_t< prx_T, fixed_t< prx_T > >
{
    //  Dirichlet boundary: ghost state fixed to a
    //  prescribed PRIMITIVE state ( the boundary functors
    //  operate on hyd.w;  the ambient [init] medium, set
    //  by the keeper at init ).  Use at boundaries with
    //  physical inflow;  see the hydro counterpart for the
    //  semantics notes.
    type::float2_t w_fix[ 5 ];
    type::float2_t b_fix[ 3 ];

    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = w_fix[ n ];
        return;
    };
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void  field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int n_gh = bc.n_spa / 2;
        for( int i = -n_gh; i < 0; ++ i )
            for( int n = 0; n < 3; ++ n )
            {
                bc( i, n ) = b_fix[ n ];
                bf( i, n ) = b_fix[ n ];
            }
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct outflow_t : phys_base_t< prx_T, outflow_t< prx_T > >
{
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        const bool right = this->side & 1;
        const int  i_mom ( this->side / 2 + idx::mom );
        if( d( i_mom ) * ( right ? -1 : 1 ) > 0 )
            d( i_mom ) = 0;
        for( int i = -d.n_spa / 2; i < 0; ++ i )
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
        return;
    };
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void  field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int n_gh = bc.n_spa / 2;
        for( int i = -n_gh; i < 0; ++ i )
            for( int n = 0; n < 3; ++ n )
            {
                bc( i, n ) = bc( n );
                bf( i, n ) = bf( n );
            }
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct reflect_t : phys_base_t< prx_T, reflect_t< prx_T > >
{
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        const int i_mom( idx::mom + this->side / 2 );
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            auto *  dl = d.at(  i     );
            auto *  dr = d.at( -i - 1 );
            for( int n = 0; n < d.n_int; ++ n )
                dl[ n ] = dr[ n ] * ( n == i_mom ? -1 : 1 );
        }
        return;
    };
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void  field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int axis ( this->side >> 1 );
        const int n_gh ( bc  .n_spa >> 1 );
        for(  int i = -n_gh; i < 0; ++ i )
            for(  int n = 0; n < 3; ++ n )
            {
                if( n != axis )
                {
                    bf( i, n ) = -bf( -i,     n );
                    bc( i, n ) = -bc( -i - 1, n );
                }
                else
                {
                    bf( i, n ) =  bf( -i,     n );
                    bc( i, n ) =  bc( -i - 1, n );
                }
            }   // Faces mirror at -i, cells at -i-1; the
                // normal component keeps its sign ( div B )
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct pole_t : phys_base_t< prx_T, pole_t< prx_T > >
{
    //  Polar-axis boundary ( theta sides of a spherical
    //  run; the keeper rejects it elsewhere ).  The
    //  through-pole continuation maps ( theta, phi ) to
    //  ( -theta, phi + pi ): the radial components are
    //  even, both tangential ones ( theta and phi ) flip
    //  sign, for hydro AND for the magnetic field -- the
    //  map is a proper rotation, unlike the mirror of
    //  reflect_t.  Same component parities as Athena++'s
    //  polar wedge.  The same-phi wedge source is exact
    //  for axisymmetric fields; generic non-axisymmetric
    //  runs additionally need phi-ring averaging near the
    //  pole ( see mesh::rebin_t ).
    //  Also pins B_theta = 0 on the pole face itself: that
    //  face has zero area ( CT skips it ), but c2p reads
    //  it and the radial boundaries copy interior values
    //  into its corner columns -- the phi-averaged
    //  regularity value at the axis is zero.
    static constexpr bool pin_bnd_face = true;
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            auto *  dl = d.at(  i     );
            auto *  dr = d.at( -i - 1 );
            for( int n = 0; n < d.n_int; ++ n )
                dl[ n ] = dr[ n ] * ( n == idx::mom + 1 ||
                                      n == idx::mom + 2
                                      ? -1 : 1 );
        }
        return;
    };
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void  field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int n_gh ( bc.n_spa >> 1 );
        for(  int i = -n_gh; i < 0; ++ i )
            for(  int n = 0; n < 3; ++ n )
            {
                const int sgn ( n == 0 ? 1 : -1 );
                bf( i, n ) = sgn * bf( -i,     n );
                bc( i, n ) = sgn * bc( -i - 1, n );
            }   // B_r even, B_theta and B_phi flip; the
                // lattice pairing matches reflect_t.
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct axis_t : phys_base_t< prx_T, axis_t< prx_T > >
{
    //  Rotation-axis boundary ( r sides of a cylindrical
    //  run at r = 0; the keeper rejects it elsewhere ).
    //  The through-axis continuation maps ( r, phi ) to
    //  ( -r, phi + pi ): the z components are even, the
    //  r and phi components flip sign, for hydro AND for
    //  the magnetic field.  Same-phi wedge source, exact
    //  for axisymmetric fields; generic non-axisymmetric
    //  runs need phi-ring averaging near the axis.
    //  Also pins B_r = 0 on the axis face itself: that
    //  face has zero area ( CT skips it ), but c2p reads
    //  it and the tangential boundaries copy interior
    //  values into its corner columns -- the phi-averaged
    //  regularity value on the axis is zero.
    static constexpr bool pin_bnd_face = true;
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            auto *  dl = d.at(  i     );
            auto *  dr = d.at( -i - 1 );
            for( int n = 0; n < d.n_int; ++ n )
                dl[ n ] = dr[ n ] * ( n == idx::mom ||
                                      n == idx::mom + 1
                                      ? -1 : 1 );
        }
        return;
    };
    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void  field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int n_gh ( bc.n_spa >> 1 );
        for(  int i = -n_gh; i < 0; ++ i )
            for(  int n = 0; n < 3; ++ n )
            {
                const int sgn ( n == 2 ? 1 : -1 );
                bf( i, n ) = sgn * bf( -i,     n );
                bc( i, n ) = sgn * bc( -i - 1, n );
            }   // B_z even, B_r and B_phi flip; the
                // lattice pairing matches reflect_t.
        return;
    };
};

};                    // namespace dynamics::mhd::boundary
