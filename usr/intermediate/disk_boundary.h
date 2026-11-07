#pragma once

#include "../../src/modules/dynamics/mhd/boundary/phys.h"
#include "../../src/modules/dynamics/hydro/boundary/phys.h"
#include "../../src/modules/chemistry/chem_mhd/mhd.h"
#include "../../src/utilities/phys/constants.h"
#include "disk_profile.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  PPD radial boundaries ( design.md sec. 6;  the WBG19
//  ghost rules, microphys disk_wind / disk_boundary.cpp ):
//
//  ppd_in ( inner r ):  hydro ghosts = verbatim copy of the
//  innermost active cell ( species included );  face-field
//  ghosts extrapolated B_r ~ r^-2,  B_th flat,  B_phi ~ r^-1.
//
//  ppd_out ( outer r ):  outflow with the inflow veto on
//  v_r;  rho, pre ~ r^-2 extrapolation;  same field
//  extrapolation.
//
//  Strip indexing ( phys_base_t ):  index 0 = the boundary
//  active cell / boundary face,  i < 0 the ghost rows.
//
//  The ghost bodies below reach the base only for side /
//  get_self / the 1-D ghost buffer,  so the same classes
//  serve the MHD and the hydro keepers:  pick the phys base
//  matching the proxy.

template< class prx_T > struct phys_base_of
{
    template< class d_T > using base_t
        = dynamics::mhd::boundary::phys_base_t< prx_T, d_T >;
};
template<> struct phys_base_of< dynamics::hydro::proxy_t >
{
    template< class d_T > using base_t
        = dynamics::hydro::boundary::phys_base_t
          < dynamics::hydro::proxy_t, d_T >;
};

template< class prx_T = dynamics::mhd::proxy_t >
struct ppd_in_t : phys_base_of< prx_T >
    ::template base_t< ppd_in_t< prx_T > >
{
    ////////// Data //////////
    interm::disk_profile_t prof;
    //  [boundary] omega_star:  the inner boundary 's rigid
    //  rotation rate ( code,  1 / t0 )  --  the ghost cells '
    //  v_phi = omega_star * R:  the stellar surface holds
    //  the field and drags it at this rate ( the disk - wind
    //  anchoring ).   0 ( default ) = Omega_K( r_in ).
    type::float2_t omega_star;

    ////////// Functions //////////
    __host__ virtual void init
    ( const input & args, mesh::mod_base_t & mod ) override
    {
        prof.read( args );
        omega_star = args.get< type::float2_t >
                     ( "boundary", "omega_star", 0 );
        if( omega_star == 0 )
        {
            type::coord2_t x_min;
            args( "mesh", "x_min", x_min );
            omega_star = sqrt( prof.gm
                / ( x_min[ 0 ] * x_min[ 0 ] * x_min[ 0 ] ) );
        }
        return;
    };

    // Ghosts:  rho / pre from the IC profile ( the zero -
    // gradient copy flattens the real gradient and forces
    // the first active cell at first order ),  velocities
    // COPIED from the active cell so inflow / the polar
    // rain transmits through the boundary ( a v = 0 ghost
    // blocks the through - flow and piles mass at r_in ).
    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        const int  ax    = this->side >> 1;
        const bool right = this->side & 1;
        const auto & geo( b_d.geo );
        const int  n_ceff = geo.n_ceff[ ax ];
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            int ai[ 3 ];
            for( int a = 0; a < 3; ++ a )
                ai[ a ] = i_3d[ a ];
            ai[ ax ] = right ? n_ceff - i - 1 : i;
            type::coord_t x;
            for( int a = 0; a < 3; ++ a )
                x[ a ] = geo.x_cc( a, ai[ a ] );
            type::float2_t rho, pre, v_phi;
            prof.hydro( x, rho, pre, v_phi );
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
            d( i, idx::rho ) = rho;
            d( i, idx::ene ) = pre;
            //  The rigid - rotation angular momentum:  the
            //  stellar surface drags the field at
            //  omega_star ( the wind - anchoring BC ).
            //  NOTE:  the boundary functors write the
            //  PRIMITIVE array hyd.w --  the mom slots hold
            //  the VELOCITY there.
            if( omega_star > 0 )
            {
                const auto Rg = x[ 0 ] * sin( x[ 1 ] );
                d( i, idx::mom + 2 ) = omega_star * Rg;
            }
        }
        return;
    };

    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int  n_gh = bc.n_spa / 2;
        const auto & geo( b_d.geo );
        const int  c0  = 0;      // boundary active cell
        const int  f0  = 0;      // boundary face
        const auto r_f0 = geo.x_fc( 0, f0 );
        const auto r_c0 = geo.x_cc( 0, c0 );
        for( int i = -n_gh; i < 0; ++ i )
        {
            const auto rr = geo.x_fc( 0, f0 + i ) / r_f0;
            const auto rc = geo.x_cc( 0, c0 + i ) / r_c0;
            bf( i, 0 ) = bf( 0, 0 ) / ( rr * rr );
            bf( i, 1 ) = bf( 0, 1 );
            bf( i, 2 ) = bf( 0, 2 ) / rc;
            bc( i, 0 ) = bc( 0, 0 ) / ( rc * rc );
            bc( i, 1 ) = bc( 0, 1 );
            bc( i, 2 ) = bc( 0, 2 ) / rc;
        }
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct ppd_abs_t : phys_base_of< prx_T >
    ::template base_t< ppd_abs_t< prx_T > >
{
    //  The ABSORBING inner boundary:  a pure zero -
    //  gradient copy of the boundary cell --  gas falling
    //  through r_in leaves the simulation ( the star eats
    //  it ).   The causality concern of a plain outflow BC
    //  ( Bai 2017 )  is carried by the relaxation ring
    //  behind the boundary.   The field side keeps the
    //  ppd_in scalings.
    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
            for( int n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
        return;
    };

    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int  n_gh = bc.n_spa / 2;
        const auto & geo( b_d.geo );
        const int  c0  = 0;
        const int  f0  = 0;
        const auto r_f0 = geo.x_fc( 0, f0 );
        const auto r_c0 = geo.x_cc( 0, c0 );
        for( int i = -n_gh; i < 0; ++ i )
        {
            const auto rr = geo.x_fc( 0, f0 + i ) / r_f0;
            const auto rc = geo.x_cc( 0, c0 + i ) / r_c0;
            bf( i, 0 ) = bf( 0, 0 ) / ( rr * rr );
            bf( i, 1 ) = bf( 0, 1 );
            bf( i, 2 ) = bf( 0, 2 ) / rc;
            bc( i, 0 ) = bc( 0, 0 ) / ( rc * rc );
            bc( i, 1 ) = bc( 0, 1 );
            bc( i, 2 ) = bc( 0, 2 ) / rc;
        }
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct ppd_out_t : phys_base_of< prx_T >
    ::template base_t< ppd_out_t< prx_T > >
{
    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        const auto & geo( b_d.geo );
        const int   n_r = geo.n_ceff[ 0 ];
        // Inflow veto on the active row ( outflow_t rule )
        if( d( idx::mom ) < 0 )
            d( idx::mom ) = 0;
        const auto r_c0 = geo.x_cc( 0, n_r - 1 );
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            //  The strip carries PRIMITIVES ( phys.h fills
            //  it from hyd.w ):  0 rho,  1 p,  2..4 v,
            //  5..7 bcc,  8+ species MASS FRACTIONS.   So
            //  only rho and p take the r^-2 wind falloff
            //  ( T zero - gradient );  v and the fractions
            //  are copied --  scaling the fractions would
            //  break  sum y = 1  and poison the chemistry
            //  EOS everywhere.
            //  Strip index i < 0 is the ghost OUTSIDE:
            //  cell n_r - 1 - i  ( phys.h's i_tgt ).   The
            //  old n_r - 1 + i pointed INWARD,  rc < 1,  so
            //  1/rc^2 AMPLIFIED the ghost density.
            const auto rc
                = geo.x_cc( 0, n_r - 1 - i ) / r_c0;
            const auto f_s( 1 / ( rc * rc ) );
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
            d( i, idx::rho ) = d( idx::rho ) * f_s;
            d( i, idx::ene ) = d( idx::ene ) * f_s;
        }
        return;
    };

    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void field
    ( d1d_T & bc, d1d_T & bf, const int i_3d[  ],
      bdt_T & b_d ) const
    {
        const int  n_gh = bc.n_spa / 2;
        const auto & geo( b_d.geo );
        const int   n_r = geo.n_ceff[ 0 ];
        const int   c0  = n_r - 1;
        const int   f0  = n_r;
        const auto r_f0 = geo.x_fc( 0, f0 );
        const auto r_c0 = geo.x_cc( 0, c0 );
        for( int i = -n_gh; i < 0; ++ i )
        {
            //  Ghost OUTSIDE:  normal face n_r - i,  cell
            //  n_r - 1 - i  ( see hydro;  the old  f0 + i /
            //  c0 + i  pointed INWARD ).
            const auto rf = geo.x_fc( 0, f0 - i ) / r_f0;
            const auto rc = geo.x_cc( 0, c0 - i ) / r_c0;
            //  Steady radial wind flux freezing  ( rho ~
            //  r^-2,  v zero - gradient ):   B_r ~ r^-2
            //  keeps the radial flux through spheres exact
            //  ( the ghost cell's radial div term then
            //  vanishes identically ),  and  B_t / rho ~ r
            //  =>  B_t ~ r^-1  for BOTH tangential
            //  components  ( the transverse stretching ).
            //  A conducting wall  ( E_t = 0 )  is NOT
            //  admissible at r_out:  once the wind is
            //  launched the flux there is whatever the
            //  interior blows out,  so the boundary plane
            //  must stay advective  ( geo.bnd_out drives
            //  calc_field's wall EMF --  ppd_keeper_t ).
            bf( i, 0 ) = bf( 0, 0 ) / ( rf * rf );
            bf( i, 1 ) = bf( 0, 1 ) / rc;
            bf( i, 2 ) = bf( 0, 2 ) / rc;
            bc( i, 0 ) = bc( 0, 0 ) / ( rc * rc );
            bc( i, 1 ) = bc( 0, 1 ) / rc;
            bc( i, 2 ) = bc( 0, 2 ) / rc;
        }
        return;
    };
};

template< class prx_T = dynamics::mhd::proxy_t >
struct ppd_fre_t : phys_base_of< prx_T >
    ::template base_t< ppd_fre_t< prx_T > >
{
    ////////// Data //////////
    //  y~_n / rho of the IC,  i.e.  m_A / mu_ave * x_n
    //  ( fixed - size for the device copy ).
    type::float2_t y_comp[ 64 ];
    int            n_sp;

    ////////// Functions //////////
    __host__ ppd_fre_t(  ) { n_sp = 0;  for( auto & y : y_comp )
                                y = 0; };

    //  The boundary ghost copies the interior state;  for the
    //  species this feeds composition drift back inward (
    //  interior drifts ->  the copy carries it ->  inflow
    //  amplifies it ).   Pinning the ghost composition to the
    //  IC breaks the loop.
    __host__ virtual void init
    ( const input & args, mesh::mod_base_t & mod ) override
    {
        auto & m = dynamic_cast< chemistry::mhd::base_t & >
                   ( mod );
        const auto & sps = m.q_che.lock(  )->species;
        n_sp = sps.size(  );
        if( n_sp > 64 )
            throw std::runtime_error
                ( "ppd_fre:  too many species" );
        std::vector< type::float2_t > x;
        type::float2_t norm( 0 );
        for( const auto & s : sps )
        {
            x.push_back( args.get< type::float2_t >
                         ( "species_init", s.name, 1e-20 ) );
            norm += x.back(  );
        }
        type::float2_t mu( 0 );
        for( int n = 0; n < n_sp; ++ n )
            mu += x[ n ] / norm * sps[ n ].mass_cgs;
        for( int n = 0; n < n_sp; ++ n )
            y_comp[ n ] = x[ n ] / norm * phys::cgs::ma / mu;
        return;
    };

    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void hydro
    ( d1d_T & d, const int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            for( int  n = 0; n < d.n_int; ++ n )
                d( i, n ) = d( n );
            for( int n = 0; n < n_sp; ++ n )
                d( i, idx::bcc + 3 + n ) = d( i, idx::rho )
                    * ( type::float_t ) y_comp[ n ];
        }
        return;
    };

    //  Same zero - gradient field copy as free_t.
    template< class d1d_T, class bdt_T >
    __device__ __forceinline__ void  field
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

////////////////////////////////////////////////////////////
//  The MHD keeper flavour for the PPD boundaries.  The
//  boundary - plane advective edge EMF  ( calc_field's
//  r_out field - anchoring fix,  and the implicit side's
//  wall EMF )  keys on  geo.bnd_out,  which  assign_phys
//  sets only for the stock outflow_t / free_t functors.
//  ppd_out_t  ( outflow semantics,  own ghost rules )
//  is not on that list,  so the anchoring fix  NEVER
//  fired  and the zero - E wall piled  B_phi  at r_out
//  ( the t = 0.53 - 1.04 runaways,  handover sec. 14-16 ).
//  This keeper re-declares the outer radial side as
//  outflow  ( the pgen's outer side is  ppd_out  by
//  construction;  no other functor sits there ).
struct ppd_keeper_t
    : dynamics::mhd::boundary::keeper_t
{
    __host__ virtual void assign_phys
    ( mesh::block_t & b, mesh::mesh_t & mesh ) override
    {
        keeper_t::assign_phys( b, mesh );
        auto & mod( * p_mod );
        auto & geo
            ( mod.p_data( b.reg )->d(  )
                   .derive< dynamics::mhd::
                            block_data_t >(  ).geo );
        if( geo.phybnd_flag[ 0 ][ 1 ] )
            geo.bnd_out[ 0 ][ 1 ] = true;
        return;
    };
};

};                              // namespace interm
