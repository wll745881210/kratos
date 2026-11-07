#pragma once

#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  WBG19 disk profile ( usr_ext/ppd/design.md sec. 2.1 ).
//
//  Spherical x = ( r, th, phi );  R = r sin th,  z = r cos th.
//  cs2( R ) = cs2_0 ( R / r0 )^t_pow  ( p = cs2 * rho ),
//  rho = n0 ( R / r0 )^rho_pow
//        exp( gm / cs2 ( 1 / r - 1 / R ) ),
//  v_phi^2 = gm / R * [ ( rho_pow + t_pow ) cs2 R / gm
//                       + 1 + t_pow - t_pow R / r ].
//  Below the density floor:  rho = floor,  but v_phi keeps
//  the profile value ( the WBG19 v_phi = 0 rule made the
//  whole off - midplane volume free - fall and rain onto the
//  disk ).   All members code units.
//
//  The SUPPORTED atmosphere ( [profile] atm_on = 1 ):  the
//  flat floor leaves the off - midplane volume without
//  pressure support --  it free - falls,  churns,  and ( with
//  chemistry on ) scrambles the trace - species composition
//  ( ledger cont. 37-38 ).   Here the sound speed transitions
//  to atm_A x the disk value across  |z| / H in
//  [ atm_z1, atm_z2 ] ( smoothstep ),  the density follows
//  from the EXACT vertical hydrostatic ODE integrated per
//  column,  and v_phi from the radial force balance with the
//  resulting p ( its R - derivative by finite difference ).
//  The disk midplane is unchanged ( the ODE reproduces the
//  gaussian where cs2 = const ).

struct disk_profile_t
{
    ////////// Types //////////
    using float_t  = type::  float_t;
    using float2_t = type::float2_t;

    ////////// Data //////////
    float2_t       n0;  // midplane n_H at r0
    float2_t  rho_pow;  // -2.25
    float2_t  cs2_0;    // p / rho at r0 ( vel0^2 units )
    float2_t    t_pow;  // -0.5
    float2_t       gm;  // G * M_star
    float2_t       r0;  // 1 au
    float2_t   floor;   // IC density floor
    //  The supported atmosphere ( see the header note )
    int       atm_on;
    float2_t  atm_A;    // corona temperature boost
    float2_t  atm_z1;   // transition start,  |z| / H
    float2_t  atm_z2;   // transition end,    |z| / H
    int       atm_nz;   // quadrature resolution per column
    //  The polar cap  ( sin th < pole_cap ):  the WBG19
    //  atmosphere solution degenerates on the axis  ( its
    //  column runs over ~ 1 / H  scale heights with
    //  H -> 0,  so  p  falls super - exponentially and the
    //  balancing  v_phi ~ sqrt( R dp / dR / rho )  diverges ).
    //  No grid resolves that:  inside the cap the profile
    //  is therefore evaluated at the CLAMPED  R = pole_cap r
    //  -- the cap value of the same  z  column --  which is
    //  constant in theta ( no theta pressure gradient,  no
    //  shear )  and continuous at the cap edge.  0 = off.
    float2_t  pole_cap;

    ////////// Functions //////////
    __host__ disk_profile_t(  )
    {
        n0      = 2.375e14;
        rho_pow = -2.25;
        cs2_0   = 0;
        t_pow   = -0.5;
        gm      = 0;
        r0      = 1;
        floor   = 1e7;
        atm_on  = 0;
        atm_A   = 100;
        atm_z1  = 2.5;
        atm_z2  = 3.5;
        atm_nz  = 1024;
        pole_cap = 0;
        return;
    };

    __host__ void read( const input & args )
    {
        phys::unit_t< float2_t > unit;
        unit.init( args );
        n0      = args.get< float2_t >
                      ( "profile", "n0",      2.375e14 );
        rho_pow = args.get< float2_t >
                      ( "profile", "rho_pow",    -2.25 );
        t_pow   = args.get< float2_t >
                      ( "profile", "T_pow",       -0.5 );
        r0      = args.get< float2_t >
                      ( "profile", "r0",         1.0 );
        floor   = args.get< float2_t >
                      ( "profile", "rho_floor",  1e7 );
        const auto T0    = args.get< float2_t >
                      ( "profile", "T0",        280. );
        const auto mu_ma = args.get< float2_t >
                      ( "profile", "mu_ma",      2.34 );
        // p / rho = kb T / ( mu m_A ),  in vel0^2 units
        cs2_0 = phys::cgs::kb * T0 / ( mu_ma
              * phys::cgs::mp ) / ( unit.vel0
                                  * unit.vel0 );
        const auto mstar = args.get< float2_t >
                      ( "prob", "mstar_cgs", 2e33 );
        gm = phys::cgs::G * mstar / unit.derive( 0, 3, -2 );
        atm_on = args.get< int >( "profile", "atm_on", 0 );
        atm_A  = args.get< float2_t >
                     ( "profile", "atm_A",  100.0 );
        atm_z1 = args.get< float2_t >
                     ( "profile", "atm_z1", 2.5 );
        atm_z2 = args.get< float2_t >
                     ( "profile", "atm_z2", 3.5 );
        atm_nz = args.get< int >
                     ( "profile", "atm_nz", 1024 );
        pole_cap = args.get< float2_t >
                     ( "profile", "pole_cap", 0 );
        return;
    };

    //  The sound - speed ramp and its derivative,
    //  f( |z| / H ) = 1 + ( A - 1 ) S( ( eta - z1 )
    //  / ( z2 - z1 ) )  with S the smoothstep.
    __host__ __device__ __forceinline__ float2_t atm_f
    ( const float2_t & eta ) const
    {
        const float2_t t = ( eta - atm_z1 ) / ( atm_z2 - atm_z1 );
        if( t <= 0 )  return 1;
        if( t >= 1 )  return atm_A;
        return 1 + ( atm_A - 1 ) * t * t * ( 3 - 2 * t );
    };
    __host__ __device__ __forceinline__ float2_t atm_fp
    ( const float2_t & eta ) const
    {
        const float2_t t = ( eta - atm_z1 ) / ( atm_z2 - atm_z1 );
        if( t <= 0 || t >= 1 )  return 0;
        return ( atm_A - 1 ) * 6 * t * ( 1 - t )
             / ( atm_z2 - atm_z1 );
    };

    //  Column pressure at ( R, |z| ) under the exact vertical
    //  hydrostatic ODE:  d ln rho / dz = - gm z / r^3 / cs2
    //  - d ln cs2 / dz,  cs2 = cs2_d * f( |z| / H ) --  a pure
    //  quadrature ( the rhs does not depend on rho ).
    __host__ __device__ __forceinline__ float2_t col_pres
    ( const float2_t & R, const float2_t & az ) const
    {
        const float2_t cs2_d = cs2_0 * pow( R / r0, t_pow );
        const float2_t rho_m = n0 * pow( R / r0, rho_pow );
        const float2_t H     = sqrt( cs2_d * R * R * R / gm );
        float2_t xi = log( rho_m );
        const float2_t dz = az / atm_nz;
        for( int k = 0; k < atm_nz; ++ k )
        {
            const float2_t za = dz * k, zb = dz * ( k + 1 );
            float2_t g_sum = 0;
            for( int e = 0; e < 2; ++ e )
            {
                const float2_t zz = e ? zb : za;
                const float2_t rr = sqrt( R * R + zz * zz );
                const float2_t cs2 = cs2_d * atm_f( zz / H );
                const float2_t dcs = cs2_d * atm_fp( zz / H ) / H;
                g_sum += - gm * zz / ( rr * rr * rr ) / cs2
                       - dcs / cs2;
            }
            xi += dz * g_sum / 2;
        }
        const float2_t cs2 = cs2_d * atm_f( az / H );
        return exp( xi ) * cs2;
    };

    //  Regularity at the axis.   A regular vector field
    //  has BOTH tangential components vanishing on the
    //  axis  ( v_phi ~ R near it ),  while the Keplerian
    //  profile gives  v_phi ~ R^{-1/2}  and DIVERGES:
    //  with theta spanning 0..pi the polar row then
    //  carries  v_phi ~ 1e13  and the CFL collapses to
    //  dt ~ 1e-16  ( the full - polar PPD run cannot
    //  start ).   Ramp the rotation to zero inside
    //  R_c = 1e-2 r0  ( v_phi -> v_phi R / R_c ),  which
    //  leaves the disk body untouched and makes the
    //  funnel non - rotating,  as the hydrostatic
    //  atmosphere is.   A non - finite value is dropped
    //  to zero  ( the same statement taken literally ).
    template< class f_T >
    __host__ __device__ __forceinline__ f_T
    v_phi_axis( const f_T & v, const f_T & R ) const
    {
        const f_T rc( 1e-2 * r0 );
        f_T o( R < rc ? v * R / rc : v );
        //  Regularity at the axis.  A smooth rotation has
        //  v_phi = Omega R  ( R = r sin th ),  and the
        //  centrifugal theta - source  v_phi^2 cot( th ) / r
        //  ( src_geo,  integrator/sph.h ) then scales as
        //  sin( th ) and dies on the axis.  The Keplerian
        //  R^{-1/2} instead makes that source diverge as
        //  1 / sin( th ):  with  cot( th ) ~ 1e2  on the
        //  polar row the full - polar run gains
        //  v_theta ~ 9 in the FIRST step and the pressure
        //  collapses a few steps later.  Cap at the r0
        //  Keplerian angular speed -- inert outside the
        //  funnel ( R > r0 ) and on the midplane.
        const f_T om( sqrt( gm / ( r0 * r0 * r0 ) ) );
        if( fabs( o ) > om * R )
            o = o > 0 ? om * R : - om * R;
        //  Absolute sanity bound:  nothing in this disk
        //  rotates faster than a few Keplerian speeds at
        //  r0.   The atmosphere quadrature returns a
        //  non - finite  d p / d R  in the axis column
        //  ( H -> 0 there ),  so a profile value can come
        //  out absurd without being NaN.   Such a value is
        //  NOT physical:  the regular state there is the
        //  non - rotating funnel,  so drop it to zero  (
        //  clamping to the bound instead leaves the polar
        //  row spinning at  3 v_K  next to a static
        //  neighbour -- a supersonic shear ).
        const f_T v_max( f_T( 3 ) * sqrt( gm / r0 ) );
        if( ! ( o == o ) || fabs( o ) > v_max )
            o = 0;
        return o;
    };

    template< class coord_T, class f_T >
    __host__ __device__ __forceinline__ void hydro
    ( const coord_T & x, f_T & rho, f_T & pre,
      f_T & v_phi )                                   const
    {
        const float2_t r = x[ 0 ];
        const float2_t st = fabs( f_T( sin( x[ 1 ] ) ) );
        const float2_t R = pole_cap > 0 && st < pole_cap
                         ? pole_cap * r : r * st;
        const float2_t z = fabs( r * cos( x[ 1 ] ) );
        const float2_t cs2
            = cs2_0 * pow( R / r0, t_pow );
        //  The vertical atmosphere quadrature is NOT valid
        //  in the axis column:  its vertical extent z/H
        //  blows up as R -> 0, so  col_pres  returns a
        //  non - finite value and the whole polar cell
        //  becomes NaN ( measured:  p = NaN at theta = 0,
        //  dt -> 1e-16 in the full - polar PPD IC ).  Below
        //  R_pole the plain profile is used instead,  with
        //  the rotation already ramped to zero.
        const float2_t R_pole( 1e-3 * r0 );
        if( atm_on && R >= R_pole )
        {
            //  The supported atmosphere:  p from the column
            //  quadrature,  rho = p / cs2( R, z ),  v_phi from
            //  the radial force balance ( finite difference of
            //  p in R ).
            const float2_t Rc = R > 1e-4 * r0 ? R
                                              : 1e-4 * r0;
            const float2_t pC = col_pres( Rc, z );
            const float2_t hR = 0.01 * Rc;
            const float2_t pL = col_pres( Rc - hR, z );
            const float2_t pR = col_pres( Rc + hR, z );
            const float2_t cs2z = cs2 * atm_f( z /
                sqrt( cs2 * Rc * Rc * Rc / gm ) );
            rho = pC / cs2z;
            pre = pC;
            const float2_t dpdr = ( pR - pL ) / ( 2 * hR );
            const float2_t v2 = Rc * ( gm * Rc / ( r * r * r )
                                     + dpdr / rho );
            v_phi = v2 > 0 ? sqrt( v2 ) : 0;
            v_phi = v_phi_axis( v_phi, f_T( R ) );
            //  The polar columns underflow  atm_f( z / H )
            //  to zero  ( H -> 0 as R -> 0 ),  so  pC  and
            //  cs2z  are both 0 and  rho = pC / cs2z  is
            //  0 / 0 = NaN --  the full - polar IC then has
            //  p = NaN and dt ~ 1e-16.  NaN - aware floor.
            if( ! ( rho == rho ) || rho < floor )
                rho = floor;
            if( ! ( pre == pre ) || pre < floor )
                pre = floor * cs2z;
            return;
        }
        const float2_t eidx
            = gm / cs2 * ( 1. / r - 1. / R );
        const float2_t rho_d
            = n0 * pow( R / r0, rho_pow ) * exp( eidx );
        const float2_t v2r
            =   ( rho_pow + t_pow ) * cs2 * R / gm
              + ( 1. + t_pow ) - t_pow * R / r;
        v_phi = v2r > 0 ? sqrt( v2r * gm / R ) : 0;
        v_phi = v_phi_axis( v_phi, f_T( R ) );
        rho   = rho_d > floor ? rho_d : floor;
        pre   = cs2 * rho;
        return;
    };
};

};                              // namespace interm
