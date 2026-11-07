#pragma once

#include "../../src/modules/dynamics/mhd/integrator/sph.h"
#include "../../src/modules/dynamics/hydro/integrator/sph.h"
#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"
#include "disk_profile.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  Zero the edge EMFs adjoining the suppression rim so the
//  rim 's magnetic field stays EXACTLY at its initial
//  condition:  the frozen rim cells keep their Keplerian
//  v_phi,  which would otherwise wind the rim field up
//  ( hydro frozen -> no advection;  eta = 0 -> no
//  diffusion;  a one - way B_phi pile - up that shrinks dt
//  and is unphysical ).   The CT face update is the curl
//  of the edge EMFs,  so replacing the EMF by one that
//  vanishes on a subset of edges still conserves div( B )
//  to machine precision.   Edges with i[ 0 ] <= supp_nr
//  ( or the theta analogues ) bound the rim cells ' faces
//  INCLUDING the rim / active interface:  the rim is an
//  insulating wall for B as well.

template< class bdt_T > __global__
void disk_emf0_k( const bdt_T b_d, const int supp_nr,
                  const int supp_nt )
{
    const auto & geo ( b_d.geo );
    type::idx_t i = { int( blockIdx.x ), int( blockIdx.y ),
                      0 };
    if( ! ( ( supp_nr > 0 && i[ 0 ] <= supp_nr )
            || ( supp_nt > 0
                 && ( i[ 1 ] <= supp_nt
                      || i[ 1 ] >= geo.n_ceff[ 1 ]
                                   - supp_nt ) ) ) )
        return;
    for( int i2 = threadIdx.x; i2 <= geo.n_ceff[ 2 ];
         i2 += blockDim.x )
    {
        i[ 2 ]    = i2;
        auto * ee = b_d.fld.ee.at( i );
        ee[ 0 ] = ee[ 1 ] = ee[ 2 ] = 0;
    }
    return;
}

////////////////////////////////////////////////////////////
//  The relaxation ring ( [ring] ) --  the continuous
//  version of Bai 's inner - rim reset / the de Val - Borro
//  wave - killing zone:  inside r < ring_r the hydro state
//  relaxes toward the IC profile on the timescale
//  ring_tau / f( r ),  with f = ( ( ring_r - r )
//  / ( ring_r - r_in ) )^2 ramping smoothly from 0 at
//  ring_r to 1 at r_in --  NO hard interface ( the frozen
//  rim 's corner was the dt limiter ).
//  This kernel relaxes the FACE B_phi toward zero:  a local
//  scaling of the phi component can not seed monopoles (
//  it damps the phi part of any existing div error ),  so
//  it is div - safe.   The poloidal field is NOT touched
//  here --  it is handled by the enhanced ring resistivity
//  in disk_eta_t ( CT diffusion,  exactly div - free ).

template< class bdt_T > __global__
void disk_bphi_rlx_k( const bdt_T b_d, const int slice,
                      const type::float_t ring_r,
                      const type::float_t ring_tau )
{
    const auto & geo ( b_d.geo );
    type::idx_t i = { int( blockIdx.x ), int( blockIdx.y ),
                      0 };
    const auto r_in = geo.x_fc( 0, 0 );
    const auto s    = ( ring_r - geo.x_fc( 0, i[ 0 ] ) )
                    / ( ring_r - r_in );
    if( s <= 0 )
        return;
    auto eps = b_d.dt * ( s > 1 ? 1 : s * s ) / ring_tau;
    eps = eps > 1 ? 1 : eps;
    for( int i2 = threadIdx.x; i2 <= geo.n_ceff[ 2 ];
         i2 += blockDim.x )
    {
        i[ 2 ] = i2;
        b_d.fld.bf.at( i, slice )[ 2 ] *= 1 - eps;
    }
    return;
}

////////////////////////////////////////////////////////////
//  Spherical MHD integrator + point-mass stellar gravity
//  ( the usr_ext / planet_atm_sph / intg.h pattern, MHD
//  flavor, inertial frame ).   Gravity enters add_flux as
//  u[ mom ] += dt rho acc with the energy from the kinetic
//  - energy difference ( ek1 - ek0 ).
//
//  The CK15 ( Chandrashekar & Klingenberg 2015 ) well -
//  balanced pair is the BENCHMARK scheme of this pgen
//  ( [prob] ck15,  default 3 = theta + radial ):  src_geo
//  replaces the pointwise centrifugal / gravity terms with the
//  face - exponential forms matched to the equilibrium -
//  variable pressure reconstruction in plm_ck15.h.

struct stp_t : dynamics::mhd::integrator::sph_t< stp_t >
{
    ////////// Data and types //////////
    using super_t
        = dynamics::mhd::integrator::sph_t< stp_t >;
    using coord_t = type::      coord_t;
    using float_t = type::      float_t;

    bool       enabled;
    float_t       gm;
    int        dbg_i[ 3 ];   // [prob] dbg_i{0,1,2}:  cell
                             // for the source - decomposition
                             // print ( -1 = off )
    //  [prob] dbg_slot:  extra conserved - slot budget in the
    //  DBG print ( the species channels;  -1 = off ).
    int      dbg_slot;
    float_t        gam;   //  [dynamics] gamma ( for p from u )
    //  [prob] ck15:  bitmask of the CK15 ( Chandrashekar &
    //  Klingenberg 2015 ) well - balanced pair -- 1 = theta,
    //  2 = radial.  The reconstruction side lives in
    //  interm::plm_ck15_t ( usr / intermediate / plm_ck15.h,
    //  enrolled by usr.cpp whenever ck15 != 0 );  THIS side
    //  replaces the pointwise centrifugal / gravity sources in
    //  src_geo with the flux - matched exponential forms.  The
    //  two sides are consistent ONLY together.  Requires
    //  omega_0 = 0.
    int        ck15;
    //  The density floor lives HERE ( the integrator 's
    //  update point ),  composition - preserving:  the species
    //  slots scale with rho so the added mass carries the
    //  local composition.   The eos d_floor = 1e-21 stays as
    //  the primitives ' last - resort positivity.
    float_t    rho_fl;
    float_t    fl_r0;
    float_t    fl_idx;
    //  The suppression rim ( [suppress],  mirrored in
    //  disk_chem_sat.h ):  cells within supp_nr of r_in or
    //  supp_nt of either theta edge keep their conserved
    //  hydro state FROZEN ( rho / mom / ene / species not
    //  written back;  the bcc slots ride the CT update so
    //  the EM field still integrates ).   These cells are
    //  baked by the non - physical boundary - adjacent
    //  radiation;  freezing them protects the pre -
    //  absorption and the wind base.
    int        supp_nr;
    int        supp_nt;
    //  [suppress] emf0 ( default 1 ):  zero the rim -
    //  adjacent edge EMFs before the CT update so the rim
    //  field stays frozen at the IC ( disk_emf0_k ).  The
    //  0 switch exists for div - B forensics.
    int        emf0;
    //  The relaxation ring ( [ring];  see disk_bphi_rlx_k
    //  for the design ):  cells with r < ring_r relax
    //  toward the IC profile on ring_tau / f( r ).  The
    //  poloidal - field side is the enhanced ring
    //  resistivity in disk_eta_t ( [ring] eta_max ).
    //  ring_r <= 0 disables.
    float_t    ring_r;
    float_t    ring_tau;
    //  [ring] mode:  0 = relax the FULL state toward the IC
    //  profile ( owns rho / p too --  pair it with the ring
    //  chem - skip,  automatic in disk_chem_sat.h );
    //  1 = Bai 's buffer style:  relax ONLY the velocities
    //  ( v_r, v_th -> 0;  v_phi -> the profile value ),
    //  leaving rho / p / composition to the physics --  no
    //  mass / energy injection,  no ring - exterior mismatch.
    int        ring_mode;
    int        ring_bphi;
    //  [ring] n_th_band:  theta - edge band width in cells;
    //  band cells take the FULL - state relaxation at full
    //  strength regardless of ring_mode / radius.
    int        ring_n_th;
    //  [ring] va_cap:  v_A cap via a local density floor (
    //  0 = off );  va_cap_nth:  theta - edge cap band width
    //  in cells at ALL radii ( the funnel cone );
    //  va_cap_ring = 1 adds the ring interior.
    //  va_cap_out:  the OUTER annulus inner edge ( 0 =
    //  off ):  x[ 0 ] > va_cap_out joins the cap zone --
    //  the r_out rim - cell B_phi pileup antidote  ( the
    //  bnd_out anchoring alone leaves the last cell winding up ).
    float_t    ring_vac;
    int        ring_vac_ring;
    int        ring_vac_nth;
    float_t    ring_vac_out;
    //  [prob] T_cap_cgs:  GLOBAL temperature ceiling,
    //  applied at the integrator level ( NOT the chem
    //  solver 's Tmax --  the sat - skipped funnel cells
    //  never see that one ):  the funnel gas is
    //  adiabatically heated by compression with no cooling
    //  outlet ( chem - skipped ),  and 5e5 K there caps dt.
    //  ~ 3e4 K is the photoionization - equilibrium scale.
    //  0 = off.
    float_t    tcap_cs2;
    //  The cylindrical - R poloidal - momentum damper ( the
    //  aggressive companion of [wedge] 's eta ):  inside
    //  R = r sin( th ) <
    //  wedge_R the poloidal momenta relax to 0 on
    //  wedge_tau,  with the energy following the kinetic
    //  difference exactly ( the mode - 1 ring bookkeeping ),
    //  so the damper neither injects nor removes thermal
    //  energy.   Prevents the explosive inflow / outflow the
    //  inner - region pressure / magnetic push would
    //  otherwise develop.   wedge_R <= 0 disables.   The
    //  R - ramp is the same smooth ( ( wedge_R - R ) /
    //  ( wedge_R - R_in ) )^2 as the eta wedge.
    float_t    wedg_R;
    float_t    wedg_tau;
    //  [wedge] v_max:  a POLOIDAL VELOCITY CAP inside the same R < wedg_R
    //  cone -- the "cap" companion of the zeroing damper below.  It removes
    //  only the EXCESS of the poloidal speed above v_max, with the removed
    //  kinetic energy taken out of ene ( exact bookkeeping, so no thermal
    //  injection ), v_phi untouched;  the removal is ramped by the same f_w
    //  so the cap fades out at the cone edge instead of building a wall.
    //  0 = off ( default is bit-identical ).
    float_t    wedg_vmax;
    disk_profile_t prof;

    __host__ stp_t(  )
    {
        enabled = true;
        ck15    = 3;
        rho_fl  = 0;
        fl_r0   = 1;
        fl_idx  = -3;
        supp_nr = 0;
        supp_nt = 0;
        emf0    = 1;
        ring_r   = 0;
        ring_tau = 1.0e-2;
        ring_mode = 0;
        ring_bphi = 0;
        ring_n_th = 0;
        ring_vac  = 0;
        ring_vac_ring = 0;
        ring_vac_nth  = 0;
        ring_vac_out  = 0;
        tcap_cs2 = 0;
        wedg_R   = 0;
        wedg_tau = 1.0e-2;
        wedg_vmax = 0;
        gam     = 1.6666666666666667;
        gm      = 0;
        dbg_slot = -1;
        dbg_i[ 0 ] = dbg_i[ 1 ] = dbg_i[ 2 ] = -1;
        return;
    };

    __host__ virtual void init
    ( const input & args,
      const::dynamics::base_t & mod ) override
    {
        super_t::init( args, mod );
        // The EOS unit is not yet initialized at integrator
        // - init time;  build a local one from [ unit ].
        phys::unit_t< type::float2_t > unit;
        unit.init( args );
        const auto mstar = args.get< float_t >
            ( "prob", "mstar_cgs", 2e33 );
        gm = phys::cgs::G * mstar / unit.derive( 0, 3, -2 );
        enabled = args.get< bool >
                ( "dynamics", "enabled", true );
        gam     = args.get< float_t >
                ( "dynamics", "gamma", 1.6666666666666667 );
        ck15    = args.get< int >
                ( "prob", "ck15", 3 );
        rho_fl  = args.get< float_t >
                ( "buffer", "rho_floor_dyn", 0 );
        fl_r0   = args.get< float_t >
                ( "buffer", "rho_floor_r0",  1.0 );
        fl_idx  = args.get< float_t >
                ( "buffer", "rho_floor_idx", -3.0 );
        supp_nr = args.get< int >( "suppress", "n_r_in", 0 );
        supp_nt = args.get< int >( "suppress", "n_th",   0 );
        emf0    = args.get< int >( "suppress", "emf0",   1 );
        ring_r   = args.get< float_t >
                         ( "ring", "r_out", 0 );
        ring_tau = args.get< float_t >
                         ( "ring", "tau",  1.0e-2 );
        ring_bphi = args.get< int >
                         ( "ring", "bphi_rlx", 0 );
        ring_mode = args.get< int >
                         ( "ring", "mode", 0 );
        ring_n_th = args.get< int >
                         ( "ring", "n_th_band", 0 );
        ring_vac  = args.get< float_t >
                         ( "ring", "va_cap", 0 );
        ring_vac_ring = args.get< int >
                         ( "ring", "va_cap_ring", 0 );
        ring_vac_out  = args.get< float_t >
                         ( "ring", "va_cap_out", 0 );
        wedg_R   = args.get< float_t >
                         ( "wedge", "R_damp", 0 );
        wedg_tau = args.get< float_t >
                         ( "wedge", "mom_tau", 1.0e-2 );
        wedg_vmax = args.get< float_t >
                         ( "wedge", "v_max", 0 );
        ring_vac_nth  = args.get< int >
                         ( "ring", "va_cap_nth", 0 );
        {
            const float_t T_cap = args.get< float_t >
                ( "prob", "T_cap_cgs", 0 );
            if( T_cap > 0 )
            {
                const auto mu_ma = args.get< float_t >
                    ( "profile", "mu_ma", 2.34 );
                tcap_cs2 = phys::cgs::kb * T_cap
                    / ( mu_ma * phys::cgs::ma )
                    / ( unit.vel0 * unit.vel0 );
            }
        }
        if( ring_r > 0 || ring_n_th > 0 )
            prof.read( args );
        dbg_slot = args.get< int >
                ( "prob", "dbg_slot", -1 );
        dbg_i[ 0 ] = args.get< int >( "prob", "dbg_i0", -1 );
        dbg_i[ 1 ] = args.get< int >( "prob", "dbg_i1", -1 );
        dbg_i[ 2 ] = args.get< int >( "prob", "dbg_i2", -1 );
        return;
    };

    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void  add_flux
    ( dat_T & d, const bdt_T & b_d,
      const eos_T & eos )                      const
    {
        const auto & geo = b_d.geo;
        const auto & self = get_self(  );

        int i[ 3 ], i_1d_c, i_1d_p;
        self.dispatch( i, i_1d_c, i_1d_p, d, b_d );

        using  f_t = typename decltype( b_d.hyd )::f_t;
        dynamics::hydro::integrator::flux_info_t< f_t > f;

        auto * u     = d.u.at( i_1d_c );
        auto * u_tgt = self.load( u, b_d.hyd.u, i, d.step );
        const auto & n_var = b_d.hyd.u.n_int;

        //  The suppression rim:  hydro frozen,  only the
        //  bcc slots write back ( the CT update runs
        //  elsewhere;  here u == the old state,  so the
        //  writeback is the identity ).
        if( ( supp_nr > 0 || supp_nt > 0 )
            && ( i[ 0 ] < supp_nr || i[ 1 ] < supp_nt
                 || i[ 1 ] >= geo.n_ceff[ 1 ] - supp_nt ) )
        {
            //  rk2 slot bookkeeping:  load( ) reads slot
            //  `step` and targets slot `1 - step`;  freezing
            //  must copy the READ slot into the TARGET slot
            //  ( writing only part of the slots leaves the
            //  rest stale --  at step 1 the stale slots mix
            //  into ( src + tgt ) / 2 and poison everything
            //  downstream ).
            const auto * u_src = b_d.hyd.u.at( i, d.step );
            for( int n = 0; n < n_var; ++ n )
                u_tgt[ n ] = u_src[ n ];
            return;
        }

        coord_t x;
        geo.x_cc( x, i );
        const bool dbg = ( dbg_i[ 0 ] >= 0
            && i[ 0 ] == dbg_i[ 0 ] && i[ 1 ] == dbg_i[ 1 ]
            && i[ 2 ] == dbg_i[ 2 ] );
        f_t du_flx[ 3 ] = { f_t( 0 ), f_t( 0 ), f_t( 0 ) };
        f_t du_flx_r[ 3 ] = { f_t( 0 ), f_t( 0 ), f_t( 0 ) };
        f_t du_rho[ 3 ] = { f_t( 0 ), f_t( 0 ), f_t( 0 ) };
        f_t du_ene[ 3 ] = { f_t( 0 ), f_t( 0 ), f_t( 0 ) };
        f_t du_spe[ 3 ] = { f_t( 0 ), f_t( 0 ), f_t( 0 ) };
        f_t du_src( 0 ), du_src_r( 0 ), du_grav( 0 );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            const f_t m0 = u[ idx::mom + 1 ];
            const f_t m0r = u[ idx::mom ];
            const f_t r0 = u[ idx::rho ];
            const f_t e0 = u[ idx::ene ];
            const f_t s0 = dbg_slot >= 0 ? f_t( u[ dbg_slot ] )
                                         : f_t( 0 );
            self.load_flux_sfc( f, d, b_d, i, i_1d_p, a );
            if( dbg && dbg_slot >= 0 && a == d.axis )
                printf( "DBGFLX: ax %d step %d  rho: %e %e"
                        "  spe: %e %e   ql/qr spe: %e %e"
                        " | %e %e   rho: %e %e | %e %e\n",
                        a, d.step,
                        double( f.flux[ 0 ][ idx::rho ] ),
                        double( f.flux[ 1 ][ idx::rho ] ),
                        double( f.flux[ 0 ][ dbg_slot ] ),
                        double( f.flux[ 1 ][ dbg_slot ] ),
                        double( d.ql.at( i_1d_p     )
                                [ dbg_slot ] ),
                        double( d.qr.at( i_1d_p     )
                                [ dbg_slot ] ),
                        double( d.ql.at( i_1d_p + 1 )
                                [ dbg_slot ] ),
                        double( d.qr.at( i_1d_p + 1 )
                                [ dbg_slot ] ),
                        double( d.ql.at( i_1d_p     )
                                [ idx::rho ] ),
                        double( d.qr.at( i_1d_p     )
                                [ idx::rho ] ),
                        double( d.ql.at( i_1d_p + 1 )
                                [ idx::rho ] ),
                        double( d.qr.at( i_1d_p + 1 )
                                [ idx::rho ] ) );
            if( dbg && dbg_slot >= 0 && a == d.axis )
            {
                printf( "DBGQ: step %d  q spe: %e %e %e %e"
                        "  w spe: %e\n", d.step,
                        double( d.q( i_1d_p - 1, dbg_slot ) ),
                        double( d.q( i_1d_p,     dbg_slot ) ),
                        double( d.q( i_1d_p + 1, dbg_slot ) ),
                        double( d.q( i_1d_p + 2, dbg_slot ) ),
                        double( b_d.hyd.w.at( i )
                                [ dbg_slot ] ) );
                printf( "DBGQR:  qr spe: %e %e %e %e %e\n",
                        double( d.qr.at( i_1d_p )[ 8 ] ),
                        double( d.qr.at( i_1d_p )[ 9 ] ),
                        double( d.qr.at( i_1d_p )[ 10 ] ),
                        double( d.qr.at( i_1d_p )[ 11 ] ),
                        double( d.qr.at( i_1d_p )[ 12 ] ) );
                printf( "DBGQL:  ql spe: %e %e %e %e %e\n",
                        double( d.ql.at( i_1d_p )[ 8 ] ),
                        double( d.ql.at( i_1d_p )[ 9 ] ),
                        double( d.ql.at( i_1d_p )[ 10 ] ),
                        double( d.ql.at( i_1d_p )[ 11 ] ),
                        double( d.ql.at( i_1d_p )[ 12 ] ) );
                printf( "DBGALL: qr:" );
                for( int n = 0; n < 13; ++ n )
                    printf( " %e",
                            double( d.qr.at( i_1d_p )[ n ] ) );
                printf( "\nDBGALL: ql:" );
                for( int n = 0; n < 13; ++ n )
                    printf( " %e",
                            double( d.ql.at( i_1d_p )[ n ] ) );
                printf( "\n" );
            }
            for( int n = 0; n < n_var; ++ n )
                self.flux_to_du_sph
                    ( u, f, b_d, i, x, a, n );
            du_flx[ a ] = u[ idx::mom + 1 ] - m0;
            du_flx_r[ a ] = u[ idx::mom ] - m0r;
            du_rho[ a ] = u[ idx::rho ] - r0;
            du_ene[ a ] = u[ idx::ene ] - e0;
            if( dbg_slot >= 0 )
                du_spe[ a ] = f_t( u[ dbg_slot ] ) - s0;
        }
        f_t rho_src( 0 );
        const f_t * w_s = b_d.hyd.w.at( i );
        {
            const f_t m0 = u[ idx::mom + 1 ];
            const f_t m0r = u[ idx::mom ];
            rho_src = f_t( w_s[ idx::rho ] );
            self.src_geo( u, w_s, b_d, i, x );
            du_src = u[ idx::mom + 1 ] - m0;
            du_src_r = u[ idx::mom ] - m0r;
        }
        if( enabled )
        {
            // Gravity must use the SAME cell - centered
            // density -- and the same time level -- as
            // src_geo's centrifugal term.  The two terms
            // cancel to |vp^2/r - gm/r^2| / (gm/r^2) ~ 3e-3,
            // so a relative density mismatch eps shows up as
            // a net radial force eps * gm/r^2,  amplified
            // ~300x.   Using the post - flux density breaks
            // the match.
            bool grav_on = true;
            if( ck15 & 2 )
            {
                //  CK15:  gravity's momentum part rides inside
                //  the matched radial source in src_geo -- but
                //  ONLY where that source's gate fired.  The
                //  gate is deterministic in ( w_s, x ),  so
                //  rebuild it here:  gate passed -> only the
                //  WORK term remains;  gate failed -> the base
                //  pointwise gravity must still apply.
                const f_t rho_w = f_t( w_s[ idx::rho ] );
                const f_t pre_w = f_t( w_s[ idx::ene ] );
                const f_t v1w = f_t( w_s[ idx::mom + 1 ] );
                const f_t vfw = f_t( w_s[ idx::mom + 2 ] );
                const f_t r_l = geo.x_fc( 0, i[ 0 ]     );
                const f_t r_r = geo.x_fc( 0, i[ 0 ] + 1 );
                const f_t lam = ( ( v1w * v1w + vfw * vfw )
                                  / x[ 0 ]
                                - gm / ( x[ 0 ] * x[ 0 ] ) )
                              * rho_w / pre_w;
                if( pre_w > 0 && rho_w > 0
                    && __abs1( lam * ( r_r - x[ 0 ] ) ) < 2
                    && __abs1( lam * ( x[ 0 ] - r_l ) ) < 2 )
                {
                    grav_on = false;
                    const f_t rr = f_t( u[ idx::rho ] );
                    if( rr > 0 )
                        u[ idx::ene ] -= b_d.dt * gm
                            * f_t( u[ idx::mom ] )
                            / ( x[ 0 ] * x[ 0 ] );
                }
            }
            if( grav_on )
            {
                const f_t acc = -gm / ( x[ 0 ] * x[ 0 ] );
                const auto  dt_rho = b_d.dt * rho_src;
                const auto p0 = f_t( u[ idx::mom ] );
                u[ idx::mom ] += acc * dt_rho;
                const auto p1 = f_t( u[ idx::mom ] );
                du_grav = p1 - p0;
                u[ idx::ene ] += ( p1 * p1 - p0 * p0 )
                               / ( 2 * rho_src );
            }
        }
        if( dbg )
            printf( "DBG: i %d %d %d dt %e | th: flx %e %e %e"
                    " src %e | r: flx %e %e %e src %e grav %e |"
                    " rho: %e %e %e | ene: %e %e %e |"
                    " spe: %e %e %e |"
                    " step %d rho_w %e rho_u %e dw %+.6e"
                    " spe_w %e\n",
                    i[ 0 ], i[ 1 ], i[ 2 ], double( b_d.dt ),
                    double( du_flx[ 0 ] ), double( du_flx[ 1 ] ),
                    double( du_flx[ 2 ] ), double( du_src ),
                    double( du_flx_r[ 0 ] ), double( du_flx_r[ 1 ] ),
                    double( du_flx_r[ 2 ] ), double( du_src_r ),
                    double( du_grav ),
                    double( du_rho[ 0 ] ), double( du_rho[ 1 ] ),
                    double( du_rho[ 2 ] ),
                    double( du_ene[ 0 ] ), double( du_ene[ 1 ] ),
                    double( du_ene[ 2 ] ),
                    double( du_spe[ 0 ] ), double( du_spe[ 1 ] ),
                    double( du_spe[ 2 ] ),
                    d.step, double( rho_src ),
                    double( u[ idx::rho ] ),
                    double( rho_src - f_t( u[ idx::rho ] ) ),
                    dbg_slot >= 0
                        ? double( w_s[ dbg_slot ] ) : 0.0 );
        if( rho_fl > 0 )
        {
            //  Composition - preserving density floor ( the
            //  rho_fl note above ):  the species slots scale
            //  with rho so the added mass carries the local
            //  composition.
            const f_t r_now = f_t( u[ idx::rho ] );
            const f_t r_fl  = rho_fl
                * pow( x[ 0 ] / fl_r0, fl_idx );
            if( r_now < r_fl && r_now > 0 )
            {
                const f_t rat = r_fl / r_now;
                u[ idx::rho ] = r_fl;
                for( int n = idx::bcc + 3; n < n_var; ++ n )
                    u[ n ] *= rat;
            }
        }
        const bool in_band = ring_n_th > 0
            && ( i[ 1 ] < ring_n_th
                 || i[ 1 ] >= geo.n_ceff[ 1 ] - ring_n_th );
        if( ( ring_r > 0 && x[ 0 ] < ring_r ) || in_band )
        {
            //  The relaxation ring ( see disk_bphi_rlx_k ):
            //  hydro state toward the IC profile,  on
            //  ring_tau / f( r ).   The species slots scale
            //  with rho ( composition - preserving );  the
            //  cell - centered B_phi slot relaxes with the
            //  same factor ( the face field is relaxed in
            //  disk_bphi_rlx_k,  launched post - CT ).
            //  The theta - edge band ( [ring] n_th_band )
            //  always takes the FULL - state relaxation at
            //  full strength ( f = 1 ):  the band is the
            //  boundary - artifact zone ( chem - skipped via
            //  [chemistry] skip_nt --  with neither heating
            //  nor cooling it would rain onto the corner
            //  and shock - heat.
            const f_t r_in = geo.x_fc( 0, 0 );
            const f_t s    = ( ring_r - f_t( x[ 0 ] ) )
                           / ( ring_r - r_in );
            const f_t f_r  = in_band ? f_t( 1 )
                : ( s > 1 ? f_t( 1 )
                          : ( s > 0 ? s * s : f_t( 0 ) ) );
            f_t eps = b_d.dt * f_r / ring_tau;
            eps = eps > 1 ? f_t( 1 ) : eps;
            f_t rho_t, pre_t, vph_t;
            prof.hydro( x, rho_t, pre_t, vph_t );
            const f_t r_o0 = f_t( u[ idx::rho ] );
            if( ring_mode == 1 && ! in_band )
            {
                //  Bai 's buffer style:  relax ONLY the
                //  velocities;  rho / thermal / composition
                //  stay with the physics.   The energy
                //  follows the kinetic difference.
                const f_t ke0 = ( u[ idx::mom ]
                                  * u[ idx::mom ]
                    + u[ idx::mom + 1 ] * u[ idx::mom + 1 ]
                    + u[ idx::mom + 2 ] * u[ idx::mom + 2 ] )
                    * f_t( 0.5 ) / r_o0;
                const f_t mp_t = r_o0 * vph_t;
                u[ idx::mom     ] *= 1 - eps;
                u[ idx::mom + 1 ] *= 1 - eps;
                u[ idx::mom + 2 ] += eps
                    * ( mp_t - f_t( u[ idx::mom + 2 ] ) );
                const f_t ke1 = ( u[ idx::mom ]
                                  * u[ idx::mom ]
                    + u[ idx::mom + 1 ] * u[ idx::mom + 1 ]
                    + u[ idx::mom + 2 ] * u[ idx::mom + 2 ] )
                    * f_t( 0.5 ) / r_o0;
                u[ idx::ene ] += ke1 - ke0;
            }
            else
            {
            const f_t rho1 = r_o0 + eps * ( rho_t - r_o0 );
            //  NOTE:  the conserved mom slots are PLAIN
            //  momenta rho v ( the eos c2p divides by rho
            //  with NO metric factor;  the angular -
            //  momentum property lives in the sph flux
            //  rescaling,  not in the state ).   The
            //  poloidal momenta relax to 0.
            const f_t mp_t = rho1 * vph_t;
            u[ idx::mom     ] *= 1 - eps;
            u[ idx::mom + 1 ] *= 1 - eps;
            u[ idx::mom + 2 ] += eps
                * ( mp_t - f_t( u[ idx::mom + 2 ] ) );
            //  Energy:  thermal toward the IC,  kinetic from
            //  the relaxed momenta,  magnetic from the
            //  relaxed bcc.
            const f_t b2 = u[ idx::bcc ] * u[ idx::bcc ]
                + u[ idx::bcc + 1 ] * u[ idx::bcc + 1 ]
                + u[ idx::bcc + 2 ] * u[ idx::bcc + 2 ];
            const f_t ke = ( u[ idx::mom ] * u[ idx::mom ]
                + u[ idx::mom + 1 ] * u[ idx::mom + 1 ]
                + u[ idx::mom + 2 ] * u[ idx::mom + 2 ] )
                * f_t( 0.5 ) / rho1;
            const f_t ene_t = pre_t / ( gam - 1 ) + ke
                            + f_t( 0.5 ) * b2;
            u[ idx::ene ] += eps
                * ( ene_t - f_t( u[ idx::ene ] ) );
            const f_t rat = r_o0 > 0 ? rho1 / r_o0
                                     : f_t( 1 );
            u[ idx::rho ] = rho1;
            for( int n = idx::bcc + 3; n < n_var; ++ n )
                u[ n ] *= rat;
            }
        }
        //  The v_A cap zone,  decoupled from the relaxation
        //  ring / band:  [ring] va_cap_nth gives a theta -
        //  edge band at ALL radii ( the funnel cone --  the
        //  junk zone where winding otherwise grows B
        //  without bound ),  va_cap_ring = 1 adds the ring
        //  interior.   GRMHD - style sigma cap:  the added
        //  mass carries the LOCAL velocity and temperature
        //  ( mom and the thermal part of ene scale with rho;
        //  the magnetic part stays --  that IS the cap );
        //  the species scale along.   The naive version (
        //  rho up,  mom / ene fixed )  pumped the thermal
        //  energy and fought the chem ene accounting --  the
        //  v9b snap at t = 0.52.
        if( ring_vac > 0
            && ( i[ 1 ] < ring_vac_nth
                 || i[ 1 ] >= geo.n_ceff[ 1 ] - ring_vac_nth
                 || ( ring_vac_ring && ring_r > 0
                      && x[ 0 ] < ring_r )
                 || ( ring_vac_out > 0
                      && x[ 0 ] > ring_vac_out ) ) )
        {
            const f_t b2c = u[ idx::bcc ] * u[ idx::bcc ]
                + u[ idx::bcc + 1 ] * u[ idx::bcc + 1 ]
                + u[ idx::bcc + 2 ] * u[ idx::bcc + 2 ];
            const f_t rho_min = b2c
                / ( ring_vac * ring_vac );
            if( u[ idx::rho ] < rho_min )
            {
                const f_t r0c = __max1
                    ( f_t( u[ idx::rho ] ), f_t( 1e-30 ) );
                const f_t rat2 = rho_min / r0c;
                const f_t m2 = u[ idx::mom ]
                             * u[ idx::mom ]
                    + u[ idx::mom + 1 ] * u[ idx::mom + 1 ]
                    + u[ idx::mom + 2 ] * u[ idx::mom + 2 ];
                const f_t emag = f_t( 0.5 ) * b2c;
                //  etk clamped at 0:  a supersonic cold
                //  cell can hold ene <= ke + emag
                //  numerically;  scaling a NEGATIVE etk
                //  pushed ene below ke + emag ->  negative
                //  pressure ->  the species NaN wave.
                const f_t etk = __max1( f_t( u[ idx::ene ] )
                    - m2 * f_t( 0.5 ) / r0c - emag,
                    f_t( 0 ) );
                u[ idx::ene ] = etk * rat2 + emag
                    + m2 * f_t( 0.5 ) * rat2 / r0c;
                u[ idx::mom     ] *= rat2;
                u[ idx::mom + 1 ] *= rat2;
                u[ idx::mom + 2 ] *= rat2;
                u[ idx::rho ] = rho_min;
                for( int n = idx::bcc + 3; n < n_var; ++ n )
                    u[ n ] *= rat2;
            }
        }
        if( tcap_cs2 > 0 )
        {
            //  The global T ceiling ( see tcap_cs2 ):  scale
            //  the thermal part of ene down to the cap,
            //  kinetic / magnetic parts untouched.
            const f_t r_n  = f_t( u[ idx::rho ] );
            const f_t m2 = u[ idx::mom ] * u[ idx::mom ]
                + u[ idx::mom + 1 ] * u[ idx::mom + 1 ]
                + u[ idx::mom + 2 ] * u[ idx::mom + 2 ];
            const f_t b2 = u[ idx::bcc ] * u[ idx::bcc ]
                + u[ idx::bcc + 1 ] * u[ idx::bcc + 1 ]
                + u[ idx::bcc + 2 ] * u[ idx::bcc + 2 ];
            const f_t ek  = f_t( 0.5 ) * m2
                / __max1( r_n, f_t( 1e-30 ) );
            const f_t em  = f_t( 0.5 ) * b2;
            const f_t eth = f_t( u[ idx::ene ] ) - ek - em;
            const f_t eth_max = r_n * tcap_cs2 / ( gam - 1 );
            if( eth > eth_max )
                u[ idx::ene ] = ek + em + eth_max;
        }
        //  The cylindrical - R poloidal - momentum damper
        //  ( [wedge] R_damp + mom_tau;  see wedg_R ):  inside
        //  R < wedg_R the poloidal momenta relax to 0 on the
        //  ramped tau,  v_phi untouched,  energy following
        //  the kinetic difference exactly ( the mode - 1
        //  bookkeeping --  no thermal injection / removal ).
        if( wedg_R > 0 )
        {
            const f_t r_in = geo.x_fc( 0, 0 );
            const f_t R    = f_t( x[ 0 ] ) * __sin1( f_t( x[ 1 ] ) );
            const f_t s    = ( wedg_R - R ) / ( wedg_R - r_in );
            if( s > 0 )
            {
                const f_t f_w = s > 1 ? f_t( 1 ) : s * s;
                const f_t r_o = __max1
                    ( f_t( u[ idx::rho ] ), f_t( 1e-30 ) );
                f_t eps = b_d.dt * f_w / wedg_tau;
                eps = eps > 1 ? f_t( 1 ) : eps;
                const f_t ke0 = ( u[ idx::mom ]
                                  * u[ idx::mom ]
                    + u[ idx::mom + 1 ] * u[ idx::mom + 1 ] )
                    * f_t( 0.5 ) / r_o;
                u[ idx::mom     ] *= 1 - eps;
                u[ idx::mom + 1 ] *= 1 - eps;
                const f_t ke1 = ( u[ idx::mom ]
                                  * u[ idx::mom ]
                    + u[ idx::mom + 1 ] * u[ idx::mom + 1 ] )
                    * f_t( 0.5 ) / r_o;
                u[ idx::ene ] += ke1 - ke0;
                //  [wedge] v_max:  the POLOIDAL VELOCITY CAP ( the
                //  "cap" companion of the damper above;  see wedg_vmax ).
                //  Only the EXCESS above v_max is removed, the removal
                //  is ramped by the same  f_w  so it fades at the cone
                //  edge,  v_phi is untouched,  and the removed kinetic
                //  energy comes out of ene exactly ( no thermal
                //  injection ):
                //      v_p^2  ->  v_max^2 + ( 1 - f_w ) ( v_p^2 - v_max^2 ).
                if( wedg_vmax > 0 )
                {
                    const f_t pm2 = u[ idx::mom ] * u[ idx::mom ]
                                  + u[ idx::mom + 1 ] * u[ idx::mom + 1 ];
                    const f_t vp2 = pm2 / ( r_o * r_o );
                    const f_t vm2 = wedg_vmax * wedg_vmax;
                    if( vp2 > vm2 )
                    {
                        const f_t v2n = vm2
                            + ( f_t( 1 ) - f_w ) * ( vp2 - vm2 );
                        const f_t sc = sqrt( v2n / vp2 );
                        const f_t k0 = pm2 * f_t( 0.5 ) / r_o;
                        u[ idx::mom     ] *= sc;
                        u[ idx::mom + 1 ] *= sc;
                        const f_t k1 = ( u[ idx::mom ]
                                         * u[ idx::mom ]
                            + u[ idx::mom + 1 ] * u[ idx::mom + 1 ] )
                            * f_t( 0.5 ) / r_o;
                        u[ idx::ene ] += k1 - k0;
                    }
                }
            }
        }
        for( int n = 0; n < n_var; ++ n )
            u_tgt[ n ] = u[ n ];
        return;
    };

    //  The CK15 matched sources:  the pointwise centrifugal /
    //  gravity terms the base src_geo applied are replaced by
    //  the face - exponential forms that the CK15 -
    //  reconstructed pressure flux cancels EXACTLY on a
    //  discrete hydrostatic state:  S = sum_f ( A_f / V ) p
    //  ( exp( +- lam d_f ) - 1 ),  with A_f / V backed out of
    //  the exact pressure moments the base src_geo uses
    //  ( A_f / V = cm0 sin( th_f ) / ( cos th_l - cos th_r )
    //  on theta faces,  2 cm0 r_f^2 / ( r_r^2 - r_l^2 ) on
    //  radial faces ).   Gated per cell to near - balance
    //  states ( rho, pre > 0 and | lam d | < 2;  failing cells
    //  keep the base pointwise term ).
    template< class f2_T, class f_T,
              class bdt_T, class idx_T >
    __device__ __forceinline__ void  src_geo
    ( f2_T * u, const f_T * w, const bdt_T & b_d,
      const idx_T & i, const coord_t & x )  const
    {
        super_t::src_geo( u, w, b_d, i, x );
        if( ck15 == 0 || omega_0 != 0 )
            return;
        const auto & geo = b_d.geo;
        const auto r    = x[ 0 ];
        const auto s_th = __sin1( x[ 1 ] );
        const auto c_th = __cos1( x[ 1 ] );
        const auto rho  = w[ idx::rho ];
        const auto pre  = w[ idx::ene ];
        const auto v1   = w[ idx::mom + 1 ];
        const auto vf   = w[ idx::mom + 2 ];
        const auto r_l  = geo.x_fc( 0, i[ 0 ]     );
        const auto r_r  = geo.x_fc( 0, i[ 0 ] + 1 );
        const auto rl2  = r_l * r_l;
        const auto rr2  = r_r * r_r;
        const auto cm0  = 3 * ( rr2 - rl2 ) / 2.f
                        / ( rr2 * r_r - rl2 * r_l );
        static const f_T pi ( 3.14159265358979323846 );
        if( ck15 & 1 )
        {
            const auto lam = vf * vf * c_th / s_th
                           * rho / pre;
            const auto a = geo.x_fc( 1, i[ 1 ] + 1 ) - x[ 1 ];
            const auto b = x[ 1 ] - geo.x_fc( 1, i[ 1 ] );
            if( pre > 0 && rho > 0
                && __abs1( lam * a ) < 2
                && __abs1( lam * b ) < 2 )
            {
                //  Remove the pointwise centrifugal term the
                //  base just added,  then the matched form.
                u[ idx::mom + 1 ] -= b_d.dt * rho
                    * vf * vf * c_th / s_th / r;
                auto th_l = geo.x_fc( 1, i[ 1 ]     );
                auto th_r = geo.x_fc( 1, i[ 1 ] + 1 );
                th_l = ( th_l < 0  ? 0  : th_l );
                th_r = ( th_r > pi ? pi : th_r );
                u[ idx::mom + 1 ] += b_d.dt * cm0 * pre
                    * ( __sin1( th_r )
                        * ( __exp1( lam * a ) - 1 )
                      - __sin1( th_l )
                        * ( __exp1( - lam * b ) - 1 ) )
                    / ( __cos1( th_l ) - __cos1( th_r ) );
            }
        }
        if( ck15 & 2 )
        {
            //  Radial:  centrifugal AND gravity are both
            //  inside lam ( gravity is reduced to the work
            //  term in add_flux where this gate fires ).
            const auto lam = ( ( v1 * v1 + vf * vf ) / r
                             - gm / ( r * r ) )
                           * rho / pre;
            const auto a = r_r - r;
            const auto b = r - r_l;
            if( pre > 0 && rho > 0
                && __abs1( lam * a ) < 2
                && __abs1( lam * b ) < 2 )
            {
                u[ idx::mom ] -= b_d.dt * rho
                    * ( v1 * v1 + vf * vf ) / r;
                u[ idx::mom ] += b_d.dt * 2 * cm0 * pre
                    * ( rr2 * ( __exp1( lam * a ) - 1 )
                      - rl2 * ( __exp1( - lam * b ) - 1 ) )
                    / ( rr2 - rl2 );
            }
        }
        return;
    };

    template < class bdt_T, class eos_T >
    __device__ __forceinline__ void  add_flux
    ( const bdt_T &  b_d , const eos_T &  eos,
      const   int & axis , const bool  & side,
      const   int & step ) const
    {
        return super_t::add_flux
             ( b_d, eos, axis, side, step );
    };

    //  Rim EMF zeroing ( see disk_emf0_k ):  launched ahead
    //  of the CT update in intg_flx AND intg_field_only (
    //  the operator - split diffusion sweep ).   emf_fix
    //  can not carry this --  sph_t::intg_flx calls it by
    //  a direct ( non - CRTP ) name,  so the intg_flx
    //  override is the only hook covering both paths.
    template< class bdt_T >
    __host__ void emf_rim0
    ( const bdt_T & b_d, const mesh::block::dual_t & d,
      device::base_t & dev )              const
    {
        if( ! emf0 || ( supp_nr <= 0 && supp_nt <= 0 ) )
            return;
        const int n_th ( 128 );
        dev.launch( disk_emf0_k< bdt_T >,
                    std::make_tuple
                    ( dim3( b_d.geo.n_ceff[ 0 ] + 1,
                            b_d.geo.n_ceff[ 1 ] + 1, 1 ),
                      dim3( n_th, 1, 1 ), 0 ),
                    d.stream, b_d, supp_nr, supp_nt );
        return;
    };

    //  Post - CT B_phi relaxation of the ring.   DEFAULT
    //  OFF ( [ring] bphi_rlx,  forensics only ):  scaling
    //  B_phi locally shifts the cell div by
    //  - eps * ( phi - part of div )  --  pointwise nonzero
    //  once the field is phi - asymmetric,  so this INJECTS
    //  div error.   The field side of the ring lives in
    //  the ring resistivity ( disk_eta_t,  CT diffusion,
    //  exactly div - free ) instead.
    template< class bdt_T >
    __host__ void bphi_rlx
    ( const bdt_T & b_d, const mesh::block::dual_t & d,
      const int & step, device::base_t & dev )    const
    {
        if( ring_r <= 0 || ! ring_bphi )
            return;
        const int n_th ( 128 );
        dev.launch( disk_bphi_rlx_k< bdt_T >,
                    std::make_tuple
                    ( dim3( b_d.geo.n_ceff[ 0 ] + 1,
                            b_d.geo.n_ceff[ 1 ] + 1, 1 ),
                      dim3( n_th, 1, 1 ), 0 ),
                    d.stream, b_d, 1 - step, ring_r,
                    ring_tau );
        return;
    };

    template< class bdt_T, class eos_T >
    __host__ void intg_flx
    ( const bdt_T & b_d, const mesh::block::dual_t & d,
      const eos_T & eos, const int    & step,
      device::base_t & dev )              const
    {
        emf_rim0( b_d, d, dev );
        super_t::intg_flx( b_d, d, eos, step, dev );
        bphi_rlx( b_d, d, step, dev );
        return;
    };

    template< class bdt_T >
    __host__ void intg_field_only
    ( const bdt_T & b_d, const mesh::block::dual_t & d,
      const int    & step, device::base_t & dev ) const
    {
        emf_rim0( b_d, d, dev );
        super_t::intg_field_only( b_d, d, step, dev );
        bphi_rlx( b_d, d, step, dev );
        return;
    };
};

////////////////////////////////////////////////////////////
//  Hydro flavor ( the pure - hydrodynamics discriminator
//  for the balance tests ):  same gravity,  base =
//  hydro sph integrator.  The CK15 pair is implemented here
//  too ( the matched sources below + rec_ck15_hyd_t in
//  usr.cpp ).

struct stp_hyd_t
    : dynamics::hydro::integrator::sph_t< stp_hyd_t >
{
    ////////// Data and types //////////
    using super_t
        = dynamics::hydro::integrator::sph_t< stp_hyd_t >;
    using coord_t = type::      coord_t;
    using float_t = type::      float_t;

    bool       enabled;
    float_t       gm;
    //  [prob] ck15:  same CK15 matched - source pair as the
    //  MHD variant ( stp_t );  default 3.
    int        ck15;

    int        dbg_i[ 3 ];   // [prob] dbg_i{0,1,2}:  cell
                             // for the balance print

    __host__ stp_hyd_t(  )
    {
        enabled = true;
        ck15    = 3;
        gm      = 0;
        dbg_i[ 0 ] = dbg_i[ 1 ] = dbg_i[ 2 ] = -1;
        return;
    };

    __host__ virtual void init
    ( const input & args,
      const::dynamics::base_t & mod ) override
    {
        super_t::init( args, mod );
        phys::unit_t< type::float2_t > unit;
        unit.init( args );
        const auto mstar = args.get< float_t >
            ( "prob", "mstar_cgs", 2e33 );
        gm = phys::cgs::G * mstar / unit.derive( 0, 3, -2 );
        enabled = args.get< bool >
                ( "dynamics", "enabled", true );
        ck15    = args.get< int >
                ( "prob", "ck15", 3 );
        dbg_i[ 0 ] = args.get< int >( "prob", "dbg_i0", -1 );
        dbg_i[ 1 ] = args.get< int >( "prob", "dbg_i1", -1 );
        dbg_i[ 2 ] = args.get< int >( "prob", "dbg_i2", -1 );
        return;
    };

    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void  add_flux
    ( dat_T & d, const bdt_T & b_d,
      const eos_T & eos )                      const
    {
        const auto & geo = b_d.geo;
        const auto & self = get_self(  );

        int i[ 3 ], i_1d_c, i_1d_p;
        self.dispatch( i, i_1d_c, i_1d_p, d, b_d );

        using  f_t = typename decltype( b_d.hyd )::f_t;
        dynamics::hydro::integrator::flux_info_t< f_t > f;

        auto * u     = d.u.at( i_1d_c );
        auto * u_tgt = self.load( u, b_d.hyd.u, i, d.step );
        const auto & n_var = b_d.hyd.u.n_int;

        coord_t x;
        geo.x_cc( x, i );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            self.load_flux_sfc( f, d, b_d, i, i_1d_p, a );
            for( int n = 0; n < n_var; ++ n )
                self.flux_to_du_sph
                    ( u, f, b_d, i, x, a, n );
        }
        const bool dbg = ( dbg_i[ 0 ] >= 0
            && i[ 0 ] == dbg_i[ 0 ] && i[ 1 ] == dbg_i[ 1 ]
            && i[ 2 ] == dbg_i[ 2 ] );
        f_t du_src_r( 0 ), du_flx_r( 0 ), du_grav( 0 );
        f_t rho_src( 0 );
        const auto * q_s = d.q.at( i_1d_p );
        {
            const f_t m0r = u[ idx::mom ];
            rho_src = f_t( q_s[ idx::rho ] );
            self.src_geo( u, q_s, b_d, x, i );
            du_src_r = u[ idx::mom ] - m0r;
        }
        if( enabled )
        {
            // Same time level as src_geo's centrifugal term
            // ( see the note in stp_t ).  The ck15 & 2 gate
            // logic is the same as stp_t's ( the matched
            // radial source carries gravity where it fires ).
            bool grav_on = true;
            if( ck15 & 2 )
            {
                const f_t rho_w = f_t( q_s[ idx::rho ] );
                const f_t pre_w = f_t( q_s[ idx::ene ] );
                const f_t v1w = f_t( q_s[ idx::mom + 1 ] );
                const f_t vfw = f_t( q_s[ idx::mom + 2 ] );
                const f_t r_l = geo.x_fc( 0, i[ 0 ]     );
                const f_t r_r = geo.x_fc( 0, i[ 0 ] + 1 );
                const f_t lam = ( ( v1w * v1w + vfw * vfw )
                                  / x[ 0 ]
                                - gm / ( x[ 0 ] * x[ 0 ] ) )
                              * rho_w / pre_w;
                if( pre_w > 0 && rho_w > 0
                    && __abs1( lam * ( r_r - x[ 0 ] ) ) < 2
                    && __abs1( lam * ( x[ 0 ] - r_l ) ) < 2 )
                {
                    grav_on = false;
                    const f_t rr = f_t( u[ idx::rho ] );
                    if( rr > 0 )
                        u[ idx::ene ] -= b_d.dt * gm
                            * f_t( u[ idx::mom ] )
                            / ( x[ 0 ] * x[ 0 ] );
                }
            }
            if( grav_on )
            {
                const f_t acc = -gm / ( x[ 0 ] * x[ 0 ] );
                const auto  dt_rho = b_d.dt * rho_src;
                const auto p0 = f_t( u[ idx::mom ] );
                u[ idx::mom ] += acc * dt_rho;
                const auto p1 = f_t( u[ idx::mom ] );
                du_grav = p1 - p0;
                u[ idx::ene ] += ( p1 * p1 - p0 * p0 )
                               / ( 2 * rho_src );
            }
        }
        if( dbg )
            printf( "HYDDBG: dt %e flx_r %e src_r %e grav %e\n",
                    double( b_d.dt ), double( du_flx_r ),
                    double( du_src_r ), double( du_grav ) );
        for( int n = 0; n < n_var; ++ n )
            u_tgt[ n ] = u[ n ];
        return;
    };

    //  The CK15 matched sources,  hydro flavor --  identical
    //  to stp_t's ( minus the magnetic terms;  the hydro base
    //  src_geo has the ( x, i ) argument order ).
    template< class f2_T, class f_T,
              class bdt_T, class idx_T >
    __device__ __forceinline__ void  src_geo
    ( f2_T * u, const f_T * w, const bdt_T & b_d,
      const coord_t & x, const idx_T & i )  const
    {
        super_t::src_geo( u, w, b_d, x, i );
        if( ck15 == 0 || omega_0 != 0 )
            return;
        const auto & geo = b_d.geo;
        const auto r    = x[ 0 ];
        const auto s_th = __sin1( x[ 1 ] );
        const auto c_th = __cos1( x[ 1 ] );
        const auto rho  = w[ idx::rho ];
        const auto pre  = w[ idx::ene ];
        const auto v1   = w[ idx::mom + 1 ];
        const auto vf   = w[ idx::mom + 2 ];
        const auto r_l  = geo.x_fc( 0, i[ 0 ]     );
        const auto r_r  = geo.x_fc( 0, i[ 0 ] + 1 );
        const auto rl2  = r_l * r_l;
        const auto rr2  = r_r * r_r;
        const auto cm0  = 3 * ( rr2 - rl2 ) / 2.f
                        / ( rr2 * r_r - rl2 * r_l );
        static const float_t pi ( 3.14159265358979323846 );
        if( ck15 & 1 )
        {
            const auto lam = vf * vf * c_th / s_th
                           * rho / pre;
            const auto a = geo.x_fc( 1, i[ 1 ] + 1 ) - x[ 1 ];
            const auto b = x[ 1 ] - geo.x_fc( 1, i[ 1 ] );
            if( pre > 0 && rho > 0
                && __abs1( lam * a ) < 2
                && __abs1( lam * b ) < 2 )
            {
                u[ idx::mom + 1 ] -= b_d.dt * rho
                    * vf * vf * c_th / s_th / r;
                auto th_l = geo.x_fc( 1, i[ 1 ]     );
                auto th_r = geo.x_fc( 1, i[ 1 ] + 1 );
                th_l = ( th_l < 0  ? 0  : th_l );
                th_r = ( th_r > pi ? pi : th_r );
                u[ idx::mom + 1 ] += b_d.dt * cm0 * pre
                    * ( __sin1( th_r )
                        * ( __exp1( lam * a ) - 1 )
                      - __sin1( th_l )
                        * ( __exp1( - lam * b ) - 1 ) )
                    / ( __cos1( th_l ) - __cos1( th_r ) );
            }
        }
        if( ck15 & 2 )
        {
            const auto lam = ( ( v1 * v1 + vf * vf ) / r
                             - gm / ( r * r ) )
                           * rho / pre;
            const auto a = r_r - r;
            const auto b = r - r_l;
            if( pre > 0 && rho > 0
                && __abs1( lam * a ) < 2
                && __abs1( lam * b ) < 2 )
            {
                u[ idx::mom ] -= b_d.dt * rho
                    * ( v1 * v1 + vf * vf ) / r;
                u[ idx::mom ] += b_d.dt * 2 * cm0 * pre
                    * ( rr2 * ( __exp1( lam * a ) - 1 )
                      - rl2 * ( __exp1( - lam * b ) - 1 ) )
                    / ( rr2 - rl2 );
            }
        }
        return;
    };

    template < class bdt_T, class eos_T >
    __device__ __forceinline__ void  add_flux
    ( const bdt_T &  b_d , const eos_T &  eos,
      const   int & axis , const bool  & side,
      const   int & step ) const
    {
        return super_t::add_flux
             ( b_d, eos, axis, side, step );
    };
};

};                              // namespace interm
