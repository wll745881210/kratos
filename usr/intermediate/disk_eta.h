#pragma once

#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"
#include "../../src/modules/chemistry/utilities/periodic.h"
#include "../../src/utilities/string.h"
#include <regex>
#include <vector>

namespace interm
{
////////////////////////////////////////////////////////////
//  The chemistry -> diffusivity bridge ( design.md sec. 4,
//  T10 ):  the conductivity tensor from the local
//  composition,  WBG19 's prescription verbatim (
//  microphys src / chemistry_ext / non_ideal.cpp ):
//
//  per species k ( charge z_k,  mass m_k,  n_k ):
//    gamma_k = <sigma v>_k / ( m_k + m_ave ),
//    b_br_k  = | z_k | e / ( m_k c gamma_k ),   the Hall -
//    parameter per unit B * rho coefficient,
//    beta_k  = z_k b_br_k B / rho,
//    sig_O  += n_k z_k^2 e c b_br_k / rho        ( B - free
//                                                  form ),
//    sig_P  += n_k z_k e c / ( B ( 1 + beta_k^2 ) ),
//    sig_H  += n_k z_k e c beta_k / ( B ( 1 + beta_k^2 ) ),
//  eta_O = c^2 / ( 4 pi sig_O ),
//  eta_A = c^2 sig_P / ( 4 pi ( sig_H^2 + sig_P^2 ) )
//          - eta_O.
//  Hall is EVALUATED but never returned ( eh = 0 while the
//  hyper - resistivity formulation is off --  the project
//  decision ).
//
//  Collision rates:  ions  2e-9 | z | / sqrt( m_ave )
//  ( Draine 2011 via Bai 2011 );   electrons  4.8e-9 max(
//  ( T / 100 + 0.03 )^0.68, 1 )  ( Draine 2011 eq. 2.41 );
//  grains  max( 1.3e-9 | z |,  4e-3 sqrt( T / 100 )
//  ( r_dust / 1e-4 cm )^2 ).
//
//  The provider is trivially copyable ( fixed - size tables
//  built at read( ) from the [ chemistry ] species list
//  alone --  no pointers into the chemistry module ).

struct disk_eta_t
{
    ////////// Constants //////////
    static constexpr int n_max = 48;

    ////////// Data //////////
    int     n_ion;                  //  charged non - e / Gr
    int     i_ion[ n_max ];         //  species slot
    type::float_t  m_ion[ n_max ];  //  CGS
    type::float_t  z_ion[ n_max ];  //  signed,  | z | = 1
    int     i_e, i_grp, i_grm;
    type::float_t  m_gr;            //  CGS
    type::float_t  r_dust;          //  CGS
    //  Unit conversions ( built from [ unit ] ):
    type::float_t  b2cgs;     //  code B -> Gauss
    type::float_t  rho2cgs;   //  code rho -> g / cc
    type::float_t  eta2code;  //  CGS eta -> code
    type::float_t  cs2_to_T;  //  code pre / rho -> K
    type::float_t  m_ave;     //  amu
    type::float_t  eta_cap;   //  code ( physical cap )
    //  [nonideal] eta_cap_c  ( DEFAULT 1 = the coefficient;  <= 0 disables ):
    //  eta SOURCE,  eta <= eta_c * dx  with  eta_c = c t0 / l0  ( the
    //  code - unit light speed ).   The bound is  eta <= eta_cap_c * eta_c * dx.
    type::float_t  eta_cap_c;
    //  [nonideal] eta_cap_ref  ( DEFAULT 1e4 => pn <= 1e4;  measured:
    //  choice ):  the coefficient of the PARABOLIC bound applied at the SAME
    //  place,      eta <= eta_cap_ref * dx^2 / dt_0        ( dt_0 = FULL step )
    //  so the parabolic number  pn = eta dt / dx^2  stays <= eta_cap_ref when
    //  the sub - step equals the full step.   <= 0 disables.
    type::float_t  eta_cap_ref;
    //  The code - unit light speed  ( = c t0 / l0 ),  set in init.
    type::float_t  eta_c;
    //  The relaxation ring 's enhanced Ohmic resistivity
    //  ( [ring];  the poloidal - field side of the ring --
    //  CT diffusion,  exactly div - free ):  eta gets
    //  ring_eta * f( r ) ADDED after the physical cap,  with
    //  f the smooth ( ( ring_r - r ) / ( ring_r - r_in ) )^2
    //  ramp.   ring_r <= 0 disables.
    type::float_t  ring_r;
    type::float_t  ring_eta;
    //  The polar funnel resistivity ( [funnel] ):  the
    //  field - free funnel IC ( psi_min_fac ) fills in over
    //  time and v_A grows in the low - density cone near
    //  the theta edges.   eta gets fun_eta * f_th ADDED (
    //  after the physical cap ),  f_th = ( 1 - d / width )^2
    //  ramping from the theta edges inward over width rad.
    //  CT diffusion --  exactly div - free.
    type::float_t  fun_w;
    type::float_t  fun_eta;
    //  The cylindrical - R winding damper ( [wedge],  the
    //  user 's 2026 - 10 - 15 direction ):  extra eta inside
    //  R = r sin( theta ) < wedge_R,  ramped smoothly from
    //  wedge_R down to R_in,  so the inner region 's toroidal
    //  wind - up ( the Zanni / MRI channel that grows B_phi
    //  without bound at r ~ 1 - 2. 6,  lat 1 - 37 --  run13
    //  t05:  B_phi 0 -> 2. 7e5 by t = 0. 5 ) stays controlled,
    //  while the wind - up OUTSIDE wedge_R is left untouched
    //  --  that one is the physical disk - wind launching
    //  mechanism ( WBG19 ).   Under the implicit mg_cc the
    //  added eta costs no parabolic dt,  so the "dare not
    //  raise the inner eta" constraint of the explicit runs
    //  is gone.   wedge_R <= 0 disables.
    type::float_t  wedge_R;
    type::float_t  wedge_eta;
    //  [wedge] polar_gate ( DEFAULT 1 ):  require the CYLINDRICAL
    //  radius to be resolved by at least this many theta cells,
    //      R > wedge_gate * dx_theta,
    //  before the damper is allowed to act.   R = r sin( theta )
    //  -> 0 for EVERY r as theta -> 0, so at the pole the ramp
    //  saturates and paints the full wedge_eta down the WHOLE axis
    //  column -- where the spherical phi metric is degenerate and
    //  the cylindrical winding the damper exists for is not even
    //  representable by the grid ( R / dx_theta = 0.5 there ).
    //  Measured (full - polar production stack, 2026 - 10 - 25):  that
    //  column is where the |v| / |B| burst and the consequent dt
    //  collapse live -- MGCCDT argmin = (0,0,0),  |v| 0 -> 82,
    //  |B| -> 259 and dt 2.68e-4 -> 7.16e-7 by cycle 3 ( the
    //  DT - COLLAPSE fuse ).   wedge_gate = 0 restores the pre - fix
    //  behaviour exactly ( A / B switch ).
    type::float_t  wedge_gate;
    //  [wedge] ramp:  0 ( default ) = quadratic s^2 ADD,
    //  1 = wide linear ADD,  2 = the BLEND ( eo -> eta_damp,
    //  ea fades ),  3 = the INVERTED blend ( eo fades,
    //  ea -> eta_damp ).
    int          wedge_lin;
    int          wedge_inv;
    //  The extra - eta parabolic - dt guard ( [nonideal]
    //  eta_dt,  code units,  0 = off ):  the ring / funnel
    //  ADDED eta is capped at 0. 25 * dx_min^2 / eta_dt,  so
    //  the explicit / STS parabolic cap stays ~ eta_dt at
    //  every cell  ( coarse cells keep the full eta,  the
    //  fine - theta cells near r_in get it scaled down;
    //  run11's t ~ 3 collapse:  ring eta 0. 3 on dx_theta
    //  1. 4e-3 -> dt_dif 1. 6e-6 ).   The PHYSICAL eta is
    //  untouched ( it owns its own WBG19 cap ).
    type::float_t  eta_dt;
    //  The suppression rim ( [suppress],  mirrored from
    //  disk_chem_sat / disk_intg ):  eta = 0 there --  no
    //  diffusion,  and in particular no eta J^2 heating of
    //  the frozen cells.   [nonideal] eta_supp = 0
    //  disables the mask ( the A / B knob for the mg_cc
    //  freeze investigation ).
    int     supp_nr;
    int     supp_nt;
    int     supp_on;
    //  DIAGNOSTIC ( [profile] eta_out_kill,  >= 1 ):  zero
    //  the PHYSICAL eo/ea in the last N radial cells
    //  ( the boundary band )  -- the A/B probe for the
    //  boundary-cell eta participation in the blow-up.
    int     out_kill;
    //  The TOTAL-eta parabolic cap ( [nonideal] eta_cap_dt,  code
    //  time units,  0 = off ).  Unlike eta_dt ( which only caps the
    //  ring / funnel ADD on top of the physical cap ) this bounds the
    //  FINAL eo / ea -- wedge blend included -- by
    //       eta <= 0.25 * dx_min^2 / eta_cap_dt
    //  so the implicit operator's parabolic number dt*eta/h^2 is
    //  bounded at EVERY cell.  Motivation ( STATUS.md sec. 3bis ):  the
    //  wedge puts one eta = 0.3 on cells spanning 50x in size ( the
    //  polar r*dtheta ~ 1.6e-3 vs the midplane ~8e-2 ),  the coarse
    //  operator then disagrees with the fine one and the V-cycle
    //  amplifies by x6.4 per iteration.
    type::float_t  eta_cap_dt;
    bool           on;

    ////////// Functions //////////
    __host__ disk_eta_t(  )
    {
        n_ion = 0;   i_e = i_grp = i_grm = -1;
        m_gr = 0;  r_dust = 5e-8;
        b2cgs = 1;   rho2cgs = 1;   eta2code = 1;
        cs2_to_T = 1;  m_ave = 2.34;  eta_cap = 0;
        eta_cap_c = 1;   eta_cap_ref = 1e4;   eta_c = 0;
        supp_nr = 0;   supp_nt = 0;   supp_on = 1;
        out_kill = 0;   eta_cap_dt = 0;
        ring_r = 0;   ring_eta = 0;
        fun_w  = 0;   fun_eta  = 0;
        wedge_R = 0;  wedge_eta = 0;
        wedge_lin = 0;   wedge_inv = 0;
        wedge_gate = 1;
        eta_dt = 0;
        on = false;
        return;
    };

    __host__ void read( const input & args )
    {
        //  The ring resistivity works with or without the
        //  chemistry bridge --  read it before the
        //  early return.
        ring_r   = args.get< type::float_t >
                         ( "ring", "r_out",    0 );
        ring_eta = args.get< type::float_t >
                         ( "ring", "eta_max", 0 );
        fun_w    = args.get< type::float_t >
                         ( "funnel", "width",   0 );
        fun_eta  = args.get< type::float_t >
                         ( "funnel", "eta_max", 0 );
        wedge_R   = args.get< type::float_t >
                         ( "wedge", "R_damp", 0 );
        wedge_eta = args.get< type::float_t >
                         ( "wedge", "eta_max", 0 );
        wedge_gate = args.get< type::float_t >
                         ( "wedge", "polar_gate", 1 );
        wedge_lin = args.get< int >
                         ( "wedge", "ramp", 0 );
        wedge_inv = ( wedge_lin == 3 );
        if( wedge_inv )
            wedge_lin = 2;
        eta_dt   = args.get< type::float_t >
                         ( "nonideal", "eta_dt", 0 );
        eta_cap_dt = args.get< type::float_t >
                         ( "nonideal", "eta_cap_dt", 0 );
        //  The bridge only makes sense with chemistry.
        on = args.get< int >( "nonideal", "chem_eta", 0 )
           && args.get< int >( "prob", "chem_on", 0 );
        if( ! on )
            return;

        phys::unit_t< type::float2_t > unit;
        unit.init( args );
        //  Heaviside - Lorentz code B ( p_mag = B^2 / 2 )
        //  to Gauss:  B_G = B_code sqrt( 4 pi rho0 v0^2 ).
        b2cgs     = sqrt( 4 * 3.141592653589793
                          * unit.rho0 * unit.vel0
                                     * unit.vel0 );
        rho2cgs   = unit.rho0;
        eta2code  = unit.t0 / ( unit.l0 * unit.l0 );
        m_ave     = args.get< type::float2_t >
                        ( "profile", "mu_ma", 2.34 );
        cs2_to_T  = m_ave * phys::cgs::ma * unit.vel0
                  * unit.vel0 / phys::cgs::kb;
        r_dust    = args.get< type::float2_t >
                        ( "chemistry", "r_dust_cgs", 5e-8 );
        const auto rho_dust = args.get< type::float2_t >
                        ( "chemistry", "rho_dust_cgs", 3.0 );
        m_gr = 4.1887902 * r_dust * r_dust * r_dust
             * rho_dust;
        //  Physical cap ( WBG19:  ~ 10 c_s h at r_in )  and
        //  the hard ceiling ( WBG19 's eta_max = 1e20;  our default is 1e22,  measured healthy ).
        const type::float2_t cap_cgs = args.get
            < type::float2_t >
            ( "nonideal", "eta_cap_cgs", 1e22 );
        eta_cap = cap_cgs * eta2code;
        eta_cap_c = args.get< type::float_t >
                  ( "nonideal", "eta_cap_c", 1 );
        eta_cap_ref = args.get< type::float_t >
                  ( "nonideal", "eta_cap_ref", 1e4 );
        //  UNIT CONVERSION,  the two bounds are NOT alike:
        //    causal   eta_cgs <= c dx_cm  =>  eta_code <= dx_code * ( c t0/l0 )
        //             ( needs the unit system );
        //    parabolic eta <= eta_cap_ref dx^2 / dt_0
        //             ( already code units:  eta_code = L^2/T -- do NOT convert ).
        eta_c = phys::cgs::c * unit.t0 / unit.l0;
        supp_nr = args.get< int >( "suppress", "n_r_in", 0 );
        supp_nt = args.get< int >( "suppress", "n_th",   0 );
        supp_on = args.get< int >( "nonideal", "eta_supp", 1 );   //  2 = debug probe
        out_kill = args.get< int >( "profile", "eta_out_kill", 0 );

        //  Build the charge / mass table from the species
        //  list alone ( the chem parser 's rules,  incl.
        //  the Gr pseudo - element ).
        ::chemistry::elem_dat_t elem_dat;
        //  enroll Gr ( chemistry::elem_t aggregate:
        //  name,  z,  n,  l,  m[amu],  I_e[eV] )
        if( ! elem_dat.contain( "Gr" ) )
        {
            ::chemistry::elem_t gr;
            gr.name = "Gr";   gr.z = -1;   gr.n = 0;
            gr.l = 0;   gr.m = 1;   gr.I_e = 4.5;
            elem_dat.enroll_elem( gr );
        }
        std::vector< std::string > sps;
        args( "chemistry", "species", sps );
        for( const auto & s0 : sps )
        {
            auto s( s0 );
            int chg = utils::string::replace_all( s, "+", "" );
            chg    -= utils::string::replace_all( s, "-", "" );
            const int idx = ( int ) ( & s0 - & sps[ 0 ] );
            if( s == "e" )
            {
                i_e = idx;
                continue;
            }
            if( s.find( "Gr" ) != std::string::npos )
            {
                if( chg > 0 )       i_grp = idx;
                else if( chg < 0 )  i_grm = idx;
                continue;   //  neutral Gr carries nothing
            }
            if( chg == 0 )
                continue;
            if( n_ion >= n_max )
                throw std::runtime_error
                    ( "disk_eta_t:  too many ions" );
            //  Mass from the periodic table ( same rules as
            //  chemistry::base_t::parse_species_single ).
            auto s_el( s );
            type::float2_t m = -chg * phys::cgs::me;
            std::regex pat_e( "[A-Z][a-z]*(?=\\d*)" );
            std::regex pat_n( "^\\d+" );
            std::smatch mm;
            while( ! s_el.empty(  ) )
            {
                if( ! std::regex_search( s_el, mm, pat_e ) )
                    break;
                const auto & el = elem_dat.elem( mm[ 0 ] );
                s_el = mm.suffix(  ).str(  );
                int cnt = 1;
                if( std::regex_search( s_el, mm, pat_n ) )
                {
                    cnt = std::stoi( mm[ 0 ] );
                    s_el = mm.suffix(  ).str(  );
                }
                m += phys::cgs::ma * cnt * el.m;
            }
            i_ion[ n_ion ] = idx;
            m_ion[ n_ion ] = m;
            z_ion[ n_ion ] = chg;
            ++ n_ion;
        }
        return;
    };

    //  One species ' contribution;  sig = [ O, H, P ].
    template< class f_T > __device__ __forceinline__
    static void add_sig
    ( const f_T b_br, const f_T z, const f_T B,
      const f_T rho, const f_T n, f_T * sig )
    {
        const f_T beta   = z * b_br * B / rho;
        const f_T b2p1_i = 1 / ( 1 + beta * beta );
        const f_T nzec_b = n * z * f_T( phys::cgs::e_charge )
                           * f_T( phys::cgs::c ) / B;
        sig[ 0 ] += nzec_b * beta;
        sig[ 1 ] += nzec_b * b2p1_i;
        sig[ 2 ] += nzec_b * b2p1_i * beta;
        return;
    };

    template< class bdt_T, class idx_T, class f2_T >
    __device__ __forceinline__ void eval
    ( const bdt_T & b_d, const idx_T & i,
      f2_T & eo, f2_T & eh, f2_T & ea )                 const
    {
        using f_t = type::float_t;
        eo = 0;   eh = 0;   ea = 0;
        //  The ring + funnel extra resistivity ( added after
        //  the physical cap at the end;  also the ONLY
        //  contribution when the chemistry bridge is off or
        //  the gas is fully neutral ).
        f_t ring_e ( 0 );
        if( ring_r > 0 )
        {
            const f_t r_in = b_d.geo.x_fc( 0, 0 );
            const f_t s    = ( ring_r
                - f_t( b_d.geo.x_cc( 0, i[ 0 ] ) ) )
                / ( ring_r - r_in );
            if( s > 0 )
                ring_e = ring_eta * ( s > 1 ? f_t( 1 )
                                            : s * s );
        }
        if( fun_w > 0 )
        {
            const f_t th   = b_d.geo.x_cc( 1, i[ 1 ] );
            const f_t t_lo = b_d.geo.x_fc( 1, 0 );
            const f_t t_hi = b_d.geo.x_fc
                ( 1, b_d.geo.n_ceff[ 1 ] );
            const f_t d = __min1( th - t_lo, t_hi - th );
            if( d < fun_w )
            {
                const f_t s = 1 - d / fun_w;
                ring_e += fun_eta * s * s;
            }
        }
        //  The cylindrical - R winding damper ( [wedge] ):  the
        //  same smooth - ramp form as the ring,  but in R = r
        //  sin( theta )  --  a CYLINDRICAL core that follows the
        //  star / inner disk,  independent of height.   The
        //  wind - up inside R_damp is numerical junk ( the
        //  inner region 's field piles up against r_in across
        //  all latitudes );  outside it the winding is the
        //  physical disk - wind launching ( WBG19 )  and is
        //  left alone.
        //  [wedge] ramp = 0 ( default ):  ADD form,  quadratic
        //  s^2.   ramp = 1:  ADD,  wide linear.   ramp = 2:
        //  the BLEND form ( the user 's direction ):
        //      eta = eta_damp * x + eta_phys * ( 1 - x ),
        //  x LINEAR from 0 at R_damp to 1 at R <= R_in ( the
        //  column above the carved - out region ),  eta_phys
        //  the CURRENT local physical value.   The total eta
        //  is then CONTINUOUS ( value AND bounded slope )
        //  across R_damp -- no edge for the implicit operator
        //  to trip on ( the ADD form's single - cell edge
        //  spike,  run13 mode - 1 ).   The blend applies to eo;
        //  ea keeps its physical value ( an Ohmic - flavored
        //  damper ).   Blend finalized after the physical eta
        //  below ( wedg_x is carried there ).
        f_t wedg_x ( -1 );
        if( wedge_R > 0 )
        {
            const f_t r_in = b_d.geo.x_fc( 0, 0 );
            const f_t R    = f_t( b_d.geo.x_cc( 0, i[ 0 ] ) )
                           * __sin1( f_t( b_d.geo.x_cc
                                    ( 1, i[ 1 ] ) ) );
            const f_t s    = ( wedge_R - R )
                           / ( wedge_R - r_in );
            //  Polar resolution gate  ( see the wedge_gate note ):
            //  near the pole R ~ r * theta,  so the scale that must
            //  resolve it is the theta cell width.   n_dim < 2 leaves
            //  the gate open ( no theta direction to resolve in ).
            const f_t dx_th
                = ( b_d.geo.n_dim > 1
                    ? f_t( b_d.geo.dx_cfl( 1, i ) ) : f_t( 0 ) );
            const bool res_ok
                = ( wedge_gate <= 0 )
               || ( R > wedge_gate * dx_th );
            if( s > 0 && res_ok )
            {
                if( wedge_lin == 2 )
                    wedg_x = s > 1 ? f_t( 1 ) : s;
                else
                {
                    const f_t f_w
                        = ( s > 1 ? f_t( 1 )
                                 : ( wedge_lin ? s : s * s ) );
                    ring_e += wedge_eta * f_w;
                }
            }
        }
        //  The parabolic - dt guard ( [nonideal] eta_dt > 0 ):
        //  cap the ADDED eta at 0. 25 dx_min^2 / eta_dt so the
        //  explicit / STS cap stays ~ eta_dt at every cell --
        //  coarse cells keep the full eta,  the fine - theta
        //  cells near r_in get it scaled down ( run11 t ~ 3
        //  collapse:  ring eta 0. 3 on dx_theta 1. 4e-3 gave
        //  dt_dif 1. 6e-6 ).   The PHYSICAL eta above is not
        //  touched ( it owns its own WBG19 cap ).
        if( eta_dt > 0 && ring_e > 0 )
        {
            f_t dx( b_d.geo.dx_cfl( 0, i ) );
            for( int a = 1; a < b_d.geo.n_dim; ++ a )
                dx = __min1( dx, b_d.geo.dx_cfl( a, i ) );
            const f_t e_cap = f_t( 0.25 ) * dx * dx / eta_dt;
            ring_e = __min1( ring_e, e_cap );
        }
        if( ! on )
        {
            eo = ring_e;
            return;
        }
        //  The suppression rim:  eta = 0 ( no diffusion,  no
        //  eta J^2 heating of the frozen cells ).
        if( supp_on
            && ( i[ 0 ] < supp_nr || i[ 1 ] < supp_nt
                 || i[ 1 ] >= b_d.geo.n_ceff[ 1 ] - supp_nt ) )
            return;
        //  DEBUG probe ( eta_supp = 2 ):  print the mask
        //  state of a probe cell once.
        if( supp_on == 2 && i[ 0 ] == 10 && i[ 2 ] == 0
            && ( i[ 1 ] == 256 || i[ 1 ] == 511
                 || i[ 1 ] == 0 ) )
            printf( "ETAMASK i1 %d n1 %d supp %d %d\n",
                    i[ 1 ], b_d.geo.n_ceff[ 1 ], supp_nr,
                    supp_nt );
        if( supp_on == 2 && i[ 0 ] == 10 && i[ 2 ] == 0
            && i[ 1 ] == 256 )
        {
            const auto * w = b_d.hyd.w.at( i );
            printf( "ETAPROBE rho %e pre %e bcc %e %e %e"
                    "  sp[ 0..3 ] %e %e %e %e\n",
                    ( double ) w[ idx::rho ],
                    ( double ) w[ idx::ene ],
                    ( double ) w[ idx::bcc ],
                    ( double ) w[ idx::bcc + 1 ],
                    ( double ) w[ idx::bcc + 2 ],
                    ( double ) w[ idx::bcc + 3 ],
                    ( double ) w[ idx::bcc + 4 ],
                    ( double ) w[ idx::bcc + 5 ],
                    ( double ) w[ idx::bcc + 6 ] );
        }
        const auto * w = b_d.hyd.w.at( i );
        const f_t rho = __max1( f_t( w[ idx::rho ] ),
                                f_t( 1e-30 ) );
        f_t b2( 0 );
        for( int n = 0; n < 3; ++ n )
            b2 += f_t( w[ idx::bcc + n ] )
                * f_t( w[ idx::bcc + n ] );
        //  CGS working values.
        const f_t Bg   = __sqrt1( b2 ) * b2cgs;
        const f_t B    = __max1( Bg, f_t( 1e-30 ) );
        const f_t rhoc = rho * rho2cgs;
        const f_t T    = f_t( w[ idx::ene ] ) / rho
                       * cs2_to_T;
        const f_t Tc   = __min1( __max1( T, f_t( 1 ) ),
                                 f_t( 1e7 ) );
        //  Species slots:  y~_n = n_n m_A / rho0;  with
        //  rho0 = m_A the slot IS n_n in 1 / cc.
        const auto * sp = w + idx::bcc + 3;
        const f_t m_ave_cgs = m_ave * f_t( phys::cgs::ma );

        f_t sig[ 3 ] = { 0, 0, 0 };
        //  Electrons ( Draine 2011 eq. 2.41 ).
        if( i_e >= 0 )
        {
            const f_t n = sp[ i_e ];
            if( n > 0 )
            {
                const f_t sv = f_t( 4.8e-9 ) * __max1
                    ( powf( Tc / 100 + f_t( 3e-2 ),
                            f_t( 0.68 ) ), f_t( 1 ) );
                const f_t gamma = sv / m_ave_cgs;
                const f_t b_br  = f_t( phys::cgs::e_charge )
                    / ( f_t( phys::cgs::me )
                        * f_t( phys::cgs::c ) * gamma );
                add_sig( b_br, f_t( -1 ), B, rhoc, n, sig );
            }
        }
        //  Ions ( Bai 2011 / Draine 2011 ).
        const f_t sv_ion = f_t( 2e-9 ) / __sqrt1( m_ave );
        for( int k = 0; k < n_ion; ++ k )
        {
            const f_t n = sp[ i_ion[ k ] ];
            if( n <= 0 )
                continue;
            const f_t gamma = sv_ion
                            / ( m_ion[ k ] + m_ave_cgs );
            const f_t b_br  = f_t( phys::cgs::e_charge )
                / ( m_ion[ k ] * f_t( phys::cgs::c )
                    * gamma );
            add_sig( b_br, z_ion[ k ], B, rhoc, n, sig );
        }
        //  Charged grains.
        for( int g = 0; g < 2; ++ g )
        {
            const int gi = g ? i_grm : i_grp;
            if( gi < 0 )
                continue;
            const f_t n = sp[ gi ];
            if( n <= 0 )
                continue;
            const f_t z   = g ? f_t( -1 ) : f_t( 1 );
            const f_t sv  = __max1( f_t( 1.3e-9 ),
                f_t( 4e-3 ) * __sqrt1( Tc / 100 )
                * ( r_dust / f_t( 1e-4 ) )
                * ( r_dust / f_t( 1e-4 ) ) );
            const f_t b_br = f_t( phys::cgs::e_charge )
                           / ( f_t( phys::cgs::c ) * sv );
            add_sig( b_br, z, B, rhoc, n, sig );
        }
        if( sig[ 0 ] <= 0 )
        {
            eo = ring_e;
            return;
        }
        const f_t c2_4pi = f_t( phys::cgs::c )
                         * f_t( phys::cgs::c )
                         / f_t( 4 * 3.141592653589793 );
        const f_t sig_b  = __sqrt1( sig[ 1 ] * sig[ 1 ]
                                  + sig[ 2 ] * sig[ 2 ] );
        f_t eo_cgs = c2_4pi / sig[ 0 ];
        f_t ea_cgs = c2_4pi / sig_b * sig[ 2 ] / sig_b
                   - eo_cgs;
        eo_cgs = __max1( eo_cgs, f_t( 1e-32 ) );
        ea_cgs = __max1( ea_cgs, f_t( 1e-32 ) );
        //  ADMISSIBILITY AT THE SOURCE  ( 2026 - 10 - 26 ):  the chemistry -
        //  derived eta is bounded HERE,  before the numerical additions:
        //    eta_cap          :  the scalar physical ceiling ( WBG19 ... ),
        //    eta_c * dx       :  CAUSAL -- the diffusive speed across one
        //                       cell must not exceed c  ( unit - converted
        //                       through  c t0 / l0 ),
        //    eta_cap_ref dx^2/dt_0 :  PARABOLIC -- pn = eta dt/dx^2 bounded
        //                       by eta_cap_ref at the FULL step  ( code units ).
        //  The extent is the SMALLEST of the three ARC lengths
        //      dr,   r dtheta,   r sin( theta ) dphi
        //  ( measured 2026-10-26 with the DBG_ETA_DX probe,  cells (0,2) and
        //  (32,256):  dx_cfl( theta ) IS the arc length r dtheta -- ratio
        //  0.835 / 1.0000 -- and dx_cfl( phi ) IS r sin(theta) dphi -- 0.987;
        //  dx_cfl( r ) is the exact spacing ).  So the per - direction
        //  dx_cfl minimum below is that rule;  it is even slightly tighter
        //  than the analytic form near the axis.
        f_t dx_e( b_d.geo.dx_cfl( 0, i ) );
        for( int a = 1; a < b_d.geo.n_dim; ++ a )
            dx_e = __min1( dx_e, b_d.geo.dx_cfl( a, i ) );
        f_t cap_e( eta_cap );
        if( eta_cap_c > f_t( 0 ) )
            cap_e = __min1( cap_e, eta_cap_c * eta_c * dx_e );
        const f_t dt_ref( b_d.dt_0 );
        if( eta_cap_ref > f_t( 0 ) && dt_ref > f_t( 0 ) )
            cap_e = __min1( cap_e, eta_cap_ref * dx_e * dx_e / dt_ref );
        eo = __min1( eo_cgs * eta2code, cap_e );
        ea = __min1( ea_cgs * eta2code, cap_e );
        //  DIAGNOSTIC eta_out_kill:  zero the physical eta
        //  in the last out_kill radial cells.   Bit flags:
        //  1 = kill both,  2 = kill eo only,  3 = kill ea
        //  only  ( the anisotropic / ohmic split probe ).
        if( out_kill > 0
            && i[ 0 ] >= b_d.geo.n_ceff[ 0 ] - out_kill )
        {
            if( out_kill == 1 ) { eo = 0;  ea = 0;  }
            if( out_kill == 2 ) { eo = 0;             }
            if( out_kill == 3 ) {           ea = 0;   }
        }
        //  [nonideal] eta_dt,  stage 2 ( see the guard above
        // for the ADDED eta ):  a CAPPED physical eta is
        // itself a WBG19 compromise ( the fully decoupled
        // wind zone,  ~3% of the domain,  Lambda_A << 1 );
        // applying the same dx - adaptive limit ON TOP of the
        // cap sacrifices only diffusion accuracy that is
        // already lost there,  keeping the parabolic dt ~
        // eta_dt at the fine cells ( run11 / probe4 residual
        // collapse:  eta_A at cap on lat ~ 10 deg,  dx_theta
        // 3. 7e-3 -> dt_dif 3. 8e-6 ).   Eta BELOW the cap is
        // untouched.
        if( eta_dt > 0 )
        {
            f_t dx( b_d.geo.dx_cfl( 0, i ) );
            for( int a = 1; a < b_d.geo.n_dim; ++ a )
                dx = __min1( dx, b_d.geo.dx_cfl( a, i ) );
            const f_t e_cap = f_t( 0.25 ) * dx * dx / eta_dt;
            if( eo_cgs * eta2code >= eta_cap )
                eo = __min1( eo, e_cap + ring_e );
            if( ea_cgs * eta2code >= eta_cap )
                ea = __min1( ea, e_cap );
        }
        eo += ring_e;
        //  The BLEND wedge finalization ( ramp = 2;  see the
        //  wedg_x note above ):  eo -> eta_damp*x + eo*(1-x),
        //  ea -> ea*(1-x).   Applied AFTER the physical eta ( and
        //  its cap )  so eta_phys is the true current value;
        //  the ring / funnel ADD parts ride inside eo.
        //  ea 's fade ( the user 's 2026 - 10 - 15 refinement ):
        //  the ambipolar term is ANISOTROPIC ( eta_A acts on
        //  J_perp only )  and at high contrast it breaks the
        //  implicit operator 's conditioning ( the run13
        //  var - eta nonconvergence history );  fading ea to
        //  zero inside the wedge leaves a purely OHMIC
        //  damper -- isotropic,  well - conditioned -- while
        //  eo transitions to eta_damp.
        if( wedg_x > 0 )
        {
            if( wedge_inv )
                //  The INVERTED blend ( [wedge] ramp = 3,  the
                //  other extreme of the ea / eo roles ):
                //  eo fades to zero,  ea -> eta_damp.   The
                //  ambipolar damper is anisotropic ( J_perp
                //  only )  -- it preserves the field along
                //  B,  which is exactly the winding - damping
                //  profile we want,  IF the operator holds.
                //  ( The run13 matrix:  fade - ea helped the
                //  operator but removed the physical
                //  ambipolar stabilization;  this variant
                //  tests the opposite trade. )
            {
                eo *= f_t( 1 ) - wedg_x;
                ea = wedge_eta * wedg_x
                   + ea * ( f_t( 1 ) - wedg_x );
            }
            else
            {
                eo = wedge_eta * wedg_x
                   + eo * ( f_t( 1 ) - wedg_x );
                ea *= f_t( 1 ) - wedg_x;
            }
        }
        //  The TOTAL parabolic cap ( [nonideal] eta_cap_dt ):
        //  bound the FINAL eo / ea -- wedge blend included -- by
        //  0. 25 dx_min^2 / eta_cap_dt,  i.e. a parabolic number of
        //  0. 25 at every cell.  This is the per-cell coefficient
        //  limit of STATUS.md sec. 3bis  ( the wedge puts one
        //  eta = 0. 3 on cells spanning 50x in size;  the fine polar
        //  cells then drive the implicit operator far out of the
        //  range the coarse levels can represent ).
        if( eta_cap_dt > 0 )
        {
            f_t dx( b_d.geo.dx_cfl( 0, i ) );
            for( int a = 1; a < b_d.geo.n_dim; ++ a )
                dx = __min1( dx, b_d.geo.dx_cfl( a, i ) );
            const f_t e_cap( f_t( 0.25 ) * dx * dx / eta_cap_dt );
            eo = __min1( eo, e_cap );
            ea = __min1( ea, e_cap );
        }
        if( supp_on == 2 && i[ 0 ] == 10 && i[ 2 ] == 0
            && i[ 1 ] == 256 )
            printf( "ETAOUT sig %e %e %e  eo ea %e %e\n",
                    ( double ) sig[ 0 ], ( double ) sig[ 1 ],
                    ( double ) sig[ 2 ], ( double ) eo,
                    ( double ) ea );
        return;
    };
};

};                              // namespace interm

namespace interm
{
////////////////////////////////////////////////////////////
//  The explicit - sweep flavor:  inert when the implicit
//  mg_cc module owns the chemistry diffusivity ( the
//  double - diffusion guard,  design.md sec. 4 ).

struct disk_eta_ni_t : disk_eta_t
{
    __host__ void read( const input & args )
    {
        disk_eta_t::read( args );
        //  The double - diffusion guard:  when the implicit
        //  mg_cc module owns the diffusivity,  the CHEMISTRY
        //  bridge part of this provider must go silent ( the
        //  mg_cc freeze reads the same provider object via
        //  mg_cc_t = mg_cc_diffusion_t< prx_cc_t, disk_eta_t >
        //  and keeps the ring / funnel ADDITIONS --  those are
        //  the inner - region B_phi winding damper and MUST
        //  stay inside the implicit operator;  the run13 bc0
        //  probe lost them via an over - broad guard and wound
        //  B_phi x1e33 in 0. 02 yr ).   The EXPLICIT sweep
        //  itself is killed in ni_t::init ( mg_cc_eta_var ).
        if( args.get< int >( "nonideal", "mg_cc_eta_var",
                             0 ) )
            on = false;
        return;
    };
};

////////////////////////////////////////////////////////////
//  The per - step eta diagnostic fill ( bdt_chem_t::eta_out,
//  written under the "ni_eta_" prefix ):  eta_O / eta_A in
//  code units,  EXACTLY as the diffusion kernels see them
//  ( the same eval,  the same cap and ring / funnel
//  additions ).   [nonideal] eta_out ( default 1 ) gates the
//  launch from the pgen's step;  the fill itself is a plain
//  cell - loop.   The refresh ORDER matters:  call it after
//  the chemistry step ( the composition the solver wrote
//  drives eval)  and after cons2prim  ( eval reads w ).

template< class bdt_T >
__global__ void disk_eta_out_k
( const bdt_T b_d, const disk_eta_t eta )
{
    const auto & geo = b_d.geo;
    const int    n0  = geo.n_ceff[ 0 ];
    const int    n1  = geo.n_ceff[ 1 ];
    const int    t   = blockIdx.x * blockDim.x
                     + threadIdx.x;
    if( t >= n0 * n1 * geo.n_ceff[ 2 ] )
        return;
    type::idx_t i;
    i[ 2 ] =   t / ( n0 * n1 );
    i[ 1 ] = ( t / n0 ) % n1;
    i[ 0 ] =   t % n0;
    type::float_t eo, eh, ea;
    eta.eval( b_d, i, eo, eh, ea );
    auto * out = b_d.eta_out.at( i );
    out[ 0 ] = eo;
    out[ 1 ] = ea;
    return;
};

};                              // namespace interm
