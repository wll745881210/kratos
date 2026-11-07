#pragma once

#include "../../usr/radiation/fld/fld_driver.h"
#include "../../usr/radiation/fld/fld_kap.h"
#include "../../src/utilities/mapping/loop.h"
#include <type_traits>
#include "disk_chem_bdt.h"
#include "disk_radiation.h"
#include "disk_dust.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  The FLD dust - temperature leg for the ppd ( T11;
//  plan:  docs / radiative_transfer /
//  fld-ppd-integration-plan.md sec. 5.3,  the stage - C
//  main - body wiring;  the isolated bench usr_ext /
//  ppd_fld_tst carried the science validation ).
//
//  Data flow per step ( the bench 's step pattern ):
//    freeze:  co_arr from the LIVE MHD state ( kappa_IR
//             from the Gr abundance,  rho from hyd.u,  the
//             corona radiative floor ) --  via the
//             disk_rad_link_t module link,  the MHD block
//             is passed to the kernels BY VALUE;
//    source:  s_ir[ 1 ] = mfp * flx * E_nu * flux_scale /
//             lum_sc * dt  on the OPTICAL band only (
//             [fld] i_hv_heat ):  the 2 eV band is absorbed
//             by dust alone ( every photochemical cross
//             section is 0 below the data grid ),  so there
//             is no double - count with the chemistry 's
//             photoheating;
//    Picard:  s_ir[ 0 ] = sig * min( E, a T_cap^4 )  ( the
//             sublimation - cap LTE closure ),
//             launch_src + solve_picard,  the change -
//             measure exit on r_tol;
//    push:    the solved E -> the MHD blocks ' e_fld (
//             real cells only ),  read by disk_dust_t::eval
//             in the fld_on mode.
//  The clock is NOT advanced here ( the dynamics module
//  owns it ).

using bdt_fld_ppd_t = rad::fld::bdt_fld_t< type::float_t >;

struct prx_fld_ppd_t : multigrid::proxy_default_t
                       < bdt_fld_ppd_t >
{
    static void set_module( mesh::mod_base_t & mod )
    {
        return;
    };
};

////////////////////////////////////////////////////////////
//  The mixed - block - data launcher ( the driver 's
//  fld_fun2_k is single - type;  ours pairs the FLD block
//  with the MHD block ).

template< class bF_T, class bM_T, class fun_T >
__global__ void ppd_fld_k
( const bF_T b_f, const bM_T b_m, const fun_T fun )
{
    return fun( b_f, b_m );
};

////////////////////////////////////////////////////////////
//  The freeze:  co[ 0 ] = c dt / ( 3 kap rho ),
//  co[ 1 ] = c kap rho dt,   with kap rho = sig_ir * n_Gr *
//  l0 ( the per - code - length opacity;  n_Gr = the sum of
//  the Gr charge states,  the  y~  slot IS the CGS number
//  density since rho0 = m_A ).   The grain - density floor
//  n_fl_gr keeps the operator 's dynamic range bounded in
//  the corona ( the bench 's rad_fl lesson ).

struct ppd_fld_freeze_t
{
    type::float_t    c_l;      //  reduced c,  code units
    type::float_t    dt;
    type::float_t    kap_rho_fac;  //  kappa_ir_cgs * rho0
                                   //  * l0  ( per code
                                   //  length per code rho )
    type::float_t    rho_fl;   //  the radiative floor in
                               //  code density ( the
                               //  bench 's corona lift )

    template< class bF_T, class bM_T >
    __device__ __forceinline__ void operator(  )
    ( const bF_T & b_f, const bM_T & b_m ) const
    {
        const auto & geo = b_f.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        const auto rho = __max1(
            type::float_t( b_m.hyd.u.at( i )[ idx::rho ] ),
            rho_fl );
        const auto kr = kap_rho_fac * rho;  //  1 / length
        b_f.co_arr.at( i )[ 0 ] = c_l * dt / ( 3 * kr );
        b_f.co_arr.at( i )[ 1 ] = c_l * kr * dt;
        return;
    };
};

////////////////////////////////////////////////////////////
//  The deposition source on the optical band:  s_ir[ 1 ] =
//  mfp * flx * E_nu * flux_scale * dt / lum_sc.

struct ppd_fld_src_t
{
    int              i_hv;
    type::float_t    e_src;    //  =  e_nu * flux_scale
                               //  / lum_sc  ( host - folded )
    type::float_t    dt;

    template< class bF_T, class bM_T >
    __device__ __forceinline__ void operator(  )
    ( const bF_T & b_f, const bM_T & b_m ) const
    {
        const auto & geo = b_f.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        const auto kap = b_m.rad.mfpa_i.at( i )[ i_hv ];
        const auto flx = b_m.rad.   flx.at( i )[ i_hv ];
        //  float_t throughout ( the user 's directive:  no
        //  FP64 intermediates on the device;  the overflow
        //  is absorbed by the rescaled constant e_src =
        //  flux_scale * e_nu / lum_sc,  the [ radiation ]
        //  flux_scale pattern ).
        b_f.s_ir.at( i )[ 1 ] = kap * flx * e_src * dt;
        return;
    };
};

////////////////////////////////////////////////////////////
//  The LTE closure with the sublimation cap ( the bench 's
//  bench_close_t verbatim ):  s_ir[ 0 ] = sig * min( E,
//  a T_cap^4 ).

struct ppd_fld_close_t
{
    type::float_t   a_c;      //  a_rad in code,  / lum_sc
    type::float_t   t_cap;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    n0  = geo.n_ceff[ 0 ];
        const int    n1  = geo.n_ceff[ 1 ];
        const int    n2  = geo.n_ceff[ 2 ];
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        type::idx_t i;
        i[ 0 ] =   t % n0;
        i[ 1 ] = ( t / n0 ) % n1;
        i[ 2 ] =   t / ( n0 * n1 );
        const type::float_t e_cap( a_c * t_cap * t_cap
                                   * t_cap * t_cap );
        auto e( b_d.mgd.fld.at( i )[ 0 ] );
        if( e > e_cap )
            e = e_cap;
        b_d.s_ir.at( i )[ 0 ] = b_d.co_arr.at( i )[ 1 ] * e;
        return;
    };
};

////////////////////////////////////////////////////////////
//  The E snapshot / change measure ( the bench 's snap_b_t
//  / chg_b_t --  the driver 's own are private ).

struct ppd_fld_snap_t
{
    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= geo.n_ceff.prod(  ) )
            return;
        type::idx_t i;
        i[ 0 ] =   t % geo.n_ceff[ 0 ];
        i[ 1 ] = ( t / geo.n_ceff[ 0 ] ) % geo.n_ceff[ 1 ];
        i[ 2 ] =   t / ( geo.n_ceff[ 0 ] * geo.n_ceff[ 1 ] );
        b_d.e_iter.at( i )[ 0 ] = b_d.mgd.fld.at( i )[ 0 ];
        return;
    };
};

struct ppd_fld_chg_t
{
    type::float_t * chg_d;
    type::float_t * fmax_d;
    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= geo.n_ceff.prod(  ) )
            return;
        type::idx_t i;
        i[ 0 ] =   t % geo.n_ceff[ 0 ];
        i[ 1 ] = ( t / geo.n_ceff[ 0 ] ) % geo.n_ceff[ 1 ];
        i[ 2 ] =   t / ( geo.n_ceff[ 0 ] * geo.n_ceff[ 1 ] );
        const auto e_n( b_d.mgd.fld.at( i )[ 0 ] );
        const auto e_o( b_d.e_iter.at( i )[ 0 ] );
        const auto df( fabs( e_n - e_o ) );
        if( df > * chg_d )
            utils::atomic_max( chg_d, df );
        if( fabs( e_n ) > * fmax_d )
            utils::atomic_max( fmax_d, fabs( e_n ) );
        return;
    };
};

//  Push the solved E to the MHD blocks ' e_fld ( real
//  cells;  e_fld carries no ghosts ).
struct ppd_fld_push_t
{
    template< class bF_T, class bM_T >
    __device__ __forceinline__ void operator(  )
    ( const bF_T & b_f, const bM_T & b_m ) const
    {
        const auto & geo = b_f.geo;
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= geo.n_ceff.prod(  ) )
            return;
        type::idx_t i;
        i[ 0 ] =   t % geo.n_ceff[ 0 ];
        i[ 1 ] = ( t / geo.n_ceff[ 0 ] ) % geo.n_ceff[ 1 ];
        i[ 2 ] =   t / ( geo.n_ceff[ 0 ] * geo.n_ceff[ 1 ] );
        b_m.e_fld.at( i )[ 0 ] = b_f.mgd.fld.at( i )[ 0 ];
        return;
    };
};

//  The E seed at init:  the analytic dust temperature
//  profile ( the disk_dust_t prescription ),  so the first
//  Picard starts at the expected state.
struct ppd_fld_seed_t
{
    type::float_t   a_c;      //  a_rad code / lum_sc
    type::float_t   T_d0, q, r0, t_cap;

    template< class bdt_T >
    __device__ __forceinline__ void operator(  )
    ( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        const int    t   = blockIdx.x * blockDim.x
                           + threadIdx.x;
        if( t >= geo.n_ceff.prod(  ) )
            return;
        type::idx_t i;
        i[ 0 ] =   t % geo.n_ceff[ 0 ];
        i[ 1 ] = ( t / geo.n_ceff[ 0 ] ) % geo.n_ceff[ 1 ];
        i[ 2 ] =   t / ( geo.n_ceff[ 0 ] * geo.n_ceff[ 1 ] );
        const auto r  = geo.x_cc( 0, i[ 0 ] );
        const auto th = geo.x_cc( 1, i[ 1 ] );
        const auto R  = r * __sin1( th );
        auto T = T_d0 * __pow1( R / r0, q );
        T = T < t_cap ? T : t_cap;
        const auto T2 = T * T;
        b_d.mgd.fld.at( i )[ 0 ] = a_c * T2 * T2;
        return;
    };
};

////////////////////////////////////////////////////////////
//  The module.

class ppd_fld_t : public rad::fld::fld_t< prx_fld_ppd_t >
{
public:                         // Types
    using sup_t = rad::fld::fld_t< prx_fld_ppd_t >;
    using f_t   = type::float_t;

public:                         // Data
    int      n_iter_t;
    f_t      t_cap;
    f_t      lum_sc;
    f_t      c_red;
    int      i_hv_heat;
    //  Bridged at read:
    f_t      a_c;        //  a_rad / lum_sc,  code
    f_t      c_code;     //  reduced c,  code
    f_t      kap_rho_fac;  //  kappa_ir * rho0 * l0
    f_t      rho_fl;       //  the radiative density floor
                           //  ( code )
    f_t      e_nu;       //  the optical photon energy,  code
    f_t      flux_sc;
    int      i_gr0, n_gr;
    //  The analytic seed parameters ( the dust provider 's ):
    f_t      T_d0, T_q, r0;

public:                         // Read / write
    __host__ ppd_fld_t(  )
    {
        n_iter_t = 30;   t_cap = 1500;   lum_sc = 1e4;
        c_red = 1e5;   i_hv_heat = 0;
        a_c = 0;   c_code = 0;
        kap_rho_fac = 0;   rho_fl = 0;
        e_nu = 0;   flux_sc = 1;
        i_gr0 = -1;   n_gr = 0;
        T_d0 = 280;   T_q = -0.5f;   r0 = 1;
        return;
    };

    virtual void read( const input & args ) override
    {
        sup_t::read( args );
        //  The host ( this module ) owns the freeze.
        freeze_byp = 1;
        s_on       = 1;
        n_iter_t = args.get< int >( "fld", "n_iter_t", 30 );
        t_cap    = args.get< f_t >( "fld", "t_cap", 1500 );
        lum_sc   = args.get< f_t >( "fld", "lum_sc", 1e4 );
        c_red    = args.get< f_t >( "fld", "c_reduce", 1e5 );
        i_hv_heat = args.get< int >
                        ( "fld", "i_hv_heat", 0 );
        //  The unit bridge ( the bench 's bridge_cgs ):
        phys::unit_t< type::float2_t > unit;
        unit.init( args );
        const type::float2_t a_cgs = 4 * phys::cgs::sigma_sb
                                   / phys::cgs::c;
        a_c = f_t( a_cgs / unit.ene0 / lum_sc );
        c_code = f_t( phys::cgs::c / unit.vel0 / c_red );
        unit_l0 = f_t( unit.l0 );
        //  The gray IR opacity:  kappa_rho per code
        //  length  =  kappa_cgs * rho_cgs * l0  =  kappa_cgs
        //  * ( rho_code * rho0 )  * l0.   The radiative
        //  density floor ( the bench 's corona lift,  1e12
        //  H / cc ):  keeps the operator 's dynamic range
        //  bounded.
        const type::float2_t kap_ir = args.get
            < type::float2_t >
            ( "fld", "kappa_ir_cgs", 30.0 );
        kap_rho_fac = f_t( kap_ir * unit.rho0 * unit.l0 );
        rho_fl = f_t( args.get< type::float2_t >
                      ( "fld", "rho_fl", 1.4e12 ) );
        flux_sc = f_t( args.get< type::float2_t >
                       ( "radiation", "flux_scale", 1 ) );
        //  The optical photon energy for the SOURCE:
        //  the deposition rate density in code is
        //    dE / dt = mfp * flx * flux_scale
        //              * hv_erg / ( ene0 * l0^3 )
        //  ( E is an energy DENSITY:  the l0^3 belongs to
        //  the photon - number density conversion;  t0
        //  cancels between the per - second flux and the
        //  per - code - time rate ).
        std::vector< type::float2_t > hv;
        args( "radiation", "hv_eV", hv );
        e_nu = f_t( hv.at( i_hv_heat ) * 1.602176634e-12
                    / unit.ene0
                    / ( unit.l0 * unit.l0 * unit.l0 ) );
        //  The Gr species slot:  the first Gr in the list;
        //  the charge states follow.
        std::vector< std::string > sps;
        args( "chemistry", "species", sps );
        i_gr0 = -1;
        for( size_t n = 0; n < sps.size(  ); ++ n )
            if( sps[ n ] == "Gr" )
            {
                i_gr0 = idx::bcc + 3 + ( int ) n;
                n_gr  = 3;      //  Gr,  Gr+,  Gr-
                break;
            }
        if( i_gr0 < 0 )
            throw std::runtime_error
                ( "ppd_fld_t:  no Gr species in"
                  " [chemistry] species" );
        T_d0 = args.get< f_t >( "dust", "T_d0", 280 );
        T_q  = args.get< f_t >( "dust", "T_pow", -0.5f );
        r0   = args.get< f_t >( "profile", "r0", 1.0f );
        return;
    };

    virtual void init_cond
    ( mesh::block::dual_t & d ) override
    {
        //  Seed E from the analytic dust profile,  host
        //  side ( the bench 's init_cond pattern;  the
        //  first push to the MHD blocks happens at the end
        //  of the first step --  the chemistry reads e_fld
        //  only after the FLD step by the step ordering ).
        auto & b_h = prx_fld_ppd_t::ref( d.h(  ) );
        const auto & geo = b_h.geo;
        for( auto i : utils::loop( geo.n_ceff ) )
        {
            const auto r  = geo.x_cc( 0, i[ 0 ] );
            const auto th = geo.x_cc( 1, i[ 1 ] );
            const auto R  = r * std::sin( th );
            auto T = T_d0 * std::pow( R / r0, T_q );
            T = T < t_cap ? T : t_cap;
            const auto T2 = T * T;
            b_h.mgd.fld.at( i )[ 0 ] = a_c * T2 * T2;
        }
        return;
    };

    //  Push the level - 0 E into the MHD blocks ' e_fld.
    void push_E( mesh::mesh_t & mesh )
    {
        auto & dev( * this->p_dev );
        for( auto & d : ( * this ) )
        {
            auto b_f = prx_fld_ppd_t::d
                ( prx_fld_ppd_t::dual( d ).d_mg( 0 ) );
            auto & b_m = disk_rad_link_t::b_mhd
                         ( d.d(  ).reg, true );
            dev.launch
                ( ppd_fld_k< decltype( b_f ),
                  std::decay_t< decltype( b_m ) >, ppd_fld_push_t >,
                  rsc_copy( b_f ), d.stream, b_f, b_m,
                  ppd_fld_push_t{ } );
        }
        return this->sync_streams( mesh );
    };

    virtual void step( mesh::mesh_t & mesh ) override
    {
        if( ! enabled )
            return;
        auto & dev( * this->p_dev );
        auto & opt = static_cast< rad::fld::fld_operation_t
                                  & >( * this->p_opt );
        const auto dt( mesh.p_cyc->dt(  ) );
        opt.d_unif = d_unif * dt;
        opt.s_unif = sig_unif * dt;
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
                opt.rbnd[ a ][ k ] = rbnd[ a ][ k ];

        ppd_fld_freeze_t fz{ c_code, dt, kap_rho_fac,
                             rho_fl };
        ppd_fld_src_t   sr{ i_hv_heat,
                            f_t( e_nu * flux_sc / lum_sc ),
                            dt };
        ppd_fld_close_t cl{ a_c, t_cap };

        for( int it = 0; it < n_iter_t; ++ it )
        {
            for( auto & d : ( * this ) )
            {
                auto & dd( prx_fld_ppd_t::dual( d ) );
                auto b_f = prx_fld_ppd_t::d( dd.d_mg( 0 ) );
                auto & b_m = disk_rad_link_t::b_mhd
                             ( d.d(  ).reg, true );
                if( it == 0 )
                {
                    //  DEBUG:  readback co_arr after the
                    //  freeze.
                    dev.launch
                    ( ppd_fld_k< decltype( b_f ),
                      std::decay_t< decltype( b_m ) >, ppd_fld_freeze_t >,
                      rsc_copy( b_f ), d.stream, b_f, b_m,
                      fz );
                    dev.launch
                    ( ppd_fld_k< decltype( b_f ),
                      std::decay_t< decltype( b_m ) >, ppd_fld_src_t >,
                      rsc_copy( b_f ), d.stream, b_f, b_m,
                      sr );
                    //  The coarse - level restriction of
                    // the frozen coefficients  ( freeze_byp
                    // skips the driver 's pass ).
                    auto b_c = b_f;
                    for( int l = -1;
                         l >= 1 - dd.n_lvl(  ); -- l )
                    {
                        auto b_l = prx_fld_ppd_t::d
                            ( dd.d_mg( l ) );
                        dev.launch
                        ( rad::fld::fld_fun2_k
                          < decltype( b_l ),
                          rad::fld::rcoarse_fun_t >,
                          rsc_copy( b_l ), d.stream,
                          b_l, b_c,
                          rad::fld::rcoarse_fun_t{ } );
                        b_c = b_l;
                    }
                }
                dev.launch
                ( rad::fld::fld_fun_k< decltype( b_f ),
                  ppd_fld_close_t >,
                  rsc_copy( b_f ), d.stream, b_f, cl );
            }
            this->sync_streams( mesh );
            //  The E snapshot for the change measure:
            for( auto & d : ( * this ) )
            {
                auto b_f = prx_fld_ppd_t::d
                    ( prx_fld_ppd_t::dual( d ).d_mg( 0 ) );
                dev.launch
                ( rad::fld::fld_fun_k< decltype( b_f ),
                  ppd_fld_snap_t >,
                  rsc_copy( b_f ), d.stream, b_f,
                  ppd_fld_snap_t{ } );
            }
            this->sync_streams( mesh );
            launch_src   ( mesh, dt );
            solve_picard ( mesh, dt );
            //  The change measure:
            dev.f_mset( p_chg_d,  0, sizeof( f_t ), 0 );
            dev.f_mset( p_fmax_d, 0, sizeof( f_t ), 0 );
            for( auto & d : ( * this ) )
            {
                auto b_f = prx_fld_ppd_t::d
                    ( prx_fld_ppd_t::dual( d ).d_mg( 0 ) );
                dev.launch
                ( rad::fld::fld_fun_k< decltype( b_f ),
                  ppd_fld_chg_t >,
                  rsc_copy( b_f ), d.stream, b_f,
                  ppd_fld_chg_t{ p_chg_d, p_fmax_d } );
            }
            this->sync_streams( mesh );
            f_t chg_h( 0 ), fmax_h( 0 );
            dev.cp( chg_h,  * p_chg_d,  1 );
            dev.cp( fmax_h, * p_fmax_d, 1 );
            dev.sync_all_streams(  );
            const f_t rel
                ( chg_h / ( fmax_h + f_t( 1e-30 ) ) );
            if( to_print && mesh.p_com->is_root(  ) )
                std::cout << "ppd_fld T - loop " << it
                          << " :  rel chg = " << rel
                          << std::endl;
            if( rel < r_tol )
                break;
        }
        push_E( mesh );
        return;
    };

protected:
    f_t unit_l0 = 0;
};

};                              // namespace interm
