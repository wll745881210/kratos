#pragma once

#include "fld_base.h"

namespace rad::fld
{

////////////////////////////////////////////////////////////
//  FLD coefficient block data:  the per - level frozen
// coefficients  ( the mirror of the nonideal  bdt_r_t,  with
// the 6 - component tensor shrunk to 2 scalars ):
//    co_arr[ 0 ]  --  D * dt   ( the dt - scaled flux -
//                           limited diffusion coefficient );
//    co_arr[ 1 ]  --  sig      ( the dt - scaled absorption
//                           diagonal,  dt c kappa_a rho );
//    co_arr[ 2..7 ]  --  the M1 Eddington tensor chi
//                           entries  (  lower - triangle
//                           xx, xy, xz, yy, yz, zz,  the
//                           fld_chi.h convention;  the
//                           FLD closure leaves them 0 ).
//  e_iter:  the Picard - outer iterate snapshot  ( level 0
// use only,  the pre - V - cycle field copy ).

template< class f_T >
struct bdt_fld_t : multigrid::data::block_data_t
{
    using super_t = multigrid::data::block_data_t;

    mesh::dat_3d_t< f_T >  co_arr;
    mesh::dat_3d_t< f_T >  e_iter;
    //  The per - cell source  ( the ppd hook:  the analytic
    // irradiation or the MCRT deposition ):
    //  s_ir[ 0 ] = the emission  a T^4  of the closure
    //  ( filled by the driver's temperature loop ),  s_ir[ 1 ]
    //  = the external deposition  ( irradiation ).
    mesh::dat_3d_t< f_T >  s_ir;
    //  The MCRT - style deposition inputs  ( n = 1 per
    // segment;  the single - segment configuration uses
    // n = 0;  the ppd hookup swaps these for the rad
    // aliases --  flx . mfp hold the MCRT convention:
    //    flx  --  the per - segment flux  ( lumin ds / V
    //            in code units,  flux_scale - rescaled );
    //    mfp  --  the per - segment absorption per unit
    //            code length  (  cross - section * number
    //            density * l0  --  density INCLUDED,  the
    //            mfpa_i convention of photoreac_prep.h ).
    mesh::dat_3d_t< f_T >  flx;
    mesh::dat_3d_t< f_T >  mfp;
    //  The M1 flux field  ( the  three  components of the
    // radiative flux  F  in code units;  rebuilt  from the
    // E  solve by the  fld_m1.h algebra ).   The closure
    // = 0  path never touches it.
    mesh::dat_3d_t< f_T >  f_arr;
    //  The base - source snapshot  (  the M1  defect -
    // correction:  each Picard round re - assembles
    // src = s_base + couple  so the explicit term never
    // accumulates ).
    mesh::dat_3d_t< f_T >  s_base;
    //  The matter temperature  (  T_m^4,  a - scaled  --
    // the two - temperature closure:  the emission is
    // sig x a T_m^4  with  T_m  an INDEPENDENT state
    // (  never  ( E / a )^1/4  --  that makes every  E
    // a fixed point  (  the  LTE identity trap  ) ).   T_m
    // relaxes per step from the local energy budget:
    //      sig ( a T_m^4 - E )  exchange  +  S  heating
    // solved implicitly with  E  ( the two - T closure ).
    mesh::dat_3d_t< f_T >  t_mat;

    //  The per - level ghost width  (  order  per non -
    // thin axis;  the setup / set_lvl  common part ).
    __host__ void n_gh_set( type::idx_t & n_gh )
    {
        const int order( mgd.order(  ) );
        for( int a = 0; a < 3; ++ a )
            n_gh[ a ] = ( geo.n_ceff[ a ] > 1 ? order : 0 );
        return;
    };

    __host__ virtual void setup
    ( const mesh::f_new_t & f_n ) override
    {
        super_t::setup( f_n );
        type::idx_t n_gh( type::idx_t::null(  ) );
        n_gh_set( n_gh );
        co_arr .init( f_n, geo.n_ceff, n_gh, 8 );
        e_iter.init( f_n, geo.n_ceff, n_gh, 1 );
        s_ir  .init( f_n, geo.n_ceff, n_gh, 2 );
        flx   .init( f_n, geo.n_ceff, n_gh, 1 );
        mfp   .init( f_n, geo.n_ceff, n_gh, 1 );
        f_arr .init( f_n, geo.n_ceff, n_gh, 3 );
        s_base.init( f_n, geo.n_ceff, n_gh, 1 );
        t_mat .init( f_n, geo.n_ceff, n_gh, 1 );
        return;
    };
    __host__ virtual void free
    ( const mesh::f_free_t & f_f ) override
    {
        co_arr .free( f_f );
        e_iter.free( f_f );
        s_ir  .free( f_f );
        flx   .free( f_f );
        mfp   .free( f_f );
        f_arr .free( f_f );
        s_base.free( f_f );
        t_mat .free( f_f );
        return super_t::free( f_f );
    };
    __host__ virtual
    std::shared_ptr< block_data_t > clone(  ) const override
    {
        return std::make_shared< bdt_fld_t >(  );
    };
    __host__ virtual void set_lvl
    ( multigrid::data::dual_t & d,
      const int & dla ) override
    {
        super_t::set_lvl( d, dla );
        type::idx_t n_gh( type::idx_t::null(  ) );
        n_gh_set( n_gh );
        const auto f_new_d = [ & ] ( const auto & c )
        {   return d.p_dev->malloc_device< char >( c ); };
        co_arr .init( f_new_d, geo.n_ceff, n_gh, 8 );
        e_iter.init( f_new_d, geo.n_ceff, n_gh, 1 );
        s_ir  .init( f_new_d, geo.n_ceff, n_gh, 2 );
        flx   .init( f_new_d, geo.n_ceff, n_gh, 1 );
        mfp   .init( f_new_d, geo.n_ceff, n_gh, 1 );
        f_arr .init( f_new_d, geo.n_ceff, n_gh, 3 );
        s_base.init( f_new_d, geo.n_ceff, n_gh, 1 );
        t_mat .init( f_new_d, geo.n_ceff, n_gh, 1 );
        return;
    };
};

};                          // namespace rad::fld
