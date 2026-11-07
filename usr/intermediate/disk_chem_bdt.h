#pragma once

#include "../../src/modules/chemistry/chem_mhd/mhd.h"
#include "../../usr/extension/radiation/block/block_data.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  Chemistry - carrying MHD block data WITH the radiation
//  arrays ( usr_ext / ppd / design.md decision 5,  ledger
//  cont. 43 ).   The extension ( chem_therm ) solver's
//  prep_dev reads b_d.rad.sp_col / flx UNCONDITIONALLY,  so
//  the block data must carry a rad member even before
//  disk_radiation ( T8 ) exists:  here the arrays are only
//  placeholders --  sp_col sized by the species count and
//  zero - filled,  flx / mfps_i / mfpa_i with n_fld = 0 and
//  shield with n_shl = 0 ( zero - size allocations are
//  safe:  cudaMalloc( 0 ) / cudaMallocHost( 0 ) return
//  success with nullptr ).   T8 sizes and fills them for
//  real;  nothing in this file needs to change then.

struct bdt_chem_t : chemistry::mhd::block_data_t
{
    ////////// Types //////////
    using super_t = chemistry::mhd::block_data_t;

    ////////// Data //////////
    extension::radiation::rad_t rad;
    //  The saturation - region flag ( disk_chem_sat.h ):
    //  1 = the chemistry solver skips this cell.   Written
    //  per step by disk_sat_k on device;  not in the I/O.
    mesh::dat_3d_t< type::float_t > chem_skip;
    //  The FLD dust - radiation energy density ( disk_fld.h,
    //  T11 ):  written per step by the ppd_fld_t module (
    //  d2d from the MG level - 0 field ),  read by
    //  disk_dust_t::eval in the E - mode.   Real cells
    //  only ( no ghost use ).
    mesh::dat_3d_t< type::float_t > e_fld;
    //  The non - ideal diffusivities,  written per step by
    //  disk_eta_out_k ( disk_eta.h ):  slots [ 0 ] = eta_O,
    //  [ 1 ] = eta_A ( code units,  cap - and - ring / funnel
    //  included --  EXACTLY the values the diffusion kernels
    //  use ).   The primary M1 diagnostic ( Elsasser numbers
    //  need no offline re - derivation ).   [nonideal]
    //  eta_out = 0 turns the fill off ( the write then
    //  carries zeros ).
    mesh::dat_3d_t< type::float_t > eta_out;


    ////////// Memory operations //////////
    __host__ virtual void setup
    ( const mesh::f_new_t & f_n ) override
    {
        super_t::setup( f_n );
        chem_skip.init( f_n, geo.n_ceff, type::idx_t::null(  ),
                        1, 1 );
        e_fld   .init( f_n, geo.n_ceff, type::idx_t::null(  ),
                        1, 1 );
        eta_out .init( f_n, geo.n_ceff, type::idx_t::null(  ),
                        2, 1 );
        auto n_gh( type::idx_t::null(  ) );
        for( int a = 0; a < geo.n_dim; ++ a )
            n_gh[ a ] = rad.order(  );
        rad.shield.init( f_n, geo.n_ceff, n_gh,
                         rad.n_shl(  ) );
        rad.sp_col.init( f_n, geo.n_ceff, n_gh,
                         rad.n_spe(  ) );
        rad.mfps_i.init( f_n, geo.n_ceff, n_gh,
                         rad.n_fld(  ) );
        rad.mfpa_i.init( f_n, geo.n_ceff, n_gh,
                         rad.n_fld(  ) );
        rad.   flx.init( f_n, geo.n_ceff, n_gh,
                         rad.n_fld(  ) );
        return;
    };
    __host__ virtual void free
    ( const mesh::f_free_t & f_f ) override
    {
        chem_skip.free( f_f );
        e_fld   .free( f_f );
        eta_out .free( f_f );
        rad.shield.free( f_f );
        rad.sp_col.free( f_f );
        rad.mfps_i.free( f_f );
        rad.mfpa_i.free( f_f );
        rad.   flx.free( f_f );
        return super_t::free ( f_f );
    };

    ////////// Data transfer //////////
    __host__ virtual void copy_input
    ( const mesh::f_cp_t       & f_cp,
      mesh::block::base_data_t & tgt_ ) override
    {
        auto & tgt
            = dynamic_cast< bdt_chem_t & >( tgt_ );
        chem_skip.cp_to( tgt.chem_skip, f_cp );
        e_fld   .cp_to( tgt.e_fld   , f_cp );
        eta_out .cp_to( tgt.eta_out , f_cp );
        rad.shield.cp_to( tgt.rad.shield, f_cp );
        rad.sp_col.cp_to( tgt.rad.sp_col, f_cp );
        rad.mfps_i.cp_to( tgt.rad.mfps_i, f_cp );
        rad.mfpa_i.cp_to( tgt.rad.mfpa_i, f_cp );
        rad.   flx.cp_to( tgt.rad.   flx, f_cp );
        return super_t::copy_input( f_cp, tgt );
    };
    __host__ virtual void copy_output
    ( const mesh::f_cp_t       & f_cp,
      mesh::block::base_data_t & tgt  ) override
    {
        return copy_input( f_cp, tgt );
    };

    ////////// I/O //////////
    __host__ virtual void read
    ( const mesh::f_read_t & f_r ) override
    {
        rad.shield.read( f_r, rad.prefix(  ) + "shield_" );
        rad.sp_col.read( f_r, rad.prefix(  ) + "sp_col_" );
        rad.mfps_i.read( f_r, rad.prefix(  ) + "mfps_i_" );
        rad.mfpa_i.read( f_r, rad.prefix(  ) + "mfpa_i_" );
        rad.   flx.read( f_r, rad.prefix(  ) + "flx_"    );
        return super_t::read ( f_r );
    };
    __host__ virtual void write
    ( const mesh::f_write_t & f_w ) override
    {
        e_fld.write( f_w, "rad_e_fld_" );
        eta_out.write( f_w, "ni_eta_" );
        rad.shield.write( f_w, rad.prefix(  ) + "shield_" );
        rad.sp_col.write( f_w, rad.prefix(  ) + "sp_col_" );
        rad.mfps_i.write( f_w, rad.prefix(  ) + "mfps_i_" );
        rad.mfpa_i.write( f_w, rad.prefix(  ) + "mfpa_i_" );
        rad.   flx.write( f_w, rad.prefix(  ) + "flx_"    );
        return super_t::write( f_w );
    };
};

//  The proxy needs no copy - constructing view:  the rad
//  arrays live IN this block data,  so the default
//  reference - returning proxy suffices.
using prx_chem2_t = mesh::block::proxy_t< bdt_chem_t >;

};                              // namespace interm
