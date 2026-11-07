#pragma once

namespace extension::radiation
{
////////////////////////////////////////////////////////////
//

using type::    idx_t;
using type::  float_t;
using type::  coord_t;
using type:: float2_t;

using mesh::   f_cp_t;
using mesh::  f_new_t;
using mesh:: f_free_t;
using mesh:: f_read_t;
using mesh::f_write_t;

struct rad_t
{
    mesh::dat_3d_t< float_t > shield;
    mesh::dat_3d_t< float_t > sp_col;
    mesh::dat_3d_t< float_t > mfps_i;
    mesh::dat_3d_t< float_t > mfpa_i;
    mesh::dat_3d_t< float_t >    flx;
    //  The  VET  chi - moment  accumulator  (  2026 - 10 - 15
    // fld-ppd-integration-plan.md § 3.19;   PURELY  ADDITIVE
    // --  7  slots  per  cell:   [ 0 ]  the  weight  sum,
    // [ 1.. 6 ]  w  n_a n_b  lower  triangle  ( xx, xy, xz,
    // yy, yz, zz ),  accumulated  by  the  photon  deposit
    // under  the  integrator 's  vet_gate.   Zero  unless
    // accumulated  --  the  FLD  /  chemistry  paths  never
    // read  it ):
    mesh::dat_3d_t< float_t >  chi_m;

    __host__ __device__ __forceinline__ int & order(  )
    {
        return mfps_i.n_gh[ 0 ];
    };
    __host__ __device__ __forceinline__
    const    int & n_fld(  ) const
    {
        return flx.n_int;
    };
    __host__ int & n_fld(  )
    {
        return flx.n_int;
    };
    __host__ __device__ __forceinline__
    const    int & n_spe(  ) const
    {
        return sp_col.n_int;
    };
    __host__ int & n_spe(  )
    {
        return sp_col.n_int;
    };
    __host__ __device__ __forceinline__
    const    int & n_shl(  ) const
    {
        return shield.n_int;
    };
    __host__ int & n_shl(  )
    {
        return shield.n_int;
    };
    __host__ std::string prefix(  ) const
    {
        return "rad_";
    };
};                              // struct rad_t

// struct block_data_t : particle::radiation::block_data_t
struct block_data_t : mesh::block::base_data_t
{
    ////////// Data //////////
    rad_t  rad ;

    ////////// Memory operations //////////
    __host__ virtual void setup( const f_new_t  & f_n )
    {
        auto n_gh( type::idx_t::null(  ) );
        for( int a = 0; a < geo.n_dim; ++ a )
            n_gh[ a ] = rad.order(  );
        const auto & n_fld( rad.n_fld  (   ) );
        const auto & n_spe( rad.sp_col.n_int );
        const auto & n_shl( rad.shield.n_int );

        rad.shield.init( f_n, geo.n_ceff, n_gh, n_shl );
        rad.sp_col.init( f_n, geo.n_ceff, n_gh, n_spe );
        rad.mfps_i.init( f_n, geo.n_ceff, n_gh, n_fld );
        rad.mfpa_i.init( f_n, geo.n_ceff, n_gh, n_fld );
        rad.   flx.init( f_n, geo.n_ceff, n_gh, n_fld );
        rad.  chi_m.init( f_n, geo.n_ceff, n_gh, 7 );
        return;
    };
    __host__ virtual void free ( const f_free_t & f_f )
    {
        rad.shield.free( f_f );
        rad.sp_col.free( f_f );
        rad.mfps_i.free( f_f );
        rad.mfpa_i.free( f_f );
        rad.   flx.free( f_f );
        rad.  chi_m.free( f_f );
        return;
    };

    ////////// Data transfer //////////
    __host__ virtual void copy_input
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt_ )
    {
        auto & tgt = dynamic_cast< block_data_t & >( tgt_ );

        rad.shield.cp_to( tgt.rad.shield, f_cp );
        rad.sp_col.cp_to( tgt.rad.sp_col, f_cp );
        rad.mfps_i.cp_to( tgt.rad.mfps_i, f_cp );
        rad.mfpa_i.cp_to( tgt.rad.mfpa_i, f_cp );
        rad.   flx.cp_to( tgt.rad.   flx, f_cp );
        rad.  chi_m.cp_to( tgt.rad.  chi_m, f_cp );
        return;
    };
    __host__ virtual void copy_output
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return copy_input( f_cp, tgt );
    };

    __host__ virtual void read ( const f_read_t  & f_r )
    {
        rad.shield.read ( f_r, rad.prefix(  ) + "shield_" );
        rad.sp_col.read ( f_r, rad.prefix(  ) + "sp_col_" );
        rad.mfps_i.read ( f_r, rad.prefix(  ) + "mfps_i_" );
        rad.mfpa_i.read ( f_r, rad.prefix(  ) + "mfpa_i_" );
        rad.   flx.read ( f_r, rad.prefix(  ) + "flx_"    );
        return;
    };
    __host__ virtual void write( const f_write_t & f_w )
    {
        rad.shield.write( f_w, rad.prefix(  ) + "shield_" );
        rad.sp_col.write( f_w, rad.prefix(  ) + "sp_col_" );
        rad.mfps_i.write( f_w, rad.prefix(  ) + "mfps_i_" );
        rad.mfpa_i.write( f_w, rad.prefix(  ) + "mfpa_i_" );
        rad.   flx.write( f_w, rad.prefix(  ) + "flx_"    );
        return;
    };
};                       // struct block_data_t
};                       // namespace extension::radiation
