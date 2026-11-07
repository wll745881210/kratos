#pragma once

#include "../../../src/mesh/module_base/module_base.h"
#include "linalg.h"

namespace extension::immersed
{
// Whether the cut-cell geometry fields are written / synced on
// output.  Set by hydro_t::read when [prob] cut_sub > 0.
inline bool cut_write = false;

////////////////////////////////////////////////////////////
// Kernels

template< class itg_T, class bdt_T > __global__
void ker_imm_weight( const itg_T itg, const bdt_T bdt,
                     const type::coord_t v_wall, const int bc,
                     const type::float_t T_wall )
{
    return itg.get_imm_weight( bdt, v_wall, bc, T_wall );
};

template< class itg_T, class bdt_T > __global__
void ker_imm_bound( const itg_T itg, const bdt_T bdt )
{
    return itg.set_imm_bound( bdt );
};

template< class itg_T, class bdt_T > __global__
void ker_imm_snap( const itg_T itg, const bdt_T bdt )
{
    return itg.snap_imm( bdt );
};

template< class itg_T, class bdt_T, class sdf_T > __global__
void ker_imm_sdf( const itg_T itg, const bdt_T bdt,
                  const sdf_T f_sdf, const type::coord_t x_shift )
{
    return itg.set_immersed_sdf( bdt, f_sdf, x_shift );
};

template< class itg_T, class bdt_T, class sdf_T > __global__
void ker_cut_geom( const itg_T itg, const bdt_T bdt,
                   const sdf_T f_sdf, const type::coord_t x_shift,
                   const int n_sub )
{
    return itg.get_cut_geom( bdt, f_sdf, x_shift, n_sub );
};

////////////////////////////////////////////////////////////
// Proxy block: adds immersed-boundary data to the hydro block

template< class sup_T, class wgt_T = weight_t >
struct prx_t : public sup_T
{
    ////////// Types //////////
    using sup_blk_t = typename sup_T::bdt_t;

    struct bdt_t : sup_blk_t
    {
        struct
        {
            mesh::dat_3d_t< int     >   inner_mask;
            mesh::dat_3d_t< wgt_T   >   weight;
            mesh::dat_3d_t< float_t >   sdf;
            // Pre-conditioning snapshot of hyd.w for the Jacobi
            // (order-free) immersed-boundary sweep.
            mesh::dat_3d_t< float_t >   w_snap;
            // Cut-cell geometry (phase A of the hybrid scheme):
            //   vol_frac : fluid volume fraction alpha in [0,1]
            //   face_frac: open fraction of the +a face, per axis
            //   wall_area: boundary-segment area vector A_b n_b
            //              (divergence identity on the face
            //              fractions), m^2.
            mesh::dat_3d_t< float_t >   vol_frac;
            mesh::dat_3d_t< float_t >   face_frac;
            mesh::dat_3d_t< float_t >   wall_area;
            // 1 where the surface normal swings sharply within one
            // cell (vertices / thin edges): only those cells use the
            // cut-cell update; smooth stretches keep the (smoother)
            // image-point ghost conditioning.
            mesh::dat_3d_t< int     >   cut_flag;
        }   imm;

        __host__ __device__ bdt_t(  ) {  };

        ////////// Allocation //////////
        __host__ virtual void setup( const mesh::f_new_t & f_new ) override
        {
            const auto & n_ceff = this->geo.n_ceff;
            const int n_gh = this->geo.n_gh;
            imm.inner_mask.init( f_new, n_ceff, 0,    1, 1 );
            imm.weight    .init( f_new, n_ceff, 0,    1, 1 );
            // The SDF needs ghost cells so that interface stencils
            // that cross a block boundary read the global field
            // instead of a clamped edge value.
            imm.sdf       .init( f_new, n_ceff, n_gh, 1, 1 );
            // w_snap mirrors the allocation of hyd.w (same ghosts,
            // same field count).
            {
                type::idx_t n_gh_w( type::idx_t::null(  ) );
                const auto order = this->hyd.order(  );
                for( int a = 0; a < 3; ++ a )
                    n_gh_w[ a ] = ( n_ceff[ a ] > 1 ? order : 0 );
                imm.w_snap .init( f_new, n_ceff, n_gh_w,
                                  this->hyd.n_fld(  ), 1 );
            }
            // Ghost ring included: the cut-cell update reads the
            // neighbour volume/face fractions across the cell bounds.
            imm.vol_frac .init( f_new, n_ceff, n_gh, 1, 1 );
            imm.face_frac.init( f_new, n_ceff, n_gh, 3, 1 );
            imm.wall_area.init( f_new, n_ceff, n_gh, 3, 1 );
            imm.cut_flag .init( f_new, n_ceff, n_gh, 1, 1 );
            return sup_blk_t::setup( f_new );
        };

        __host__ virtual void write( const mesh::f_write_t & f_w ) override
        {
            imm.sdf       .write( f_w, "sdf_"  );
            imm.inner_mask.write( f_w, "mask_" );
            if( cut_write )
            {
                imm.vol_frac .write( f_w, "vfr_" );
                imm.face_frac.write( f_w, "ffr_" );
                imm.wall_area.write( f_w, "war_" );
                imm.cut_flag .write( f_w, "flg_" );
            }
            return sup_blk_t::write( f_w );
        };

        __host__ virtual void read( const mesh::f_read_t & f_r ) override
        {
            imm.sdf.read( f_r, "sdf_" );
            if( cut_write )
            {
                imm.vol_frac .read( f_r, "vfr_" );
                imm.face_frac.read( f_r, "ffr_" );
                imm.wall_area.read( f_r, "war_" );
                imm.cut_flag .read( f_r, "flg_" );
            }
            return sup_blk_t::read( f_r );
        };

        __host__ virtual void free( const mesh::f_free_t & f_free ) override
        {
            imm.inner_mask.free( f_free );
            imm.weight    .free( f_free );
            imm.sdf       .free( f_free );
            imm.w_snap    .free( f_free );
            imm.vol_frac  .free( f_free );
            imm.face_frac .free( f_free );
            imm.wall_area .free( f_free );
            imm.cut_flag  .free( f_free );
            return sup_blk_t::free( f_free );
        };

        __host__ virtual void copy_input
        ( const mesh::f_cp_t & f_cp, mesh::block::base_data_t & tgt ) override
        {
            imm.sdf.cp_to( tgt.derive< bdt_t >(  ).imm.sdf, f_cp );
            return sup_blk_t::copy_input( f_cp, tgt );
        };

        __host__ virtual void copy_output
        ( const mesh::f_cp_t & f_cp, mesh::block::base_data_t & tgt ) override
        {
            imm.sdf       .cp_to( tgt.derive< bdt_t >(  ).imm.sdf,       f_cp );
            imm.inner_mask.cp_to( tgt.derive< bdt_t >(  ).imm.inner_mask, f_cp );
            if( cut_write )
            {
                imm.vol_frac .cp_to( tgt.derive< bdt_t >(  )
                                     .imm.vol_frac,  f_cp );
                imm.face_frac.cp_to( tgt.derive< bdt_t >(  )
                                     .imm.face_frac, f_cp );
                imm.wall_area.cp_to( tgt.derive< bdt_t >(  )
                                     .imm.wall_area, f_cp );
                imm.cut_flag .cp_to( tgt.derive< bdt_t >(  )
                                     .imm.cut_flag,  f_cp );
            }
            return sup_blk_t::copy_output( f_cp, tgt );
        };
    };

    ////////// Block-data accessors //////////
    template< class bd0_T > __host__
    static bdt_t & ref( const bd0_T & bd )
    {
        return dynamic_cast< bdt_t & >
             ( const_cast< bd0_T & > ( bd ) );
    };
    template< class bd0_T > __host__
    static bdt_t & d_i( const bd0_T & bd ) { return ref( bd ); };
    template< class bd0_T > __host__
    static bdt_t & d  ( const bd0_T & bd ) { return ref( bd ); };
};

};                              // namespace extension::immersed
