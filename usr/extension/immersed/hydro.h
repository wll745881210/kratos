#pragma once

#include "../../../src/modules/dynamics/hydro/hydro.h"
#include "../../../src/utilities/mapping/loop.h"
#include "../../../src/types.h"
#include "../algo/interp.h"

#include "itg.h"
#include "keeper.h"
#include "boundary.h"
#include "recon.h"

#include <stdexcept>
#include <string>
#include <cstdio>

namespace extension::immersed
{
////////////////////////////////////////////////////////////
// Hydrodynamics with immersed boundaries
//
// rec_T selects the reconstruction scheme.  The default is the
// base van Leer-harmonic PLM; a pgen may pass a derived
// reconstructor here.  Alternatively [dynamics] reconstruction
// in the input file selects one of the built-in variants at
// runtime (0 = template argument, 1 = minmod, 2 = MC).

template
< class rec_T = dynamics::hydro::reconstruct::plm_t< > >
class hydro_t : public dynamics::hydro_t
{
protected:                      // Types
    using float2_t = type::float2_t;
    using coord2_t = type::coord2_t;

    using bdk_t = extension::immersed::keeper_t;
    using prx_t = extension::immersed::prx_t
                      < dynamics::hydro::proxy_t >;
    using stp_t = extension::immersed::stp_t<  >;

    extension::interp_t< type::float_t, 3 > f_sdf;
    coord2_t body_vel;
    int imm_bc;                 // 0 = slip, 1 = no-slip
    type::float_t T_wall;       // > 0 : isothermal wall
    int cut_sub;                // > 0 : sub-samples/axis for the
                                // cut-cell geometry (hybrid scheme)
    type::float_t cut_alpha;    // volume-fraction threshold between
                                // the cut-cell and ghost treatments
    type::float_t cut_angle;    // sharp-feature selector, degrees
    type::float_t nu_lam;       // kinematic viscosity (m^2/s) for the
                                // log-law wall model
    bool wall_model;            // true: log-law wall stress active

public:                         // Functions
    virtual void read( const input & args ) override
    {
        rho0 = args.get< float2_t >( "prob", "rho0", 1 );
        pre0 = args.get< float2_t >( "prob", "pre0", 1 );
        vel0 = args.get< coord2_t >( "prob", "vel0", 0 );
        body_vel = args.get< coord2_t >( "prob", "body_vel", 0 );
        const auto bc_s = args.get< std::string >
            ( "prob", "imm_bc", "slip" );
        imm_bc = ( bc_s == "noslip" || bc_s == "no-slip" ) ? 1 : 0;
        T_wall = args.get< float2_t >( "prob", "wall_T", 0 );
        // Hybrid cut-cell scheme (DEFAULT ON): cut_sub sub-samples per
        // axis for the geometry (vol_frac / face_frac / wall_area);
        // SHARP cells (surface normal swinging by more than cut_angle
        // within one cell) with vol_frac > cut_alpha evolve from the
        // physical fluxes (face fluxes x open fraction + the wall
        // pressure term); smooth and nearly-solid band cells keep the
        // image-point ghost condition.  Set cut_sub = 0 for the pure
        // ghost-cell method.
        cut_sub   = args.get< int     >( "prob", "cut_sub",   8 );
        cut_alpha = args.get< float2_t>( "prob", "cut_alpha", 0.5 );
        cut_angle = args.get< float2_t>( "prob", "cut_angle", 20. );
        cut_write = ( cut_sub > 0 );

        // Log-law wall model (optional): retards the tangential flow
        // in the ghost band and applies a shear stress on cut cells.
        wall_model = args.get< std::string >
            ( "prob", "wall_model", "off" ) == "loglaw";
        nu_lam    = args.get< float2_t>( "prob", "nu_lam", 1.5e-5 );

        enroll_bdk< bdk_t >(  );
        // Reconstruction selection: a positive [dynamics]
        // reconstruction overrides the template argument with a
        // built-in variant (1 = minmod, 2 = monotonized central).
        const int recon = args.get< int >
            ( "dynamics", "reconstruction", 0 );
        using eos_t = dynamics::hydro::eos::adiabatic_t< >;
        if( recon == 1 )
            enroll< prx_t, stp_t, eos_t, recon::plm_minmod_t >(  );
        else if( recon == 2 )
            enroll< prx_t, stp_t, eos_t, recon::plm_mc_t     >(  );
        else
            enroll< prx_t, stp_t, eos_t, rec_T >(  );
        dynamics::base_t::read( args );
        p_bdk->enroll_phys< inflow_t< prx_t > >( "inflow" );

        const auto file = args.get< std::string >
            ( "prob", "sdf_file", "" );
        if( file.empty(  ) )
            throw std::runtime_error
                ( "No SDF file: set [prob] sdf_file. " );
        f_sdf.read_file( file, args.get< std::string >
            ( "prob", "sdf_pref", "" ) );
        f_sdf.set_nearest(  );
        return;
    };

    // Wall velocity in single precision.
    type::coord_t wall_vel(  ) const
    {
        return type::coord_t
            ( (float) body_vel[ 0 ], (float) body_vel[ 1 ],
              (float) body_vel[ 2 ] );
    };

    virtual void init
    ( const input & args, mesh::mesh_t & mesh ) override
    {
        dynamics::hydro_t::init( args, mesh );
        f_sdf.to_device( * p_dev );
        p_eos->d_floor = args.get< float2_t >
            ( "prob", "rho_floor", 1e-8 );
        p_eos->t_floor = args.get< float2_t >
            ( "prob", "tmp_floor", 1e-8 );
        // Base init already copied the host ghost-cell SDF to the
        // device, so the interpolation weights can now be built.
        const type::coord_t x_shift( 0 );
        const type::coord_t v_wall = wall_vel(  );
        for( auto & d : * this )
        {
            const auto & b_d = prx_t::d( d.d(  ) );
            auto & itg = dynamic_cast< stp_t & >( * p_itg );
            itg.alpha_min = ( cut_sub > 0 ? (type::float_t) cut_alpha
                                          : (type::float_t) -1 );
            itg.cut_cos   = std::cos( cut_angle
                                      * type::float_t( 3.14159265358979323846 )
                                      / type::float_t( 180 ) );
            itg.nu_wm     = wall_model ? (type::float_t) nu_lam
                                       : (type::float_t) 0;
            itg.update_immersed( b_d, f_sdf, x_shift, v_wall,
                                 imm_bc, T_wall, * p_dev, d.stream,
                                 cut_sub );
        }
        // The immersed wall condition is applied from the boundary
        // pass, so that every block is conditioned after the global
        // c2p and before any flux is evaluated; cross-block stencils
        // then read the conditioned interface state.
        auto & bdk = dynamic_cast< keeper_t & >( * p_bdk );
        bdk.act_imm = [ this ]( mesh::mesh_t &, const int & )
        {
            for( auto & d : ( * this ) )
            {
                const auto & b_d = prx_t::d( d.d(  ) );
                auto & itg = dynamic_cast< stp_t & >( * p_itg );
                itg.apply_imm_bound( b_d, d, * p_dev );
            }
            // The conditioning kernels run on the block streams and
            // record d.event; the boundary re-communication that
            // follows waits on those events ( same protocol as the
            // c2p pass ), so no host-side sync is needed here.
            return;
        };
        return;
    };

    // Ghost-cell data lives on the host before copy_h2d, so fill
    // the SDF there; the weights are built on the device in init().
    virtual void init_cond( mesh::block::dual_t & d ) override
    {
        dynamics::hydro_t::init_cond( d );
        auto & b_h = prx_t::ref ( d.h(  ) );
        const auto & geo = d.b_w.get(  ).geo;
        // SDF alignment guard: the cell centres coincide with SDF
        // nodes only when h_cell / dx_sdf is a positive EVEN
        // integer; an odd ratio ( including 1 ) puts the centres at
        // node midpoints, where the 8-node trilinear average
        // collapses |phi| across a ROUGH surface and degenerates
        // the interface stencils ( the 512^3 incident: dt collapsed
        // to 1e-22 within a cycle ).  Smooth ANALYTIC SDFs tolerate
        // midpoint sampling, so warn rather than abort.
        {
            static bool warned = false;
            for( int a = 0; a < 3 && ! warned; ++ a )
            {
                const float h_a = geo.dx0[ a ];
                if( h_a <= 0 || f_sdf.dx[ a ] <= 0 )
                    continue;         // inactive / non-uniform axis
                const float r  = h_a / f_sdf.dx[ a ];
                const int   ri = (int)( r * 0.5f + 0.5f ) * 2;
                const float dr = r - (float)ri;
                if( ri < 2 || dr * dr > 1e-6f )
                {
                    warned = true;
                    fprintf( stderr,
                        "WARNING: immersed SDF alignment: h/dx_s = "
                        "%.4f along axis %d is not an even integer "
                        ">= 2; cell centres sit at SDF node midpoints."
                        "  Smooth analytic SDFs tolerate this, but a "
                        "rough MESH-derived SDF collapses |phi| at "
                        "the surface ( regenerate with gen_* n_h >= "
                        "2 for this grid ).\n", r, a );
                }
            }
        }
        for( auto i : utils::loop( geo.n_ceff ) )
        {
            type::coord_t x;
            geo.x_cc( x, i );
            b_h.imm.sdf( i ) = f_sdf( x );
        }
        return;
    };

    // Rigid-body translation: before each stage loop, move the
    // body by body_vel * t and refit the SDF and the weights.
    virtual void step( mesh::mesh_t & mesh ) override
    {
        if( body_vel.norm(  ) > 0 )
        {
            const auto t = mesh.p_cyc->t;
            const type::coord_t x_shift
                ( (float) ( body_vel[ 0 ] * t ),
                  (float) ( body_vel[ 1 ] * t ),
                  (float) ( body_vel[ 2 ] * t ) );
            const type::coord_t v_wall
                ( (float) body_vel[ 0 ], (float) body_vel[ 1 ],
                  (float) body_vel[ 2 ] );
            for( auto & d : * this )
            {
                const auto & b_d = prx_t::d( d.d(  ) );
                auto & itg = dynamic_cast< stp_t & >( * p_itg );
                itg.update_immersed( b_d, f_sdf, x_shift, v_wall,
                                     imm_bc, T_wall, * p_dev, d.stream,
                                     cut_sub );
            }
        }
        return dynamics::base_t::step( mesh );
    };
};
};                              // namespace extension::immersed
