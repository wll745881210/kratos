#pragma once

#include "../../../src/modules/dynamics/hydro/integrator/rk2.h"
#include "../../../src/utilities/mapping/mapping.h"
#include "proxy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace extension::immersed
{
////////////////////////////////////////////////////////////
// Immersed-boundary integrator

template< typename T = crtp::dummy_t >
using rk2_t = dynamics::hydro::integrator::rk2_t< T >;

template< class derived_T = crtp::dummy_t >
struct stp_t : crtp::helper_t< stp_t, derived_T, rk2_t >
{
    ////////// Types //////////
    using float_t  = type:: float_t;
    using float2_t = type::float2_t;
    using coord_t  = type:: coord_t;
    using  idx_t   = type:: idx_t;
    using dual_t   = mesh::block::dual_t;
    using super_t  = rk2_t< stp_t< derived_T > >;

    ////////// Surface flux at a coarse-fine face //////////
    // The flux saved on a refine face comes from the coarse neighbor;
    // it must not update a cell that is frozen inside the solid.
    template< class bdt_T, class eos_T > __device__ __forceinline__
    void add_flux
    ( const bdt_T & b_d, const eos_T & eos, const int & axis,
      const bool &  side, const int & step ) const
    {
        idx_t i;
        i[ axis ] = side * b_d.geo.n_ceff[ axis ];
        i[ utils::mod3( axis + 2 ) ] = blockIdx.x;
        i[ utils::mod3( axis + 1 ) ] = threadIdx.x
                        + blockIdx.z * blockDim.x;
        i[ axis ] -= side;
        if( b_d.imm.inner_mask( i ) <= 0 )
            return;
        return super_t::add_flux( b_d, eos, axis, side, step );
    };

    ////////// Hybrid cut-cell switch //////////
    // < 0 : pure ghost-cell IBM (default; the behaviour validated by
    //       the regression suite, bit-identical).
    // >= 0: cut-cell hybrid -- cells with vol_frac > alpha_min evolve
    //       from the physical fluxes (face fluxes scaled by the open
    //       fraction, plus the wall pressure term p S_b), cells below
    //       the threshold keep the image-point ghost condition.
    //       Set by hydro_t::init when [prob] cut_sub > 0.
    float_t alpha_min = -1;
    // Sharp-feature selector: the cut-cell update is used only where
    // the surface normal swings by more than acos(cut_cos) within one
    // cell ([prob] cut_angle, default 20 deg); smooth stretches keep
    // the smoother ghost conditioning.
    float_t cut_cos = 0.9396926f;
    // > 0 : log-law wall model (kinematic viscosity [m^2/s]).  Ghost
    // band cells get the tangential retardation (weight.h), cut cells
    // the wall shear stress (add_flux).
    float_t nu_wm = 0;

    // A cell is advanced by the cut-cell update only in cut mode AND
    // above the volume threshold AND flagged sharp.
    template< class bdt_T > __device__ __forceinline__
    bool cut_evolve( const bdt_T & b_d, const idx_t & i ) const
    {
        return alpha_min >= 0
            && b_d.imm.vol_frac( i ) > alpha_min
            && b_d.imm.cut_flag( i ) != 0;
    }

    ////////// Cell update: skip solid cells //////////
    template< class dat_T, class bdt_T, class eos_T > __device__
    void add_flux( dat_T & d, const bdt_T & b_d, const eos_T & eos ) const
    {
        idx_t i; int i_1d_c, i_1d_p;
        this->dispatch( i, i_1d_c, i_1d_p, d, b_d );
        if( alpha_min < 0 )     // pure ghost-cell IBM
        {
            if( b_d.imm.inner_mask( i ) > 0 )
                super_t::add_flux( d, b_d, eos );
            return;
        }
        ////////// Cut-cell update (hybrid scheme) //////////
        // Cut-evolving set: SHARP cells with vol_frac > alpha_min.
        // Everything else follows the ghost scheme: full-fluid cells
        // integrate normally, band/solid cells stay frozen.
        if( ! cut_evolve( b_d, i ) )
        {
            if( b_d.imm.inner_mask( i ) > 0 )
                super_t::add_flux( d, b_d, eos );
            return;
        }
        const float_t alpha = b_d.imm.vol_frac( i );
        using  f_t = typename decltype( b_d.hyd ):: f_t;
        using f2_t = typename decltype( b_d.hyd )::f2_t;
        dynamics::hydro::integrator::flux_info_t< f_t > f;
        auto * u     = d.u.at( i_1d_c );
        auto * u_tgt = this->load( u, b_d.hyd.u, i, d.step );
        const int n_var = b_d.hyd.n_fld(  );
        f2_t du[ 16 ];
        for( int n = 0; n < n_var; ++ n )
            du[ n ] = 0;
        const float_t p_i  = b_d.hyd.w.at( i )[ idx::ene ];
        for( int axis = 0; axis < b_d.geo.n_dim; ++ axis )
        {
            this->load_flux_sfc( f, d, b_d, i, i_1d_p, axis );
            idx_t j = i;
            -- j[ axis ];       // -a face = +a face of the neighbour
            const float_t fr_l = b_d.imm.face_frac.at( j )[ axis ];
            const float_t fr_r = b_d.imm.face_frac.at( i )[ axis ];
            // Open-fraction scaling everywhere.  Ghost-conditioned
            // neighbours (mask == 0, alpha <= alpha_min) still supply
            // their mirrored state to the reconstruction, so the flux
            // through a partially blocked face uses the physical
            // Riemann state with reduced area.  (Dropping the mass /
            // energy flux there starves cut cells next to ghost
            // cells -- the observed mass-drain instability.)
            for( int n = 0; n < n_var; ++ n )
            {
                if( f.f_a[ 0 ] > 0 && fr_l > 0 )
                    du[ n ] += fr_l * f.f_a[ 0 ] * f.flux[ 0 ][ n ];
                if( f.f_a[ 1 ] > 0 && fr_r > 0 )
                    du[ n ] -= fr_r * f.f_a[ 1 ] * f.flux[ 1 ][ n ];
            }
            // Wall pressure on the BLOCKED part of each face: the body
            // pushes the fluid away from the wall.  By the divergence
            // identity this reduces to p * S_b for the cell holding
            // the wall segment, and additionally delivers the missing
            // pressure force to full-fluid cells adjacent to a
            // partially blocked face (without it their pressure is
            // systematically drained at the surface).
            if( f.f_a[ 0 ] > 0 && fr_l < 1 )
                du[ idx::mom + axis ] += f.f_a[ 0 ] * p_i
                                       * ( 1.f - fr_l );
            if( f.f_a[ 1 ] > 0 && fr_r < 1 )
                du[ idx::mom + axis ] -= f.f_a[ 1 ] * p_i
                                       * ( 1.f - fr_r );
        }
        // Wall shear stress (log-law wall model) over the solid area
        // S_b: tau_w = rho u_tau^2, opposing the local tangential
        // velocity.  u_tau solves the equilibrium log law at the
        // exchange height y_e = |phi| + h_n/2 from the cell state.
        if( nu_wm > 0 )
        {
            const auto * S_b = b_d.imm.wall_area.at( i );
            float_t Ab2( 0 );
            for( int a = 0; a < 3; ++ a )
                Ab2 += S_b[ a ] * S_b[ a ];
            const float_t Ab = __sqrt1( Ab2 );
            if( Ab > 0 )
            {
                const auto * q_i = b_d.hyd.w.at( i );
                float_t vn( 0 );
                for( int a = 0; a < 3; ++ a )
                    vn += q_i[ idx::mom + a ] * S_b[ a ] / Ab;
                float_t vt[ 3 ], ue2( 0 );
                for( int a = 0; a < 3; ++ a )
                {
                    vt[ a ] = q_i[ idx::mom + a ] - vn * S_b[ a ] / Ab;
                    ue2    += vt[ a ] * vt[ a ];
                }
                const float_t ue = __sqrt1( ue2 );
                if( ue > 0 )
                {
                    const float_t kappa( 0.41 ), B( 5.2 );
                    float_t hn( 0 );
                    for( int a = 0; a < b_d.geo.n_dim; ++ a )
                        hn += fabsf( S_b[ a ] ) / Ab
                            * ( b_d.geo.x_cc( a, i[ a ] + 1 )
                              - b_d.geo.x_cc( a, i[ a ] ) );
                    const float_t ye = fabsf( b_d.imm.sdf( i ) )
                                     + 0.5f * hn;
                    float_t ut = ue * 0.06f;
                    for( int it = 0; it < 6; ++ it )
                    {
                        float_t arg = ye * ut / nu_wm;
                        if( arg < 1.05f )
                            arg = 1.05f;
                        const float_t la = logf( arg );
                        const float_t f  = ut * ( la / kappa + B ) - ue;
                        const float_t df = la / kappa + B + 1 / kappa;
                        ut -= f / df;
                        const float_t u_min = nu_wm / ye;
                        if( ut < u_min )
                            ut = u_min;
                    }
                    const float_t tau = q_i[ idx::rho ] * ut * ut;
                    const float_t dtv = b_d.dt / f.vol;
                    for( int a = 0; a < 3; ++ a )
                        du[ idx::mom + a ] -= dtv * tau * Ab
                                            * vt[ a ] / ue;
                }
            }
        }
        // u is the fluid-volume density, so the increment applies to
        // the fluid content alpha * V.
        for( int n = 0; n < n_var; ++ n )
            u[ n ] += du[ n ] / alpha;
        for( int n = 0; n < n_var; ++ n )
            u_tgt[ n ] = u[ n ];
        return;
    };

    ////////// Host-side drivers //////////
    // Rebuild the SDF (shifted by x_shift) and the interpolation
    // weights, storing v_wall as the wall velocity.  cut_sub > 0
    // additionally sub-samples the SDF (cut_sub points per axis) to
    // build the cut-cell geometry (vol_frac / face_frac / wall_area).
    template< class bdt_T, class sdf_T > __host__
    void update_immersed
    ( const bdt_T & b_d, const sdf_T & f_sdf, const coord_t & x_shift,
      const coord_t & v_wall, const int & bc, const float_t & T_wall,
      device::base_t & dev, const device::stream_t & stream,
      const int & cut_sub = 0 ) const;

    template< class bdt_T > __host__
    void set_immersed_weight
    ( const bdt_T & b_d, const coord_t & v_wall, const int & bc,
      const float_t & T_wall, device::base_t & dev,
      const device::stream_t & stream ) const;

    // Apply the wall condition to the interface cells of one block.
    // Launched from the boundary pass ( keeper_t::act_bnd ), i.e.
    // after the global c2p and before any flux computation.
    template< class bdt_T > __host__
    void apply_imm_bound
    ( const bdt_T & b_d, const dual_t & d,
      device::base_t & dev ) const;

    ////////// Device-side workers //////////
    template< class idx_T > __device__ __forceinline__
    void dispatch_imm( idx_T & i ) const
    {
        i[ 0 ] = blockIdx.z * blockDim.x + threadIdx.x;
        i[ 1 ] = blockIdx.x;
        i[ 2 ] = blockIdx.y;
        return;
    };

    // Bounds-safe SDF lookup: the SDF is allocated with n_gh ghost
    // cells (filled from the global field), so clamp the sample to
    // that extended range and never read out of the allocation.
    template< class bdt_T, class idx_T > __device__ __forceinline__
    float_t sdf_at( const bdt_T & b_d, const idx_T & i ) const
    {
        const auto & geo = b_d.geo;
        idx_t j;
        for( int a = 0; a < 3; ++ a )
        {
            const int lo = - geo.n_gh;
            const int hi = geo.n_ceff[ a ] + geo.n_gh - 1;
            int k = i[ a ];
            if( k < lo )
                k = lo;
            else if( k > hi )
                k = hi;
            j[ a ] = k;
        }
        return b_d.imm.sdf( j );
    };

    template< class bdt_T > __device__
    void get_imm_weight( const bdt_T & b_d, const coord_t & v_wall,
                         const int & bc, const float_t & T_wall ) const;

    template< class bdt_T > __device__ __forceinline__
    void set_imm_bound( const bdt_T & b_d ) const
    {
        idx_t i;
        dispatch_imm( i );
        if( b_d.imm.inner_mask( i ) == 0 )
        {
            // Hybrid cut-cell mode: sharp cells above the alpha
            // threshold evolve from the physical fluxes, so only the
            // smooth / nearly-solid band cells keep the ghost
            // condition.
            if( cut_evolve( b_d, i ) )
                return;
            b_d.imm.weight( i ) ( i, b_d.imm.w_snap,
                                  b_d.hyd.w );
        }
        return;
    };

    // Jacobi (two-buffer) conditioning: snapshot the primitives
    // before the sweep so that every interface cell interpolates
    // from the same pre-conditioning state. An in-place sweep
    // would be Gauss-Seidel order-dependent, which breaks the
    // mirror symmetry of symmetric bodies. The WHOLE allocation is
    // copied, ghost cells included: stencils of interface cells near
    // a block face reach into the communicated ghost layer, whose
    // values at this point are the pre-conditioning neighbor state.
    template< class bdt_T > __device__
    void snap_imm( const bdt_T & b_d ) const
    {
        const auto & w  = b_d.hyd.w;
        auto       & ws = b_d.imm.w_snap;
        // NOTE dat_3d_t::n_cell is the ALLOCATED size (ghosts included)
        const int n0 = ws.n_cell[ 0 ];
        const int n1 = ws.n_cell[ 1 ];
        const int n2 = ws.n_cell[ 2 ];
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        idx_t i;
        i[ 0 ] = t % n0 - ws.n_gh[ 0 ];  t /= n0;
        i[ 1 ] = t % n1 - ws.n_gh[ 1 ];  t /= n1;
        i[ 2 ] = t % n2 - ws.n_gh[ 2 ];
        auto       * q_s = ws.at( i );
        const auto * q_w = w .at( i );
        for( int n = 0; n < w.n_int; ++ n )
            q_s[ n ] = q_w[ n ];
        return;
    };

    // Fill the whole SDF allocation, including ghost cells, from the
    // global field. Ghost values are needed by interface stencils that
    // reach across a block boundary.
    template< class bdt_T, class sdf_T > __device__
    void set_immersed_sdf
    ( const bdt_T & b_d, const sdf_T & f_sdf,
      const coord_t & x_shift ) const
    {
        const auto & geo = b_d.geo;
        const int n0 = geo.n_ceff[ 0 ] + 2 * geo.n_gh;
        const int n1 = geo.n_ceff[ 1 ] + 2 * geo.n_gh;
        const int n2 = geo.n_ceff[ 2 ] + 2 * geo.n_gh;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        idx_t i;
        i[ 0 ] = t % n0 - geo.n_gh;  t /= n0;
        i[ 1 ] = t % n1 - geo.n_gh;  t /= n1;
        i[ 2 ] = t % n2 - geo.n_gh;
        coord_t x;
        geo.x_cc( x, i );
        for( int a = 0; a < 3; ++ a )
            x[ a ] -= x_shift[ a ];
        b_d.imm.sdf( i ) = f_sdf( x );
        return;
    };

    // Sub-cell cut geometry from the SDF (phase A of the hybrid
    // cut-cell scheme): per cell, the fluid volume fraction, the
    // open fraction of each +a face, and the boundary-segment area
    // vector.  The global SDF field is sampled at n_sub^3 sub-cell
    // centres and 2 x n_sub^2 sub-face centres, so block faces need
    // no special casing.  Cells outside the surface band (one cell
    // diagonal) are wholly fluid or solid; band cells get the
    // sampled values.  The wall area vector follows from the face
    // fractions through the divergence identity on the solid
    // polyhedron,
    //     S_b = sum_a e_a A_a ( frac_{+a} - frac_{-a} ),
    // points from the solid into the fluid, and |S_b| = A_b.
    template< class bdt_T, class sdf_T > __device__
    void get_cut_geom
    ( const bdt_T & b_d, const sdf_T & f_sdf,
      const coord_t & x_shift, const int n_sub ) const
    {
        using f_t = float_t;
        const auto & geo = b_d.geo;
        // Flat dispatch over the FULL allocation (ghosts included):
        // the cut-cell update reads the neighbour face fractions and
        // volume fractions, so the ghost ring needs the geometry too.
        const int n0 = geo.n_ceff[ 0 ] + 2 * geo.n_gh;
        const int n1 = geo.n_ceff[ 1 ] + 2 * geo.n_gh;
        const int n2 = geo.n_ceff[ 2 ] + 2 * geo.n_gh;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if( t >= n0 * n1 * n2 )
            return;
        idx_t i0;
        i0[ 0 ] = t % n0 - geo.n_gh;  t /= n0;
        i0[ 1 ] = t % n1 - geo.n_gh;  t /= n1;
        i0[ 2 ] = t % n2 - geo.n_gh;

        f_t h[ 3 ], diag2( 0 );
        for( int a = 0; a < 3; ++ a )
        {
            h[ a ] = geo.x_fc( a, i0[ a ] + 1 )
                   - geo.x_fc( a, i0[ a ]     );
            diag2 += h[ a ] * h[ a ];
        }
        const f_t band = __sqrt1( diag2 );

        auto & vf = b_d.imm.vol_frac;
        auto & ff = b_d.imm.face_frac;
        auto & wa = b_d.imm.wall_area;

        const f_t sdf_c = b_d.imm.sdf( i0 );
        if( sdf_c > band || sdf_c < - band )
        {
            const f_t s = ( sdf_c > 0 ? f_t( 1 ) : f_t( 0 ) );
            vf( i0 ) = s;
            for( int a = 0; a < 3; ++ a )
            {
                ff.at( i0 )[ a ] = s;
                wa.at( i0 )[ a ] = 0;
            }
            b_d.imm.cut_flag( i0 ) = 0;
            return;
        }

        // Fluid volume fraction from sub-cell centres.
        int n_fl = 0;
        for( int k = 0; k < n_sub; ++ k )
        for( int j = 0; j < n_sub; ++ j )
        for( int l = 0; l < n_sub; ++ l )
        {
            coord_t x;
            x[ 0 ] = geo.x_fc( 0, i0[ 0 ] )
                   + ( l + f_t( 0.5 ) ) * h[ 0 ] / n_sub;
            x[ 1 ] = geo.x_fc( 1, i0[ 1 ] )
                   + ( j + f_t( 0.5 ) ) * h[ 1 ] / n_sub;
            x[ 2 ] = geo.x_fc( 2, i0[ 2 ] )
                   + ( k + f_t( 0.5 ) ) * h[ 2 ] / n_sub;
            for( int a = 0; a < 3; ++ a )
                x[ a ] -= x_shift[ a ];
            if( f_sdf( x ) > 0 )
                ++ n_fl;
        }
        vf( i0 ) = f_t( n_fl )
                 / f_t( n_sub * n_sub * n_sub );

        // Open face fractions and the boundary area vector.
        for( int a = 0; a < 3; ++ a )
        {
            const int b = ( a + 1 ) % 3, c = ( a + 2 ) % 3;
            f_t fr[ 2 ];
            for( int pm = 0; pm < 2; ++ pm )    // 0 = -a, 1 = +a
            {
                int nfl = 0;
                for( int j = 0; j < n_sub; ++ j )
                for( int l = 0; l < n_sub; ++ l )
                {
                    coord_t x;
                    x[ a ] = geo.x_fc( a, i0[ a ] + pm );
                    x[ b ] = geo.x_fc( b, i0[ b ] )
                           + ( j + f_t( 0.5 ) ) * h[ b ] / n_sub;
                    x[ c ] = geo.x_fc( c, i0[ c ] )
                           + ( l + f_t( 0.5 ) ) * h[ c ] / n_sub;
                    for( int d = 0; d < 3; ++ d )
                        x[ d ] -= x_shift[ d ];
                    if( f_sdf( x ) > 0 )
                        ++ nfl;
                }
                fr[ pm ] = f_t( nfl ) / f_t( n_sub * n_sub );
            }
            ff.at( i0 )[ a ] = fr[ 1 ];
            wa.at( i0 )[ a ] = h[ b ] * h[ c ]
                             * ( fr[ 1 ] - fr[ 0 ] );
        }

        // Sharp-feature selector: the image-point ghost stencil
        // assumes a smooth surface (a well-defined normal over the
        // ~2h stencil reach).  Where the normal swings by more than
        // acos(cut_cos) between adjacent cell centres, mark the cell
        // for the cut-cell update; smooth stretches keep the smoother
        // ghost conditioning (hybrid by surface regularity).
        b_d.imm.cut_flag( i0 ) = 0;
        const f_t afl = vf( i0 );
        if( afl > 0 && afl < 1 )
        {
            auto nrm = [ & ] ( const coord_t & x, f_t * n )
            {
                f_t n2( 0 );
                for( int a = 0; a < geo.n_dim; ++ a )
                {
                    coord_t xp = x, xm = x;
                    xp[ a ] += h[ a ];  xm[ a ] -= h[ a ];
                    for( int d = 0; d < 3; ++ d )
                    {
                        xp[ d ] -= x_shift[ d ];
                        xm[ d ] -= x_shift[ d ];
                    }
                    n[ a ]  = f_sdf( xp ) - f_sdf( xm );
                    n2     += n[ a ] * n[ a ];
                }
                for( int a = geo.n_dim; a < 3; ++ a )
                    n[ a ] = 0;
                if( n2 > 0 )
                {
                    const f_t in = 1 / __sqrt1( n2 );
                    for( int a = 0; a < 3; ++ a )
                        n[ a ] *= in;
                }
            };
            coord_t xc;
            for( int a = 0; a < 3; ++ a )
                xc[ a ] = geo.x_cc( a, i0[ a ] );
            f_t n0[ 3 ];
            nrm( xc, n0 );
            f_t cmin( 1 );
            for( int a = 0; a < geo.n_dim; ++ a )
            for( int pm = -1; pm <= 1; pm += 2 )
            {
                coord_t xn = xc;
                xn[ a ] += pm * h[ a ];
                f_t ni[ 3 ];
                nrm( xn, ni );
                const f_t dt = n0[ 0 ] * ni[ 0 ] + n0[ 1 ] * ni[ 1 ]
                             + n0[ 2 ] * ni[ 2 ];
                cmin = dt < cmin ? dt : cmin;
            }
            if( cmin < cut_cos )
                b_d.imm.cut_flag( i0 ) = 1;
        }
        return;
    };

    template< class bdt_T > __host__ std::tuple< dim3, dim3, int >
    rsc_imm( const bdt_T & b_d ) const
    {
        const auto & n_ceff = b_d.geo.n_ceff;
        const int n_th = std::min( n_ceff[ 0 ], this->n_th_lim );
        if( n_ceff[ 0 ] % n_th != 0 )
            throw std::runtime_error
                ( "Indivisible stride: immersed::stp_t. " );
        const dim3 n_bl( n_ceff[ 1 ], n_ceff[ 2 ],
                         n_ceff[ 0 ] / n_th );
        return std::make_tuple( n_bl, dim3( n_th ), 0 );
    };

    // Flat 1-D resource covering every allocated cell, ghosts
    // included; used to fill the SDF from the global field.
    template< class bdt_T > __host__ std::tuple< dim3, dim3, int >
    rsc_imm_all( const bdt_T & b_d ) const
    {
        const auto & geo = b_d.geo;
        size_t n_tot = 1;
        for( int a = 0; a < 3; ++ a )
            n_tot *= ( geo.n_ceff[ a ] + 2 * geo.n_gh );
        const int n_th = int( std::min< size_t >
                            ( n_tot, this->n_th_lim ) );
        const int n_bl = int( ( n_tot + n_th - 1 ) / n_th );
        return std::make_tuple( dim3( n_bl ), dim3( n_th ), 0 );
    };

    // Flat 1-D resource over the w_snap allocation (same shape as
    // hyd.w): the snapshot kernel copies the whole buffer including
    // the communicated ghost cells.  n_cell is the allocated size.
    template< class bdt_T > __host__ std::tuple< dim3, dim3, int >
    rsc_imm_snap( const bdt_T & b_d ) const
    {
        const auto & ws = b_d.imm.w_snap;
        size_t n_tot = 1;
        for( int a = 0; a < 3; ++ a )
            n_tot *= ws.n_cell[ a ];
        const int n_th = int( std::min< size_t >
                            ( n_tot, this->n_th_lim ) );
        const int n_bl = int( ( n_tot + n_th - 1 ) / n_th );
        return std::make_tuple( dim3( n_bl ), dim3( n_th ), 0 );
    };
};

////////////////////////////////////////////////////////////
// Host-side drivers

template< class derived_T >
template< class bdt_T, class sdf_T > __host__
void stp_t< derived_T >::update_immersed
( const bdt_T & b_d, const sdf_T & f_sdf, const coord_t & x_shift,
  const coord_t & v_wall, const int & bc, const float_t & T_wall,
  device::base_t & dev, const device::stream_t & stream,
  const int & cut_sub ) const
{
    dev.launch( ker_imm_sdf, rsc_imm_all( b_d ), stream,
                *this, b_d, f_sdf, x_shift );
    if( cut_sub > 0 )
        dev.launch( ker_cut_geom, rsc_imm_all( b_d ), stream,
                    *this, b_d, f_sdf, x_shift, cut_sub );
    return set_immersed_weight( b_d, v_wall, bc, T_wall, dev, stream );
};

template< class derived_T >
template< class bdt_T > __host__
void stp_t< derived_T >::set_immersed_weight
( const bdt_T & b_d, const coord_t & v_wall, const int & bc,
  const float_t & T_wall, device::base_t & dev,
  const device::stream_t & stream ) const
{
    return dev.launch( ker_imm_weight, rsc_imm( b_d ), stream,
                       *this, b_d, v_wall, bc, T_wall );
};

template< class derived_T >
template< class bdt_T > __host__
void stp_t< derived_T >::apply_imm_bound
( const bdt_T & b_d, const dual_t & d, device::base_t & dev ) const
{
    dev.launch( ker_imm_snap, rsc_imm_snap( b_d ), d.stream,
                *this, b_d );
    dev.launch( ker_imm_bound, rsc_imm( b_d ), d.stream,
                *this, b_d );
    // Same event protocol as the c2p pass: the communication that
    // refreshes the ghosts after the conditioning waits on this.
    dev.event_record( d.event, d.stream );
    return;
};

////////////////////////////////////////////////////////////
// Device-side workers

template< class derived_T >
template< class bdt_T > __device__
void stp_t< derived_T >::get_imm_weight
( const bdt_T & b_d, const coord_t & v_wall, const int & bc,
  const float_t & T_wall ) const
{
    ////////// Types //////////
    using f_t   = float_t;
    using mat_T = mat_t< f_t >;

    ////////// Dispatching //////////
    idx_t i0, i;
    dispatch_imm( i0 );
    i = i0;
    const auto & geo = b_d.geo;

    ////////// SDF gradient and unit normal //////////
    coord_t dsdf;
    f_t norm( 0 );
    for( int a = 0; a < 3; ++ a )
    {
        if( a >= geo.n_dim )
        {
            dsdf[ a ] = 0;
            continue;
        }
        ++ i[ a ];
        const auto xr = geo.x_cc( a, i[ a ] );
        dsdf[ a ] = sdf_at( b_d, i );
        i[ a ] -= 2;
        const auto xl = geo.x_cc( a, i[ a ] );
        dsdf[ a ] = ( dsdf[ a ] - sdf_at( b_d, i ) ) / ( xr - xl );
        ++ i[ a ];
        norm += dsdf[ a ] * dsdf[ a ];
    }
    norm = __sqrt1( norm );
    for( int a = 0; a < 3; ++ a )
        dsdf[ a ] /= norm;

    ////////// Inside / outside / boundary cell //////////
    // SDF is in physical length.  A fluid cell's MUSCL stencil reads
    // two cells along each axis, so the deepest ghost it can read is
    // 2 * h_a * |n_a| deep (axis a, wall normal n).  Cells deeper
    // than that cover - 2 * h_n with
    //     h_n = max_a ( |n_a| * h_a )
    // (the cell's footprint along the normal) - are never read by any
    // fluid stencil: classify them fully solid (mask = -1, frozen).
    // NOTE the plain mean width is wrong on anisotropic meshes: for
    // 256 x 128 x 2 cells the mean is 0.177 vs 0.0156 in x, which
    // marked the WHOLE cylinder interior as interface (conditioned,
    // holding mirrored flow) instead of frozen solid.
    f_t h_n( 0 );
    for( int a = 0; a < 3; ++ a )
    {
        const f_t h_a = geo.x_cc( a, i0[ a ] + 1 )
                      - geo.x_cc( a, i0[ a ]     );
        const f_t dep = std::fabs( dsdf[ a ] ) * h_a;
        h_n = ( dep > h_n ? dep : h_n );
    }

    const auto sdf_c = b_d.imm.sdf( i0 );
    int mask = 0;
    if( sdf_c > 0 )
        mask = 1;
    else if( sdf_c < -2 * h_n )
        mask = -1;
    b_d.imm.inner_mask( i ) = mask;
    if( mask != 0 )
        return;

    ////////// Interpolation weights of the image cell //////////
    auto & w = b_d.imm.weight( i0 );
    for( int a = 0; a < 3; ++ a )
        w.norm[ a ] = dsdf[ a ];
    for( int a = 0; a < 3; ++ a )
        w.v_wall[ a ] = v_wall[ a ];
    w.bc     = bc;
    w.T_wall = T_wall;
    w.phi0   = sdf_c;
    w.nu_lam = nu_wm;

    const int n_dim = geo.n_dim;
    const int m_dim = ( n_dim == 3 ? 8 : 4 );

    coord_t x_c, x, dx, xi, xci;
    geo.x_cc( x_c, i0 );

    idx_t img, i_shift, di;
    for( int a = 0; a < 3; ++ a )
    {
        xi[ a ] = x_c[ a ] - sdf_c * dsdf[ a ] * 2;
        img[ a ] = i0[ a ];
        i_shift[ a ] = 0;
        if( a >= n_dim )
            break;
        // The image point may sit past a block face (an interface cell
        // within ~2 h_n of the face reflects ~4 h_n away).  Allow the
        // walk into the communicated ghost layer of hyd.w / w_snap:
        // img may leave the owned range by n_gh-1, so the stencil
        // corners img + i_shift stay inside the allocation
        // [ -n_gh, n_ceff + n_gh - 1 ].  Ghost values are the
        // pre-conditioning communicated state (Jacobi-consistent with
        // the interior reads).
        const auto & gh_w = b_d.imm.w_snap.n_gh;
        const int r_a = ( gh_w[ a ] > 0 ? gh_w[ a ] - 1 : 0 );
        const int lo  = - r_a;
        const int hi  = geo.n_ceff[ a ] - 1 + r_a;
        int n_walk = 0;
        while( ( geo.x_fc( a, img[ a ] ) > xi[ a ]
              || geo.x_fc( a, img[ a ] + 1 ) <= xi[ a ] )
            && n_walk ++ < 2 * geo.n_ceff[ a ] )
        {
            img[ a ] += ( dsdf[ a ] > 0 ? 1 : -1 );
            if( img[ a ] < lo )
                img[ a ] = lo;
            else if( img[ a ] > hi )
                img[ a ] = hi;
        }
        xci[ a ] = geo.x_cc( a, img[ a ] );
        i_shift[ a ] = ( xi[ a ] - xci[ a ] > 0 ? 1 : -1 );
    }

    // Stencil corners: the cube spans { img, img + i_shift } per
    // axis. The ghost cell itself may be one of the corners - the
    // geometric trigger of the reference's degenerate ( flag_b )
    // branch.
    bool self_s[ 8 ];
    bool degen( false );
    for( int m = 0; m < m_dim; ++ m )
    {
        utils::muldim( di, m, n_dim );
        for( int a = 0; a < 3; ++ a )
        {
            di[ a ] *= i_shift[ a ];
            i[ a ] = img[ a ] + di[ a ];
            w.w[ m ].i[ a ] = i[ a ];
        }
        self_s[ m ] = ( i[ 0 ] == i0[ 0 ] )
                   && ( i[ 1 ] == i0[ 1 ] )
                   && ( i[ 2 ] == i0[ 2 ] );
        degen = degen || self_s[ m ];
    }

    coord_t xb, rb;
    for( int a = 0; a < 3; ++ a )
    {
        xb[ a ] = x_c[ a ] - sdf_c * dsdf[ a ];
        rb[ a ] = xb[ a ] - xci[ a ];
    }

    ////////// Weights //////////
    // Weight construction follows the reference generator
    // ( combust/model.ipynb cell 6 and Piquet et al. 2016
    // Eq. 30-32 ).  The stencil is the set of 2^n_dim corners of
    // the cell that contains the mirrored (image) point.
    //   normal ( flag_b = False ): the corners form a plain
    //     trilinear stencil, so the weights are the CLOSED-FORM
    //     trilinear fractions - a partition of unity ( sum = 1
    //     exactly ), bounded in [ 0, 1 ], and no linear solve is
    //     needed.  ( Solving the raw Vandermonde instead is
    //     ill-conditioned in single precision: the monomial
    //     columns span ~1e6 in norm, which is what produced
    //     sum(w) ~ 1.25 ghosts and a mass leak at the wall. )
    //   degenerate ( flag_b = True ): the ghost cell itself is
    //     one of the corners.  ONLY its row is replaced - by the
    //     normal-gradient ( Neumann ) constraint evaluated at the
    //     boundary point xb - and the interpolant is evaluated at
    //     the ghost centre ( Piquet Eq. 31: "directly impose the
    //     value of the wall" ).  The constraint row carries no
    //     constant, so the applied weights sum to one.  Monomials
    //     are scaled by the cell widths so every entry is O( 1 )
    //     and the LU solve is single-precision safe.
    // The reference applies a second weight set ( f0 ) to momentum
    // with the flat-wall reflection built into the weights; here
    // the reflection is applied separately ( curved / moving
    // walls ), so the single set uses the f1 ( Neumann )
    // structure.  The trigger is purely geometric - no
    // conditioning threshold.
    if( ! degen )
    {
        f_t frac[ 3 ] = { 0, 0, 0 };
        for( int a = 0; a < n_dim; ++ a )
        {
            const f_t x0 = geo.x_cc( a, img[ a ] );
            const f_t x1 = geo.x_cc( a, img[ a ] + i_shift[ a ] );
            f_t t = ( xi[ a ] - x0 ) / ( x1 - x0 );
            frac[ a ] = t < 0 ? 0 : ( t > 1 ? 1 : t );
        }
        for( int m = 0; m < m_dim; ++ m )
        {
            utils::muldim( di, m, n_dim );
            w.w[ m ].w_s = 1;
            for( int a = 0; a < n_dim; ++ a )
                w.w[ m ].w_s *= ( di[ a ] ? frac[ a ]
                                            : 1 - frac[ a ] );
        }
        w.n_wght = m_dim;
        return;
    }

    ////////// Degenerate: scaled Vandermonde + LU //////////
    f_t hs[ 3 ];
    for( int a = 0; a < 3; ++ a )
        hs[ a ] = geo.x_cc( a, i0[ a ] + 1 )
                - geo.x_cc( a, i0[ a ] );

    mat_T V( m_dim );
    for( int m = 0; m < m_dim; ++ m )
    {
        for( int a = 0; a < 3; ++ a )
            i[ a ] = w.w[ m ].i[ a ];
        geo.x_cc( x, i );
        for( int a = 0; a < n_dim; ++ a )
            dx[ a ] = ( x[ a ] - xci[ a ] ) / hs[ a ];
        w.w[ m ].w_s = 0;
        if( ! self_s[ m ] )
        {
            V( m, 0 ) = 1;
            for( int a = 0; a < n_dim; ++ a )
                V( m, a + 1 ) = dx[ a ];
            if( n_dim == 3 )
            {
                for( int a = 0; a < 3; ++ a )
                    V( m, a + 4 ) = dx[ utils::mod3( a + 1 ) ]
                                  * dx[ utils::mod3( a + 2 ) ];
                V( m, 7 ) = dx.prod(  );
            }
            else
                V( m, 3 ) = dx[ 0 ] * dx[ 1 ];
        }
        else
        {
            w.w[ m ].w_s = -1;
            // Normal derivative of each ( scaled ) basis monomial
            // at the boundary point, in xci-origin coordinates.
            V( m, 0 ) = 0;
            for( int a = 0; a < n_dim; ++ a )
                V( m, a + 1 ) = dsdf[ a ] / hs[ a ];
            if( n_dim == 3 )
            {
                for( int a = 0; a < 3; ++ a )
                {
                    const int b = utils::mod3( a + 1 );
                    const int c = utils::mod3( a + 2 );
                    V( m, a + 4 ) = ( dsdf[ b ] * rb[ c ]
                                    + dsdf[ c ] * rb[ b ] )
                                  / ( hs[ b ] * hs[ c ] );
                }
                V( m, 7 ) = ( dsdf[ 0 ] * rb[ 1 ] * rb[ 2 ]
                            + dsdf[ 1 ] * rb[ 2 ] * rb[ 0 ]
                            + dsdf[ 2 ] * rb[ 0 ] * rb[ 1 ] )
                          / ( hs[ 0 ] * hs[ 1 ] * hs[ 2 ] );
            }
            else
                V( m, 3 ) = ( dsdf[ 0 ] * rb[ 1 ]
                            + dsdf[ 1 ] * rb[ 0 ] )
                          / ( hs[ 0 ] * hs[ 1 ] );
        }
    }

    // Solve V^T w = b with b the scaled monomials at the ghost
    // centre ( ~ the wall value, cf. Piquet Eq. 31 ).
    mat_T M( m_dim );
    for( int l = 0; l < m_dim; ++ l )
        for( int m = 0; m < m_dim; ++ m )
            M( l, m ) = V( m, l );
    for( int a = 0; a < n_dim; ++ a )
        dx[ a ] = ( x_c[ a ] - xci[ a ] ) / hs[ a ];
    f_t b[ 8 ];
    b[ 0 ] = 1;
    for( int a = 0; a < n_dim; ++ a )
        b[ a + 1 ] = dx[ a ];
    if( n_dim == 3 )
    {
        for( int a = 0; a < 3; ++ a )
            b[ a + 4 ] = dx[ utils::mod3( a + 1 ) ]
                       * dx[ utils::mod3( a + 2 ) ];
        b[ 7 ] = dx.prod(  );
    }
    else
        b[ 3 ] = dx[ 0 ] * dx[ 1 ];

    int lu_idx[ 8 ];
    f_t lu_v[ 8 ];
    lu_decomp( M, lu_idx, lu_v, lu_v );
    backst( b, M, lu_idx );

    // Renormalise the applied weights: the constraint row carries
    // no constant, so an exact solve gives sum = 1; guard against
    // residual single-precision drift.
    f_t sw( 0 );
    for( int m = 0; m < m_dim; ++ m )
        if( w.w[ m ].w_s >= 0 )
            sw += b[ m ];
    if( sw > 0.5 && sw < 2 )
    {
        const f_t inv( 1 / sw );
        for( int m = 0; m < m_dim; ++ m )
            b[ m ] *= inv;
    }
    for( int m = 0; m < m_dim; ++ m )
        if( w.w[ m ].w_s >= 0 )
            w.w[ m ].w_s = b[ m ];
    w.n_wght = m_dim;
    return;
};

};                              // namespace extension::immersed
