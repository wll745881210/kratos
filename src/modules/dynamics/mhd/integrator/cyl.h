#pragma once

#include "rk2.h"
#include "../../../../utilities/types/my_type_traits.h"

namespace dynamics::mhd::integrator
{
////////////////////////////////////////////////////////////
// MHD integrator in the cylindrical coordinates
// ( r, phi, z ).  It extends the Cartesian MHD RK2
// integrator the same way the spherical polar one does:

//  1.  The conserved azimuthal momentum is the angular
//      momentum  rho * v_phi * r:  its radial flux is
//      rescaled by r_f / r_c at application time
//      ( flux_to_du_cyl ),  so the scheme telescopes
//      exactly over the radial faces.  The matching
//      -rho v_r v_phi / r source part is carried by the
//      rescaled flux and must NOT appear in src_geo.

//  2.  The geometric source ( src_geo ) is assembled
//      from CELL-CENTERED kinetic / magnetic terms:
//      radial momentum gets ( p* + rho v_phi^2
//      - B_phi^2 ) / r with the total pressure
//      p* = p + B^2 / 2;  the cell radius
//      ( r_l + r_r ) / 2 is the exact volume-averaged
//      < 1 / r > moment.  The phi / z equations have no
//      geometric source.

//  3.  Constrained transport uses the cylindrical edge
//      lengths ( edg_f:  phi edges carry r_f ) and the
//      generic zero-area guard of the base intg_field.

////////////////////////////////////////////////////////////
//  On the rotation axis ( r = 0 ) the z-running edges of
//  all phi cells coincide, so the phi-loop of emfs around
//  the zero-area r face there telescopes only if the axial
//  emf on the axis is phi-independent.  Average ee[ 2 ]
//  over the phi ring on the axis ( counterpart of
//  pole_emf_k for the spherical polar pole ).  Launched
//  from cyl_t::intg_flx, after the E-field exchange.
//  LIMITATION: the average is over the block-local phi
//  range; blocks split along phi need a cross-block ring
//  average ( mesh::rebin_t - style machinery ) instead.

template< class bdt_T > __global__
void cyl_axis_emf_k( const bdt_T b_d )
{
    using   f_t = typename decltype( b_d.fld )::f_t;
    const auto & geo ( b_d.geo );
    type::idx_t i = { 0, 0, int( blockIdx.x ) };
    //  Only a domain whose r-minus face sits on the axis
    //  ( within roundoff of r = 0 ) needs the average.
    const auto dr = geo.x_fc( 0, 1 ) - geo.x_fc( 0, 0 );
    if( geo.x_fc( 0, 0 ) <= -1.e-6 * dr ||
        geo.x_fc( 0, 0 ) >=  1.e-6 * dr )
        return;
    __dyn_shared__( f_t, sh );
    const int n1 ( geo.n_ceff[ 1 ] );
    f_t acc ( 0 );
    for( int i1 = threadIdx.x; i1 < n1; i1 += blockDim.x )
    {
        i[ 1 ] = i1;
        acc += b_d.fld.ee.at( i )[ 2 ];
    }
    sh[ threadIdx.x ] = acc;
    __syncthreads(  );
    for( int s = blockDim.x / 2; s > 0; s >>= 1 )
    {
        if( threadIdx.x < s )
            sh[ threadIdx.x ] += sh[ threadIdx.x + s ];
        __syncthreads(  );
    }
    const f_t mean ( sh[ 0 ] / n1 );
    //  Include the seam slot i1 == n1 ( read by the last
    //  phi-face update; the E-field exchange keeps it
    //  consistent with the periodic image of slot 0 ).
    for( int i1 = threadIdx.x; i1 <= n1; i1 += blockDim.x )
    {
        i[ 1 ] = i1;
        b_d.fld.ee.at( i )[ 2 ] = mean;
    }
    return;
}

template< class derived_T = crtp::dummy_t >
struct cyl_t : crtp::helper_t< cyl_t, derived_T, rk2_t >
{
    ////////// CRTP functions //////////
    __crtp_def_self__( cyl_t, derived_T );
    using super_t
        = crtp::helper_t< cyl_t, derived_T, rk2_t >;

    ////////// Data //////////
    type::float_t omega_0;

    ////////// Functions //////////
    __host__ cyl_t(  ) : super_t(  ), omega_0( 0 ) {  };

    __host__ virtual void init
    ( const input & args, const::dynamics::base_t & mod )
    {
        omega_0 = args.get < type::float_t >
                ( "dynamics", "omega_0", 0 );
        return super_t::init( args, mod );
    };

    //  On-axis emf regularisation ( CRTP hook shadowing
    //  the base no-op ),  mirroring the spherical pole
    //  treatment.  Applied between the emf exchange and
    //  the CT update;  called by intg_flx and by
    //  intg_field_only ( the operator-split diffusion
    //  sweep ).
    template< class bdt_T >
    __host__ void emf_fix
    ( const bdt_T & b_d, const dual_t & d,
      device::base_t & dev )              const
    {
        if( b_d.geo.phybnd_flag[ 0 ][ 0 ] )
        {
            using f_t = typename decltype( b_d.fld )::f_t;
            const int n_th ( 256 );
            dev.launch( cyl_axis_emf_k< bdt_T >,
                        std::make_tuple
                        ( dim3( b_d.geo.n_ceff[ 2 ], 1, 1 ),
                          dim3( n_th, 1, 1 ),
                          n_th * sizeof( f_t ) ),
                        d.stream, b_d );
        }
        return;
    };

    template< class bdt_T, class eos_T >
    __host__ void intg_flx
    ( const bdt_T & b_d, const dual_t & d,
      const eos_T & eos, const int    & step,
      device::base_t & dev )              const
    {
        emf_fix( b_d, d, dev );
        super_t::intg_flx( b_d, d, eos, step, dev );
        return;
    };

    //  The default Cartesian flux_to_du must never be
    //  reached in cylindrical coordinates.
    template< class  f2_T, class bdt_T,
              class fif_T, class idx_T, int I = 1 >
    __device__ __forceinline__ f2_T  flux_to_du
    ( const fif_T & f, const bdt_T & b_d,
      const idx_T & i , const int   & ax,
      const int   &  n )                 const
    {
        static_assert( I * I < 0,
            " Do NOT use default flux_to_du with"
            " cylindrical coordinates !!!" );
    };

    __device__ __host__ const super_t & get_super(  ) const
    {
        return static_cast< const super_t & > ( * this );
    };

    //  Constrained-transport edge lengths in cylindrical
    //  coordinates:  only the phi edges carry the
    //  cylindrical factor r_f.
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto  edg_f
    ( const int &  ax, const idx_T & i,
      const geo_T & geo )                       const
    {
        if( ax == 1 )
            return geo.x_fc( 0, i[ 0 ] )
                 * geo.dx_f( 1, i );
        return geo.dx_f( ax, i );
    };

    //  Face area of the thin ( single-cell ) axis "a".
    //  Only the ( r, phi )-active 2-D case needs an
    //  override ( thin z, a == 2 ):  the z-face area
    //  r_c * dr * d_phi,  which the mesh generator does
    //  not store for thin axes.  A thin phi axis has the
    //  plain area dr * dz.
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto  sfc_thin
    ( const int &  a, const idx_T & i,
      const geo_T & geo )                       const
    {
        if( a == 2 )
        {
            const auto r_l = geo.x_fc( 0, i[ 0 ]     );
            const auto r_r = geo.x_fc( 0, i[ 0 ] + 1 );
            return ( r_r * r_r - r_l * r_l ) / 2
                 * ( geo.x_fc( 1, i[ 1 ] + 1 )
                   - geo.x_fc( 1, i[ 1 ]     ) );
        }
        const int ax_1 ( utils::mod3( a + 1 ) );
        const int ax_2 ( utils::mod3( a + 2 ) );
        return geo.dx_f( ax_1, i ) * geo.dx_f( ax_2, i );
    };

    //  Dual edge lengths ( non-ideal current kernel ):
    //  cell-centred counterparts of edg_f.  i[ ax ] is
    //  the FACE the segment is centred on; the other
    //  axes carry cell positions.  Only the azimuthal
    //  segment carries a metric factor ( r_c ).
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto  edg_d
    ( const int &  ax, const idx_T & i,
      const geo_T & geo )                       const
    {
        auto j( i ); -- j[ ax ];
        const auto w = ( geo.dx_f( ax, j )
                       + geo.dx_f( ax, i ) ) / 2;
        if( ax == 1 )
            return geo.x_cc( 0, i[ 0 ] ) * w;
        return w;
    };

////////////////////////////////////////////////////////////
//  Flux divergence with the cylindrical correction.
//  The azimuthal momentum is transported as the
//  LAB-frame total angular momentum:  its flux is
//  augmented by the frame advection omega * lever_f
//  * F( rho ) and rescaled by lever_f / lever_c on
//  radial faces ( the lever arm r is constant along
//  phi- and z-faces, so only r-faces are rescaled ),
//  then the frame part is removed again below from
//  the mass divergence.  The energy row picks up the
//  rotational kinetic-energy flux and frame work.
//  With omega_0 = 0 all augmentations are exact +0
//  and the scheme is bit-neutral.
////////////////////////////////////////////////////////////
    template< class  f2_T, class bdt_T,
              class fif_T, class idx_T, class  x_T >
    __device__ __forceinline__ f2_T  flux_to_du_cyl
    ( f2_T * u, const fif_T & f, const bdt_T & b_d,
      const idx_T & i, const x_T & x,
      const int & ax, const int   & n )        const
    {
        const auto & geo( b_d.geo );
        const auto r  (           x[ 0 ] );
        const auto vf0( omega_0 * r      );
        f2_T res ( 0 );
        for( int j = 0; j < 2 ; ++ j )
        {
            if( f.f_a[ j ] <= 0 )
                continue;
            const auto f_side( 1 - 2 * j );
            if( n == idx::mom + 1 )
            {
                auto vff( omega_0 );
                if( ax == 0 )
                    vff *= geo.x_fc( 0, i[ 0 ] + j );
                else
                    vff *= r;
                auto f_mphi = f.flux[ j ][ n ]
                            + f.flux[ j ][ idx::rho ] * vff;
                if( ax == 0 )
                    f_mphi *= geo.x_fc( 0, i[ 0 ] + j ) / r;
                res += f.f_a[ j ] * f_mphi * f_side;
            }
            else if( n == idx::ene )
            {
                auto vff( omega_0 );
                if( ax == 0 )
                    vff *= geo.x_fc( 0, i[ 0 ] + j );
                else
                    vff *= r;
                const auto f_ene = f.flux[ j ][ n ]
                    + f.flux[ j ][ idx::rho ]
                        * vff * vff / 2
                    + f.flux[ j ][ idx::mom + 1 ] * vff;
                res += f.f_a[ j ] * f_ene * f_side;
            }
            else
                res += f.f_a[ j ] * f.flux[ j ] [ n ]
                    *  f_side;
        }
        if( omega_0 != 0 )
        {
            if( n == idx::rho )
            {
                u[ idx::mom + 1 ] -= res * vf0;
                u[ idx::ene     ] += res
                    * ( vf0 * vf0 / 2 );
            }
            else if( n == idx::mom + 1 )
                u[ idx::ene ] -= res * vf0;
        }
        u[ n ] += res;
        return res;
    };

////////////////////////////////////////////////////////////
//  Geometric source term in cylindrical coordinates
//  ( see the header comment ).
////////////////////////////////////////////////////////////
    template< class f2_T, class f_T,
              class bdt_T, class idx_T >
    __device__ __forceinline__ void  src_geo
    ( f2_T * u, const f_T * w, const bdt_T & b_d,
      const idx_T &  i , const coord_t & x )  const
    {
        const auto & geo = b_d.geo;
        const auto r_c = ( geo.x_fc( 0, i[ 0 ] + 1 )
                         + geo.x_fc( 0, i[ 0 ]     ) ) / 2;

        const auto * vel ( w + idx::mom );
        const auto * b_f ( w + idx::bcc );

        const auto rho ( w[ idx::rho ] );
        //  Total ( thermal + magnetic ) pressure
        const auto pre_t
            ( w[ idx::ene ]
            + ( b_f[ 0 ] * b_f[ 0 ] + b_f[ 1 ] * b_f[ 1 ]
              + b_f[ 2 ] * b_f[ 2 ] ) / 2 );

        //  Centripetal term uses the LAB-frame azimuthal
        //  velocity:  ( v_phi + omega * r )^2 expands to
        //  the curvature + Coriolis + centrifugal forces.
        const auto vf  = vel[ 1 ] + omega_0 * r_c;
        const auto v22 = vf * vf;
        const auto b22 = b_f[ 1 ] * b_f[ 1 ];

        coord_t d_mom;
        d_mom[ 0 ] = b_d.dt / r_c
                   * ( pre_t + rho * v22 - b22 );
        d_mom[ 1 ] = 0;
        d_mom[ 2 ] = 0;

        for( int a = 0; a < 3; ++ a )
            u[ idx::mom + a ] += d_mom[ a ];
        return;
    };

////////////////////////////////////////////////////////////
//  Flux divergence with the cylindrical corrections
////////////////////////////////////////////////////////////
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void  add_flux
    ( dat_T & d, const bdt_T & b_d,
      const eos_T & eos )                      const
    {
        const auto & geo = b_d.geo;
        const auto & self = get_self(  );

        int i[ 3 ], i_1d_c, i_1d_p;
        self.dispatch( i, i_1d_c, i_1d_p, d, b_d );

        using f_t = typename decltype( b_d.hyd )::f_t;
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
                self.flux_to_du_cyl
                    ( u, f, b_d, i, x, a, n );
        }
        self.src_geo( u, b_d.hyd.w.at( i ), b_d, i, x );
        for( int n = 0; n < n_var; ++ n )
            u_tgt[ n ] = u[ n ];
        return;
    };

////////////////////////////////////////////////////////////
//  Refluxing on the refine faces ( cylindrical version )
////////////////////////////////////////////////////////////
    template< class bdt_T, class eos_T >
    __device__ __forceinline__ void  add_flux
    ( const bdt_T & b_d, const eos_T & eos,
      const int & axis, const bool & side,
      const int & step )                       const
    {
        idx_t i;
        i[ axis ] = side * b_d.geo.n_ceff[ axis ];
        i[ utils::mod3( axis + 2 ) ] = blockIdx.x;
        i[ utils::mod3( axis + 1 ) ] =
            threadIdx.x + blockIdx.z * blockDim.x;
        using f_t = typename decltype( b_d.hyd )::f_t;
        dynamics::hydro::integrator::flux_info_t< f_t > f;
        i[ axis ] -= side;
        f.vol = b_d.geo.volume( i );
        const auto s
            ( step == b_d.hyd.u.n_step - 1 ? 0 : step + 1 );
        auto * u = b_d.hyd.u.at( i, s );
        coord_t x;
        b_d.geo.x_cc( x, i );
        i[ axis ] += side;
        f.flux[ side ] = b_d.hyd.f[ axis ].at( i );
        f.f_a[ side ] =
            b_d.geo.surface( i, axis ) * b_d.dt / f.vol;
        f.flux[ ! side ] = nullptr;
        f.f_a[ ! side ] = -1;
        for( int n = 0; n < b_d.hyd.n_fld(  ); ++ n )
            flux_to_du_cyl( u, f, b_d, i, x, axis, n );
        return;
    };

};
};// namespace dynamics::mhd::integrator
