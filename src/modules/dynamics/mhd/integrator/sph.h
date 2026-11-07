#pragma once

#include "rk2.h"
#include "pole_rebin.h"
#include "../../../../utilities/types/my_type_traits.h"

namespace dynamics::mhd::integrator
{
////////////////////////////////////////////////////////////
//  On the polar axis the radial edges of all phi cells
//  coincide, so the phi-loop of emfs around the zero-area
//  pole face telescopes only if the radial emf on the
//  axis is phi-independent.  A generic ( non-axisymmetric )
//  flow computes a different E_r per phi cell there, which
//  leaks div( B ) into the pole-adjacent cells.  Average
//  ee[ 0 ] over the phi ring on the pole faces ( the
//  counterpart of Athena++'s PolarSingleEMF ).  Launched
//  from sph_t::intg_flx, after the E-field exchange.
//  LIMITATION: the average is over the block-local phi
//  range; blocks split along phi need a cross-block ring
//  average ( mesh::rebin_t - style machinery ) instead.

template< class bdt_T > __global__
void pole_emf_k( const bdt_T b_d )
{
    using   f_t = typename decltype( b_d.fld )::f_t;
    const auto & geo ( b_d.geo );
    if( ! geo.phybnd_flag[ 1 ][ blockIdx.y ] )
        return;
    type::idx_t i = { int( blockIdx.x ),
                      blockIdx.y ? geo.n_ceff[ 1 ] : 0, 0 };
    //  Same 1e-6 snap as the edge lengths in edg_f: only
    //  faces within roundoff of a multiple of pi qualify.
    const auto sn = __sin1( geo.x_fc( 1, i[ 1 ] ) );
    if( sn <= -1.e-6 || sn >= 1.e-6 )
        return;
    __dyn_shared__( f_t, sh );
    const int n2 ( geo.n_ceff[ 2 ] );
    f_t acc ( 0 );
    for( int i2 = threadIdx.x; i2 < n2; i2 += blockDim.x )
    {
        i[ 2 ] = i2;
        acc += b_d.fld.ee.at( i )[ 0 ];
    }
    sh[ threadIdx.x ] = acc;
    __syncthreads(  );
    for( int s = blockDim.x / 2; s > 0; s >>= 1 )
    {
        if( threadIdx.x < s )
            sh[ threadIdx.x ] += sh[ threadIdx.x + s ];
        __syncthreads(  );
    }
    const f_t mean ( sh[ 0 ] / n2 );
    //  Include the seam slot i2 == n2 ( read by the last
    //  phi-face update; the E-field exchange keeps it
    //  consistent with the periodic image of slot 0 ).
    for( int i2 = threadIdx.x; i2 <= n2; i2 += blockDim.x )
    {
        i[ 2 ] = i2;
        b_d.fld.ee.at( i )[ 0 ] = mean;
    }
    return;
}
////////////////////////////////////////////////////////////
//  Debug:  dump one cell 's CT ingredients  ( the face
//  values / areas and the four corner EMFs per face )
//  right before the CT update.   Knob:
//  [nonideal] dbg_cell = i0 i1 i2.
template< class bdt_T >
__global__ void dbg_ee_k( const bdt_T b_d,
                          const type::idx_t tgt )
{
    if( blockIdx.x != 0 || threadIdx.x != 0 )
        return;
    type::idx_t i( tgt );
    const auto & w( b_d.hyd.w.at( i ) );
    //  w  layout:  0 rho,  1 p,  2..4 v,  5..7 bcc.
    printf( "DBGEE (%d,%d,%d) dt=%.3e rho=%.4e p=%.3e "
            "v=[%.3e %.3e %.3e]\n",
            ( int ) i[ 0 ], ( int ) i[ 1 ], ( int ) i[ 2 ],
            ( double ) b_d.dt, ( double ) w[ 0 ],
            ( double ) w[ 1 ], ( double ) w[ 2 ],
            ( double ) w[ 3 ], ( double ) w[ 4 ] );
    for( int a = 0; a < 3; ++ a )
    {
        const int a1( utils::mod3( a + 1 ) );
        const int a2( utils::mod3( a + 2 ) );
        type::idx_t p2( i ), p1( i );
        ++ p2[ a2 ];
        ++ p1[ a1 ];
        const auto A ( b_d.geo.surface( i, a ) );
        const auto A1( b_d.geo.surface( i, a1 ) );
        const auto A2( b_d.geo.surface( i, a2 ) );
        printf( "  a%d bf=%.4e A=%.4e | E%d(i)=%.3e E%d(i+e%d)=%.3e "
                " E%d(i)=%.3e E%d(i+e%d)=%.3e | A%d=%.3e A%d=%.3e\n",
                a, ( double ) b_d.fld.bf.at( i )[ a ],
                ( double ) A,
                a1, ( double ) b_d.fld.ee.at( i  )[ a1 ],
                a1, a2, ( double ) b_d.fld.ee.at( p2 )[ a1 ],
                a2, ( double ) b_d.fld.ee.at( i  )[ a2 ],
                a2, a1, ( double ) b_d.fld.ee.at( p1 )[ a2 ],
                a1, ( double ) A1, a2, ( double ) A2 );
    }
    return;
}

////////////////////////////////////////////////////////////
// MHD integrator in the spherical polar coordinates
// ( r, theta, phi ).  It extends the Cartesian MHD RK2
// integrator the same way the hydrodynamic spherical
// integrator extends the hydrodynamic one:

//  1.  The conserved azimuthal momentum is the TOTAL
//      angular momentum  rho * v_phi * r * sin(theta ).
//      The radial and polar fluxes of the momentum are
//      rescaled by the ratio of the metric factors so
//      that the transport of the angular momentum is
//      consistent with the finite-volume geometry.
//      In contrast to the source terms in Athena++,
//      the angular momentum transport here needs NO
//      face-averaged flux source terms: the rescaling
//      of the total momentum flux ( which already
//      includes the magnetic tension ) carries the
//      correct radial / polar advection of the angular
//      momentum.
//  2.  The geometric source terms are assembled from
//      CELL-CENTERED quantities, except for the
//      thermal + magnetic pressure parts, which use
//      the exact geometric moments of the cells ( the
//      same split as the hydrodynamic spherical
//      integrator ):
//         d_mom_r  = dt [ < 2 p* / r > + ( T_3 + T_4 ) ]
//         d_mom_th = dt [ < p* d(theta) > + ( T_5 - T_6 ) ]
//      where p* = p + B^2/2, T_i are pointwise kinetic
//      and magnetic terms, and < ... > are the exact
//      volume moments.  This keeps v_r*v_theta - like
//      terms from ever reaching face centers.
//  3.  The constrained transport uses the physical
//      edge lengths and face areas of the spherical
//      cells, so that the discrete div( B ) = 0
//      identity holds exactly the same way as in the
//      Cartesian case.

template< class derived_T = crtp::dummy_t >
struct sph_t : crtp::helper_t< sph_t, derived_T, rk2_t >
{
    ////////// CRTP functions //////////
    __crtp_def_self__( sph_t, derived_T );
    using super_t
        = crtp::helper_t< sph_t, derived_T, rk2_t >;

    ////////// Data //////////
    type::float_t omega_0;
    //  The polar phi - rebin  ( [mesh] pole_rebin_lvl ).
    mhd_rebin_t      rebin;
    //  [nonideal] dbg_cell = i0 i1 i2 :  the per - CT
    //  ingredient dump at that cell  ( -1 = off ).
    bool              dbg_on;
    type::idx_t        dbg_i;
    //  [mesh] pole_emf  ( 1 = default ):  the polar EMF
    //  regularisation  ( pole_emf_k,  Athena++'s
    //  PolarSingleEMF ).  0 skips it.
    int          pole_emf_on;

    ////////// Functions //////////
    __host__ sph_t(  ) : super_t(  ), omega_0( 0 ),
                        pole_emf_on( 1 ) {  };

    __host__ virtual void init
    ( const input & args, const::dynamics::base_t & mod )
    {
        omega_0 = args.get < type::float_t >
                ( "dynamics", "omega_0", 0 );
        rebin.read( args );
        pole_emf_on = args.get< int >
                    ( "mesh", "pole_emf", 1 );
        dbg_on = false;
        dbg_i  = -1;
        if( args.found( "nonideal", "dbg_cell" ) )
        {
            args( "nonideal", "dbg_cell", dbg_i );
            dbg_on = true;
        }
        return super_t::init( args, mod );
    };

    //  The default Cartesian flux_to_du must never be
    //  used with spherical polar coordinates: it does
    //  not know about the metric factors.
    template< class  f2_T, class bdt_T,
              class fif_T, class idx_T, int I = 1 >
    __device__ __forceinline__ f2_T  flux_to_du
    ( const fif_T & f, const bdt_T & b_d, const idx_T & i ,
      const int &  ax, const int   & n )  const
    {
        static_assert( I * I < 0,
            " Do NOT use default flux_to_du with spherical"
            " polar coordinates !!!" );
        return f2_T ( 0 );
    };

    __device__ __host__ const super_t & get_super(  ) const
    {
        return static_cast< const super_t & > ( * this );
    };

    //  Polar emf regularisation ( CRTP hook shadowing the
    //  base no-op ).  Applied between the emf exchange and
    //  the CT update:  the fine-coarse edge exchange can
    //  overwrite pole edges with non-averaged values,  so
    //  this must run after act_flx and before field_intg.
    //  Called by intg_flx and by intg_field_only ( the
    //  operator-split diffusion sweep ).
    template< class bdt_T >
    __host__ void emf_fix
    ( const bdt_T & b_d, const dual_t & d,
      device::base_t & dev )              const
    {
        if( pole_emf_on
         && (  b_d.geo.phybnd_flag[ 1 ][ 0 ]
            || b_d.geo.phybnd_flag[ 1 ][ 1 ] ) )
        {
            using f_t = typename decltype( b_d.fld )::f_t;
            const int n_th ( 256 );
            dev.launch( pole_emf_k< bdt_T >,
                        std::make_tuple
                        ( dim3( b_d.geo.n_ceff[ 0 ], 2, 1 ),
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
        if( dbg_on )
            dev.launch
            ( dbg_ee_k< bdt_T >,
              std::make_tuple( dim3( 1 ), dim3( 1 ), 0 ),
              d.stream, b_d, dbg_i );
        super_t::intg_flx( b_d, d, eos, step, dev );
        //  The polar phi - rebin of the freshly written
        //  slice  ( the  rk2  load / add_flux  convention:
        //  the updated state lives in slice 0 after the
        //  last substep,  else in slice 1 ).
        const int s( step == this->time_order(  ) - 1 ? 0
                                                    : step + 1 );
        rebin( b_d, d, s, dev );
        return;
    };

////////////////////////////////////////////////////////////
//  Physical length of the edge running along axis "ax"
//  at the corner "i".  For the spherical coordinates:
//      edge_r   : d( r )
//      edge_th  : r_f * d( theta )
//      edge_phi : r_f * sin( theta_f ) * d( phi )
//  where r_f and theta_f are evaluated at the FACE
//  positions of the two axes the edge does NOT run
//  along ( the indexing convention of the EMF array ).
////////////////////////////////////////////////////////////
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto  edg_f
    ( const int &  ax, const idx_T & i,
      const geo_T & geo )                       const
    {
        if( ax == 0 )
            return geo.dx_f( 0, i );
        if( ax == 1 )
            return geo.x_fc( 0, i[ 0 ] )
                 * geo.dx_f( 1, i );
        //  The phi-edge length carries sin( theta_f ):
        //  snap it to zero at the pole.  No finite-
        //  precision coordinate has sin( pi ) == 0 (
        //  sinf( pi_f ) = -8.7e-8 ); a non-zero pole-edge
        //  length breaks the exact CT telescoping because
        //  the zero-area pole face is skipped.  The 1e-6
        //  threshold is ~ 10x the float artifact and far
        //  below any physical grid scale.
        const auto sn = __sin1( geo.x_fc( 1, i[ 1 ] ) );
        if( sn > -1.e-6 && sn < 1.e-6 )
            return geo.x_fc( 0, i[ 0 ] ) * 0.f
                 * geo.dx_f( 2, i );
        return  geo.x_fc( 0, i[ 0 ] ) * sn
              * geo.dx_f( 2, i );
    };

    //  Face area of the thin ( single-cell ) axis "a".
    //  Only the ( r, theta )-active 2-D case reaches
    //  here ( thin phi, a == 2 ): the exact phi-face
    //  area, which the mesh generator does not store
    //  for thin axes.
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
    //  the cell-centred counterparts of edg_f.  i[ ax ]
    //  is the FACE the segment is centred on; the other
    //  axes carry cell positions.
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto  edg_d
    ( const int &  ax, const idx_T & i,
      const geo_T & geo )                       const
    {
        auto j( i ); -- j[ ax ];
        const auto w = ( geo.dx_f( ax, j )
                       + geo.dx_f( ax, i ) ) / 2;
        if( ax == 0 )
            return w;
        const auto r = geo.x_cc( 0, i[ 0 ] );
        if( ax == 1 )
            return r * w;
        return r * __sin1( geo.x_cc( 1, i[ 1 ] ) ) * w;
    };

////////////////////////////////////////////////////////////
//  Conserved to primitive corrections in the spherical
//  polar coordinates.  Ported from the hydrodynamic
//  spherical integrator; the MHD total azimuthal
//  momentum flux ( including the magnetic tension )
//  is rescaled the same way as the hydrodynamic one.
////////////////////////////////////////////////////////////
    template< class  f2_T, class bdt_T,
              class fif_T, class idx_T, class  x_T >
    __device__ __forceinline__ f2_T  flux_to_du_sph
    ( f2_T * u, const fif_T & f, const bdt_T & b_d,
      const idx_T & i, const x_T & x,
      const int & ax, const int   & n )        const
    {
        const auto & geo = b_d.geo;
        const auto r   (           x[ 0 ]   ) ;
        const auto s_th(   __sin1( x[ 1 ] ) ) ;
        const auto vf0 ( omega_0 * r * s_th ) ;
        f2_T res ( 0 );
        for( int j = 0; j < 2 ; ++ j )
        {
            if( f.f_a[ j ] <= 0 )
                continue;
            const auto f_side( 1 - 2 * j );
            //  theta-momentum, radial faces: rescale by
            //  r_f / r_c -> the r^3 ( slice angular
            //  momentum ) transport form.  CONSISTENT ONLY
            //  with the tensor-form src_geo source below
            //  ( whose -rho v_r v_th / r + B_r B_th / r
            //  parts are REMOVED because the rescaled flux
            //  already carries T_rth / r ).  Keeping both
            //  double-counts T_rth/r and a uniform
            //  Cartesian vector field is not held.
            if( ax == 0 && n == idx::mom + 1 )
                res += f.f_a[ j ] * f.flux[ j ] [ n ] / r
                    *  geo.x_fc( 0, i[ 0 ] + j ) * f_side;
            else if( n == idx::mom + 2 )
            {
                //  Azimuthal momentum: the conserved
                //  variable is the total angular momentum
                //  rho * v_phi * r * sin( theta ).  The
                //  whole flux ( including the vf0 shift )
                //  is rescaled by the metric factors.
                auto vf0_face = omega_0;
                if( ax == 0 )
                    vf0_face *= geo.x_fc( 0, i[ 0 ] + j )
                        * s_th;
                else if( ax == 1 )
                    vf0_face *= r * __sin1
                         ( geo.x_fc( 1, i[ 1 ] + j ) );
                else
                    vf0_face *= r * s_th;

                auto f_mphi = f.flux[ j ][ n ]
                    + f.flux[ j ][ idx::rho ] * vf0_face;
                if( ax == 0 )
                    f_mphi *= geo.x_fc( 0, i[ 0 ] + j ) / r;
                else if( ax == 1 )
                    f_mphi *= 1 / s_th
                      * __sin1( geo.x_fc( 1, i[ 1 ] + j ) );
                res  += f.f_a[ j ] * f_mphi * f_side;
            }
            else if( n == idx::ene )
            {
                //  Energy: convert the rotating-frame
                //  flux to the inertial frame.
                auto vf( omega_0 );
                auto f_ene = f.flux[ j ][ n ];
                if( ax == 0 )
                    vf *= geo.x_fc( 0, i[ 0 ] + j ) * s_th;
                else if( ax == 1 )
                    vf *= r * __sin1
                         ( geo.x_fc( 1, i[ 1 ]  + j ) );
                else
                    vf *= r * s_th;
                f_ene
                    += f.flux[ j ][ idx::rho ] * vf * vf / 2
                    +  f.flux[ j ][ idx::mom + 2 ]  * vf ;
                res += f.f_a [ j ] * f_ene * f_side;
            }
            else
                res += f.f_a[ j ] * f.flux[ j ][ n ]
                    * f_side;
        }
        if( omega_0 != 0 )
        {
            if( n == idx::rho )
            {
                u[ idx::mom + 2 ] -= res * vf0 ;
                u[ idx::ene ] += res * ( vf0 * vf0 / 2 );
            }   // from -res * vf0**2 / 2 but with dp_phi...
            else if( n == idx::mom + 2 )
                u[ idx::ene ] -= res * vf0;
        }
        u[ n ] += res;
        return res;
    };

////////////////////////////////////////////////////////////
//  Geometric source terms, assembled from CELL-CENTERED
//  kinetic / magnetic terms and exact pressure moments.
////////////////////////////////////////////////////////////
    template< class f2_T, class f_T,
              class bdt_T, class idx_T >
    __device__ __forceinline__ void  src_geo
    ( f2_T * u, const f_T * w, const bdt_T & b_d,
      const idx_T &  i , const coord_t & x )  const
    {
        const auto & geo = b_d.geo;
        auto r_l  = geo.x_fc( 0, i[ 0 ]     );
        auto r_r  = geo.x_fc( 0, i[ 0 ] + 1 );
        auto th_l = geo.x_fc( 1, i[ 1 ]     );
        auto th_r = geo.x_fc( 1, i[ 1 ] + 1 );
        static const f_T pi ( 3.14159265358979323846 );
        th_l = ( th_l < 0   ? 0    : th_l );
        th_r = ( th_r > pi  ? pi   : th_r );

        const auto  s_th ( __sin1( x[ 1 ] ) );
        const auto  c_th ( __cos1( x[ 1 ] ) );

        const auto * vel ( w + idx::mom );
        const auto * b_f ( w + idx::bcc );

        //  LAB-frame azimuthal velocity
        const auto vf0 = s_th * x[ 0 ] * omega_0;
        const auto v12 = vel[ 1 ] * vel[ 1 ];
        const auto vf  = vel[ 2 ] + vf0;
        const auto v22 = vf * vf;

        const auto rho    ( w[ idx::rho ] );
        const auto & pre  ( w[ idx::ene ] );
        const auto dt_rho ( b_d.dt * rho );

        const auto b12 = b_f[ 1 ] * b_f[ 1 ];
        const auto b22 = b_f[ 2 ] * b_f[ 2 ];

        //  Total ( thermal + magnetic ) pressure
        const auto pre_t
            ( pre + ( b12 + b22
                    + b_f[ 0 ] * b_f[ 0 ] ) / 2 );

        const auto rl2 = r_l * r_l;
        const auto rr2 = r_r * r_r;
        const auto cm0 = 3 * ( rr2 - rl2 ) / 2.f
                       / ( rr2 * r_r - rl2 * r_l );

        coord_t d_mom;
        //  Radial: exact pressure moment ( T_1 + T_2 )
        d_mom[ 0 ] = b_d.dt * cm0 * 2 * pre_t;
        //  Polar: exact pressure moment ( T_3 )
        d_mom[ 1 ] = pre_t * b_d.dt * cm0
                   * ( __sin1( th_r ) - __sin1( th_l ) )
                   / ( __cos1( th_l ) - __cos1( th_r ) );
        //  Radial: pointwise kinetic + magnetic
        d_mom[ 0 ] += dt_rho * ( v12 + v22 ) / x[ 0 ];
        d_mom[ 0 ] -= b_d.dt * ( b12 + b22 ) / x[ 0 ];
        //  Polar: pointwise kinetic + magnetic, TENSOR
        //  form: consistent with the r^3 theta-momentum
        //  flux above, the T_rth / r parts ( -rho v_r v_th
        //  and + B_r B_th ) are carried by the rescaled
        //  flux and must NOT appear in the source.
        d_mom[ 1 ] += dt_rho / x[ 0 ]
             * ( v22 * c_th / s_th );
        d_mom[ 1 ] -= b_d.dt / x[ 0 ]
             * ( b22 * c_th / s_th );
        d_mom[ 2 ] = 0;

        for( int a = 0; a < 3; ++ a )
            u[ idx::mom + a ] += d_mom[ a ];
        return;
    };

////////////////////////////////////////////////////////////
//  Flux divergence with the spherical corrections
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
                self.flux_to_du_sph
                    ( u, f, b_d, i, x, a, n );
        }
        self.src_geo( u, b_d.hyd.w.at( i ), b_d, i, x );
        for( int n = 0; n < n_var; ++ n )
            u_tgt[ n ] = u[ n ];
        return;
    };

////////////////////////////////////////////////////////////
//  Refluxing on the refine faces ( spherical version )
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
            flux_to_du_sph( u, f, b_d, i, x, axis, n );
        return;
    };

};
};// namespace dynamics::mhd::integrator
