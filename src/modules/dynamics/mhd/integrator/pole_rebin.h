#pragma once

#include "../../../../mesh/geometry/pole_rebin.h"

namespace dynamics::mhd::integrator
{
////////////////////////////////////////////////////////////
//  The polar phi - rebin of the MHD state.
// The hydro part is the  mesh::rebin_t  phi - group
// average.  The face field takes a DIVERGENCE - PRESERVING
// rebin:  the groups average the two face families that
// CROSS phi  ( the r and theta faces, whose area is phi -
// uniform, so the plain average IS the area - weighted
// one )  and rebuild the internal phi faces as the linear
// ramp that keeps every cell 's flux balance.

struct pole_rebin_hlp_t
{
    template< class bdt_T > __device__ __host__
    const auto & get_data_fld( const bdt_T & b_d ) const
    {
        return b_d.hyd.u;
    };
};

//  Phase 1:  average  B_r  over the radial faces and
//  B_theta  over the polar faces of one phi group.
template< class bdt_T, class reb_T >
__global__ void ker_pole_rebin_bf_avg
( const bdt_T b_d, const reb_T reb,
  const int step, const int side )
{
    const auto & geo( b_d.geo );
    type::idx_t i;
    i[ 0 ] = threadIdx.x + blockIdx.x * blockDim.x;
    if( i[ 0 ] >= geo.n_ceff[ 0 ] )
        return;
    i[ 1 ] = ( side == 0
               ? blockIdx.y
               : geo.n_ceff[ 1 ] - 1 - blockIdx.y );
    const int n_2( geo.n_ceff[ 2 ] );
    const int n_g( reb.num_rebin( geo, i ) );
    if( n_g <= 1 )
        return;
    using f2_t = typename decltype( b_d.fld.bf )::f_t;
    //  The band - edge taper.   A hard edge  ( the last
    //  rebinned row fully smoothed,  the next one free )
    //  leaves a  d / d theta  jump that the ambipolar
    //  operator at r_out turns into a velocity blow - up.
    //  Blend the group average in with  alpha  that fades
    //  to 1/3 at the outermost band row;  at alpha = 0
    //  nothing changes.   ( The phi faces are rebuilt from
    //  the blended balance,  see the ramp kernel,  so the
    //  divergence stays preserved. )
    const int d_th( utils::min( i[ 1 ],
                                geo.n_ceff[ 1 ] - 1 - i[ 1 ] ) );
    const f2_t alpha( reb.pole_rebin_alpha( d_th ) );
    for( int j0 = 0; j0 + n_g <= n_2; j0 += n_g )
    {
        f2_t b_r( 0 ), b_t( 0 );
        for( int k = 0; k < n_g; ++ k )
        {
            i[ 2 ] = j0 + k;
            const auto * b( b_d.fld.bf.at( i, step ) );
            b_r += b[ 0 ];
            b_t += b[ 1 ];
        }
        b_r = b_r / f2_t( n_g );
        b_t = b_t / f2_t( n_g );
        for( int k = 0; k < n_g; ++ k )
        {
            i[ 2 ] = j0 + k;
            auto * b( b_d.fld.bf.at( i, step ) );
            b[ 0 ] += alpha * ( b_r - b[ 0 ] );
            b[ 1 ] += alpha * ( b_t - b[ 1 ] );
        }
    }
    return;
}

//  Phase 2:  the internal phi faces.  After phase 1 the
//  r / theta balance  T  is group - constant;  the cell
//  divergence ( the UPPER minus LOWER flux convention )
//  vanishes when the phi flux  A_p B_phi  falls by  T
//  per face, so  B_phi  is a ramp anchored at the
//  unchanged low group face.  The high face then lands on
//  its unchanged value by the pre - rebin div - free
//  identity  sum( T ) = F_phi( high ) - F_phi( low ).
template< class bdt_T, class reb_T >
__global__ void ker_pole_rebin_bf_ramp
( const bdt_T b_d, const reb_T reb,
  const int step, const int side )
{
    const auto & geo( b_d.geo );
    type::idx_t i;
    i[ 0 ] = threadIdx.x + blockIdx.x * blockDim.x;
    if( i[ 0 ] >= geo.n_ceff[ 0 ] )
        return;
    i[ 1 ] = ( side == 0
               ? blockIdx.y
               : geo.n_ceff[ 1 ] - 1 - blockIdx.y );
    const int n_2( geo.n_ceff[ 2 ] );
    const int n_g( reb.num_rebin( geo, i ) );
    if( n_g <= 1 )
        return;
    using f2_t = typename decltype( b_d.fld.bf )::f_t;
    type::idx_t j( i );
    //  A group that wraps the WHOLE periodic ring has no
    //  free end faces  ( slot 0 and slot n_2 are the same
    //  physical face ),  so the div - free profile there is
    //  a CONSTANT -- the ramp is inadmissible.  Its slope
    //  V / A_phi  amplifies the field's own residual  (
    //  A_phi ~ r dr dth  is far smaller than the r / theta
    //  areas )  and the whole swing would land on the seam
    //  cell.  The constant is the area - weighted mean  (
    //  the areas are phi - uniform,  so the toroidal flux
    //  is conserved );  the residual uniform divergence  V
    //  is the field's own,  left unamplified.
    const int d_th( utils::min( i[ 1 ],
                                geo.n_ceff[ 1 ] - 1 - i[ 1 ] ) );
    const f2_t alpha( reb.pole_rebin_alpha( d_th ) );
    //  A fully smoothed full ring:  the constant is the
    //  only closed profile  ( the residual balance would
    //  otherwise land on the seam cell ).   A TAPERED row
    //  ( alpha < 1 ) keeps  V  proportional to the
    //  original per - cell balance,  whose sum over the
    //  ring is zero,  so the general ramp closes by
    //  itself and is the right profile there.
    if( n_g == n_2 && alpha >= f2_t( 0.999 ) )
    {
        //  CLOSED FULL RING.  Replacing the whole ring by its
        //  constant mean would zero the phi - flux difference of
        //  EVERY cell in the band while the r/theta balance is
        //  untouched -- a per-cell divergence
        //  - a_p * ( B_phi^{old}(k+1) - B_phi^{old}(k) ).
        //  The ring's own per-cell balances instead give the
        //  unique div-free profile up to a constant,  and their
        //  sum is zero for a div-free ring ( the phi terms
        //  telescope ),  so the cumulative profile closes.
        //  Subtract the mean of the balances ( the field's own
        //  residual ) so the ring closes EXACTLY and each cell
        //  keeps only that small uniform residual.
        type::idx_t q0( i ), qr( i ), qt( i );
        q0[ 2 ] = 0;   qr = q0;   ++ qr[ 0 ];   qt = q0;   ++ qt[ 1 ];
        const f2_t a_r0( geo.surface( q0, 0 ) );
        const f2_t a_r1( geo.surface( qr, 0 ) );
        const f2_t a_t0( geo.surface( q0, 1 ) );
        const f2_t a_t1( geo.surface( qt, 1 ) );
        const f2_t a_p ( geo.surface( q0, 2 ) );
        if( a_p <= 0 )
            return;
        f2_t sum_t( 0 );
        for( int k = 0; k < n_g; ++ k )
        {
            type::idx_t g_0( i ), g_r( i ), g_t( i );
            g_0[ 2 ] = k;
            g_r = g_0;   ++ g_r[ 0 ];
            g_t = g_0;   ++ g_t[ 1 ];
            sum_t = sum_t
              + a_r0 * b_d.fld.bf.at( g_0, step )[ 0 ]
              - a_r1 * b_d.fld.bf.at( g_r, step )[ 0 ]
              + a_t0 * b_d.fld.bf.at( g_0, step )[ 1 ]
              - a_t1 * b_d.fld.bf.at( g_t, step )[ 1 ];
        }
        const f2_t dm( sum_t / ( a_p * f2_t( n_g ) ) );
        type::idx_t fa( i );
        fa[ 2 ] = 0;
        f2_t b_k( b_d.fld.bf.at( fa, step )[ 2 ] );
        for( int k = 0; k < n_g - 1; ++ k )
        {
            type::idx_t g_0( i ), g_r( i ), g_t( i );
            g_0[ 2 ] = k;
            g_r = g_0;   ++ g_r[ 0 ];
            g_t = g_0;   ++ g_t[ 1 ];
            const f2_t t_k
            (   a_r0 * b_d.fld.bf.at( g_0, step )[ 0 ]
              - a_r1 * b_d.fld.bf.at( g_r, step )[ 0 ]
              + a_t0 * b_d.fld.bf.at( g_0, step )[ 1 ]
              - a_t1 * b_d.fld.bf.at( g_t, step )[ 1 ] );
            b_k += t_k / a_p - dm;
            j[ 2 ] = k + 1;
            b_d.fld.bf.at( j, step )[ 2 ] = b_k;
        }
        return;
    }
    for( int j0 = 0; j0 + n_g <= n_2; j0 += n_g )
    {
        type::idx_t f_0( i ), f_r( i ), f_t( i );
        f_0[ 2 ] = j0;
        f_r = f_0;   ++ f_r[ 0 ];
        f_t = f_0;   ++ f_t[ 1 ];
        const f2_t a_r0( geo.surface( f_0, 0 ) );
        const f2_t a_r1( geo.surface( f_r, 0 ) );
        const f2_t a_t0( geo.surface( f_0, 1 ) );
        const f2_t a_t1( geo.surface( f_t, 1 ) );
        const f2_t a_p ( geo.surface( f_0, 2 ) );
        if( a_p <= 0 )
            continue;
        //  Interior group:  the ramp is the linear
        //  interpolation between the two genuine end
        //  faces  ( for a div - free field  V / A_phi
        //  equals  - ( B_phi( hi ) - B_phi( lo ) ) / n_g ),
        //  so no amplification.
        const f2_t t_bal
        (   a_r0 * b_d.fld.bf.at( f_0, step )[ 0 ]
          - a_r1 * b_d.fld.bf.at( f_r, step )[ 0 ]
          + a_t0 * b_d.fld.bf.at( f_0, step )[ 1 ]
          - a_t1 * b_d.fld.bf.at( f_t, step )[ 1 ] );
        j = f_0;
        const f2_t b_0( b_d.fld.bf.at( j, step )[ 2 ] );
        //  PER-CELL CUMULATIVE RAMP.  The constant slope
        //  t_bal / a_p is exact only if the r/theta balance is
        //  GROUP-CONSTANT;  the band-edge taper breaks it,  so
        //  the balance varies inside the group and the constant
        //  slope leaves the residual
        //      ( r + theta )_k  -  ( r + theta )_{ f_0 }
        //  on exactly the tapered rows.  Each cell closes on its
        //  OWN balance.  The group still lands on its ORIGINAL
        //  group-boundary face:  the Phase 1 blend is
        //  sum-preserving per face row  ( sum_j alpha ( avg -
        //  old ) = 0 ),  so  sum_k t_bal_k  is unchanged and the
        //  pre-rebin div-free identity still closes the group.
        //  Only the INTERNAL faces are written  ( the group
        //  boundary at  j0 + n_g  stays anchored ).
        f2_t b_k( b_0 );
        for( int k = 0; k < n_g - 1; ++ k )
        {
            type::idx_t g_0( i ), g_r( i ), g_t( i );
            g_0[ 2 ] = j0 + k;
            g_r = g_0;   ++ g_r[ 0 ];
            g_t = g_0;   ++ g_t[ 1 ];
            const f2_t t_k
            (   a_r0 * b_d.fld.bf.at( g_0, step )[ 0 ]
              - a_r1 * b_d.fld.bf.at( g_r, step )[ 0 ]
              + a_t0 * b_d.fld.bf.at( g_0, step )[ 1 ]
              - a_t1 * b_d.fld.bf.at( g_t, step )[ 1 ] );
            b_k += t_k / a_p;
            j[ 2 ] = j0 + k + 1;
            b_d.fld.bf.at( j, step )[ 2 ] = b_k;
        }
        //  t_bal is now unused on this branch ( kept for the
        //  full-ring test above,  which needs the group balance ).
        ( void ) t_bal;
    }
    return;
}

struct mhd_rebin_t : mesh::rebin_t
{
    //  [mesh] pole_rebin_bf  ( 1 ) :  the face - field
    //  rebin;  0 leaves bf alone.
    int rebin_bf;
    //  [mesh] pole_rebin_ntaper:  the number of band rows the
    //  taper fades over ( >= 1 ).   Default 3.
    int n_taper_;

    __host__ mhd_rebin_t(  )
    {
        rebin_bf = 1;
        n_taper_ = 3;
        return;
    };

    __host__ virtual void read( const input & args )
    {
        mesh::rebin_t::read( args );
        rebin_bf = args.get< int >
                 ( "mesh", "pole_rebin_bf", rebin_bf );
        n_taper_ = args.get< int >
                 ( "mesh", "pole_rebin_ntaper", n_taper_ );
        if( n_taper_ < 1 )
            n_taper_ = 1;
        return;
    };

    //  The band - edge taper factor for the row  d_th
    //  away from the axis  ( 1 inside,  1 / 3 at the
    //  outermost rebinned row,  0 beyond ).
    template< class int_T >
    __host__ __device__ __forceinline__ type::float_t
    pole_rebin_alpha( const int_T & d_th ) const
    {
        //  BAND-EDGE TAPER  ( [mesh] pole_rebin_ntaper ).
        //  The rebuild's outermost in-band row keeps  alpha =
        //  1 / n_taper,  so the jump to the untouched rows
        //  beyond is exactly 1 / n_taper.  The amplification is
        //  at that EDGE:  a larger n_taper shrinks the jump
        //  proportionally.
        const int n_taper( n_taper_ );
        if( d_th >= rebin_lvl )
            return type::float_t( 0 );
        const type::float_t a
        ( type::float_t( rebin_lvl - d_th )
          / type::float_t( n_taper ) );
        return a < type::float_t( 1 )
             ? a : type::float_t( 1 );
    };

    //  The rebin applies only where the theta face lies
    //  ON the axis ( sin( theta ) = 0 ).   The base's
    //  at_pole also accepts a wedge whose theta_min is
    //  within  th_lim  of the pole but is NOT singular
    //  ( e.g. an  out  boundary at theta = 0.2 ):  there
    //  the phi cells keep a finite width and the
    //  averaging would only smooth a healthy region.
    template< class geo_T >
    __host__ __device__ __forceinline__ bool at_axis
    ( const geo_T & geo, const bool & rhs ) const
    {
        if( ! geo.phybnd_flag[ 1 ][ rhs ? 1 : 0 ] )
            return false;
        const auto sn( sin( geo.x_fc
                       ( 1, rhs ? geo.n_ceff[ 1 ] : 0 ) ) );
        return sn > -1e-6 && sn < 1e-6;
    };

    template< class bdt_T >
    __host__ void operator(  )
    ( const bdt_T & b_d, const dual_t & d,
      const int & step, device::base_t & dev ) const
    {
        if( rebin_lvl < 1 )
            return;
        //  The HOST geometry:  the device view's coordinate
        //  tables ( xf / xv ) are DEVICE pointers,  so any
        //  x_fc on the host view segfaults as soon as an
        //  axis is non - uniform.   The base's at_pole has
        //  the same hazard,  which is why the launches below
        //  are written out here.
        const auto & geo( d.h(  ).geo );
        //  The rebin counts phi as axis 2 and the polar
        //  faces as axis 1:  spherical only.
        if( geo.geo_type != mesh::geo_t::spherical )
            return;
        const bool lo( at_axis( geo, false ) );
        const bool hi( at_axis( geo, true  ) );
        if( ! lo && ! hi )
            return;
        pole_rebin_hlp_t hlp;
        //  The conserved - state phi - group average
        //  ( the base 's launch block ).
        const auto & u( hlp.get_data_fld( b_d ) );
        const int  n_var( u.n_int );
        using f_t = typename std::remove_cv
                  < std::remove_reference_t
                    < decltype( u ) > >::type::f_t;
        const int n_bl_y = utils::min
            ( rebin_lvl, geo.n_ceff[ 1 ] );
        const int n_th   = utils::min
            ( n_th_lim,  geo.n_ceff[ 0 ] );
        const int n_bl_x =
            ( geo.n_ceff[ 0 ] + n_th - 1 ) / n_th;
        const int n_sh   = n_var * int( sizeof( f_t ) ) * n_th;
        const auto rsc_u = std::make_tuple
            ( dim3( n_bl_x, n_bl_y ), dim3( n_th ), n_sh );
        if( lo )
            dev.launch
            ( mesh::ker_pole_rebin
              < bdt_T, pole_rebin_hlp_t, mhd_rebin_t >,
              rsc_u, d.stream, b_d, hlp, * this, step, 0 );
        if( hi )
            dev.launch
            ( mesh::ker_pole_rebin
              < bdt_T, pole_rebin_hlp_t, mhd_rebin_t >,
              rsc_u, d.stream, b_d, hlp, * this, step, 1 );
        if( rebin_bf <= 0 )
            return;
        //  The face - field launches reuse the same
        //  host - geometry dispatch extents.
        const auto rsc = std::make_tuple
            ( dim3( n_bl_x, n_bl_y ), dim3( n_th ), 0 );
        if( lo )
            dev.launch
            ( ker_pole_rebin_bf_avg< bdt_T, mhd_rebin_t >,
              rsc, d.stream, b_d, * this, step, 0 );
        if( hi )
            dev.launch
            ( ker_pole_rebin_bf_avg< bdt_T, mhd_rebin_t >,
              rsc, d.stream, b_d, * this, step, 1 );
        //  The ramp reads the averaged neighbours:  every
        //  average must land first.
        if( lo )
            dev.launch
            ( ker_pole_rebin_bf_ramp< bdt_T, mhd_rebin_t >,
              rsc, d.stream, b_d, * this, step, 0 );
        if( hi )
            dev.launch
            ( ker_pole_rebin_bf_ramp< bdt_T, mhd_rebin_t >,
              rsc, d.stream, b_d, * this, step, 1 );
        return;
    };
};

};                          // namespace dynamics::mhd::integrator
