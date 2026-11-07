#pragma once

#include <cmath>
#include <vector>

//  Non-ideal ( Ohmic + Hall + ambipolar ) magnetic
//  diffusion for the kratos MHD module, after the
//  Athena++ / Microthena field_diffusion design:
//      e_dif = eta_O * J + eta_H * ( J x B^ )
//            + eta_A * ( J - ( J . B^ ) B^ )
//  with the cell-centred Hall parameterisation
//      eta_H = hall_l * | B_cc | / sqrt( rho_cc )
//  ( in code units  Hall = eta_H / | B |  is the
//  ion-skin length;  this is the natural form when
//  the chemistry module is not providing the
//  conductivity tensor ).
//
//  Structure:
//    - The current density J is evaluated at the
//      edges of the computational cells ( the curl
//      dual of the CT update:  J is the circulation
//      of B around the faces dual to each edge,
//      normalised by the volume-centred dual area --
//      the discrete adjoint of the emf curl, so the
//      diffusion operator preserves div B = 0 to
//      machine precision in all geometries ).
//    - e_dif is added into the emf array ee in
//      calc_flx, *before* the SMR / periodic edge
//      exchange, so the total emf stays globally
//      consistent and the pole / axis averaging
//      kernels act on the total emf.
//    - The magnetic energy flux of the diffusion (
//      S = e_dif x B ) carries BOTH the transport and the
//      dissipation of the magnetic energy; depositing it
//      into u[ ene ] leaves the gas with exactly
//      dt eta J^2 ( without it the gas would also absorb
//      the transport term dt div S as spurious heat ).
//      Same structure as Athena++'s AddPoyntingFlux -> IEN.
//      [nonideal] mg_cc_ene ( default 2 ) selects the
//      deposit form;  0 turns it off,  and the Ohmic and
//      ambipolar heating powers are accumulated in
//      b_d.fld.heat ( Hall does not heat:
//      J . ( J x B^ ) == 0 ).
//    - The parabolic time-step constraint is
//      enforced by an eos wrapper ( ni_eos_t ),
//      using the Athena++ coefficients fac_oa =
//      1 / ( 2 n_dim ) and fac_h = 1 ( multi-D ).
//
//  Everything is gated on the run-time diffusivities
//  being non-zero:  with eta_o = eta_h = eta_a =
//  hall_l = 0 no kernel is launched and the scheme
//  is bit-identical to the ideal integrator.  All
//  geometry enters through the integrator hooks (
//  edg_f, edg_d, surface, volume ), so the module
//  inherits Cartesian / cylindrical / spherical
//  geometry, thin axes, the pole and axis treatment
//  and the SMR geometry tables from the base
//  integrator it wraps.

#include "../../../../../../src/modules/dynamics/mhd/integrator/rk2.h"
#include "../../../../../../src/modules/dynamics/mhd/eos/adiabatic.h"
#include "../../../../../../src/modules/dynamics/mhd/types/block_data.h"
#include "implicit.h"
#include "eta_tensor.h"
#include "../edge_mg/edge_mg_iface.h"

namespace mhd_ee::ni
{
using type::float_t;
using type::coord_t;


////////////////////////////////////////////////////////////
//  Diffusivity providers.  A provider evaluates the three
//  diffusivities at a cell from the local state;  the
//  provider object is carried by value into the kernels,
//  so it must stay trivially copyable.  The chemistry
//  module plugs in here ( e.g. a provider evaluating the
//  conductivity tensor from the local composition ).

//  Constant diffusivities ( test / switch default ).
struct const_eta_t
{
    type::float_t eta_o, eta_h, eta_a;
    bool            on    { false };

    __host__ void read( const input & args );

    template< class bdt_T, class idx_T, class f2_T >
    __device__ __forceinline__ void eval
    ( const bdt_T & b_d, const idx_T & i,
      f2_T & eo, f2_T & eh,
      f2_T & ea )                 const
    {
        eo = eta_o;  eh = eta_h;  ea = eta_a;
        return;
    };
};

//  Hall parameterisation eta_H = hall_l | B | / rho^0.5
//  ( cell-centred ),  Ohmic / ambipolar constant.
struct hall_len_eta_t
{
    type::float_t eta_o, eta_h, eta_a, hall_l;
    bool            on    { false };

    __host__ void read( const input & args );

    template< class bdt_T, class idx_T, class f2_T >
    __device__ __forceinline__ void eval
    ( const bdt_T & b_d, const idx_T & i,
      f2_T & eo, f2_T & eh,
      f2_T & ea )                 const
    {
        using f2_t = type::float_t;
        eo = eta_o;  ea = eta_a;
        const auto * w = b_d.hyd.w.at( i );
        f2_t b2( 0 );
        for( int n = 0; n < 3; ++ n )
            b2 += f2_t( w[ idx::bcc + n ] )
                * f2_t( w[ idx::bcc + n ] );
        eh =   eta_h
             + hall_l * __sqrt1( b2 )
             / __sqrt1( __max1( f2_t( w[ idx::rho ] ),
                                f2_t( 1.e-30 ) ) );
        return;
    };
};

//  Power - law - in - density diffusivities:  eo = eta_o
//  ( rho / rho_ref )^alpha_o,  ea likewise;  eh = 0 ( the
//  Hall term stays out ).  alpha = 0 reduces to the
//  constant.   Separate prof_* pars so  ni_t 's explicit
//  sweep  ( keyed on eta_* )  stays off.
//  [profile] eta_out_kill ( bit flags:  1 = both,
//  2 = eo,  3 = ea )  zeroes the eta in the last N
//  x0 - rows.
struct prof_eta_t
{
    type::float_t eta_o, eta_a, rho_ref, alpha_o, alpha_a;
    int           out_kill;
    bool            on;

    __host__ void read( const input & args );

    template< class bdt_T, class idx_T, class f2_T >
    __device__ __forceinline__ void eval
    ( const bdt_T & b_d, const idx_T & i,
      f2_T & eo, f2_T & eh,
      f2_T & ea )                 const
    {
        using f_t = type::float_t;
        const f_t rr
        (   f_t( __max1( b_d.hyd.w.at( i )[ idx::rho ],
                         f_t( 1.e-30 ) ) ) / f_t( rho_ref ) );
        eo = f_t( eta_o );   eh = 0;   ea = f_t( eta_a );
        if( alpha_o != f_t( 0 ) )
            eo *= __pow1( rr, f_t( alpha_o ) );
        if( alpha_a != f_t( 0 ) )
            ea *= __pow1( rr, f_t( alpha_a ) );
        if( out_kill > 0
            && i[ 0 ] >= b_d.geo.n_ceff[ 0 ] - out_kill )
        {
            if( out_kill == 1 ) { eo = 0;  ea = 0;  }
            if( out_kill == 2 ) { eo = 0;             }
            if( out_kill == 3 ) {           ea = 0;   }
        }
        return;
    };
};

//  Wiring point for the chemistry module:  evaluates
//  eta from the local composition / ionisation state.
//  To be implemented together with the chemistry
//  module.
struct chem_eta_t
{
    bool on { false };

    __host__ void read( const input & args );

    template< class bdt_T, class idx_T, class f2_T >
    __device__ __forceinline__ void eval
    ( const bdt_T & b_d, const idx_T & i,
      f2_T & eo, f2_T & eh,
      f2_T & ea )                 const
    {
        eo = 0;  eh = 0;  ea = 0;
        return;
    };
};

////////////////////////////////////////////////////////////
//  Current density on the edge lattices.  J_c at corner
//  i is the circulation of B around the face dual to the
//  edge ( c, i ),  divided by the volume-centred dual
//  area:
//      J_c = [ D_ax1( B_ax2 * W_ax2 )
//            - D_ax2( B_ax1 * W_ax1 ) ] / A_dual,
//  where W is the face-centred segment length ( edg_d )
//  and A_dual the a-face area averaged between the two
//  faces bounding the dual cell.  This is the discrete
//  adjoint of the CT edge curl, so the diffusion
//  operator keeps div B = 0 to machine precision.  A
//  component c is stored for i[ c ] in [ 0, n_ceff[c] ]
//  ( the +1 row is needed by the transverse-component
//  interpolation at the right corners ).
template< class bdt_T, class itg_T >
__global__
void ni_cur_k( const bdt_T b_d, const itg_T f_itg,
               const int bc_e )
{
    using  f2_t = type::float_t;
    const auto & geo    = b_d.geo;
    const int    n_dm   = geo.n_dim;

    //  The current is evaluated on the owned corner box
    //  [ 0, n_ceff ]:  the emf kernel interpolates J
    //  one-sided at boundary-adjacent edges,  so no ghost
    //  rows are needed ( the face-field tangential ghosts
    //  are only one row deep ).  The +1 row logic matches
    //  ni_emf_k / calc_field:  the last z-block covers the
    //  corner row i[0] == n_ceff[0],  which the emf
    //  kernel's transverse interpolation reads.
    const int    n_feff = blockDim.x +
        ( blockIdx.z == gridDim.z - 1 ? 1 : 0 );
    type::idx_t i = { 0, int( blockIdx.x ),
                       int( blockIdx.y ) };
    for( int j = threadIdx.x; j < n_feff;
              j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        //  Same thin-layer skip as the emf kernel.
        bool th_skip( false );
        for( int a = n_dm; a < 3; ++ a )
            th_skip |= i[ a ] > 0;
        if( th_skip )
            continue;
        for( int c = 0; c < 3; ++ c )
        {
            if( i[ c ] > geo.n_ceff[ c ] )
                continue;
            const int ax_1 = utils::mod3( c + 1 );
            const int ax_2 = utils::mod3( c + 2 );
            const bool th_1 ( ax_1 >= n_dm );
            const bool th_2 ( ax_2 >= n_dm );
            // The edge lies in a physical - boundary plane
            // iff either in - plane corner coordinate sits
            // on it ( mirrors ct_curl_edge ).   bc_edge = 1
            // ( essential ):  boundary - plane edges carry
            // E = 0,  hence J = 0 here.
            if( bc_e == 1
                && (   ( i[ ax_1 ] == 0
                        && geo.phybnd_flag[ ax_1 ][ 0 ] )
                    || ( i[ ax_1 ] == geo.n_ceff[ ax_1 ]
                        && geo.phybnd_flag[ ax_1 ][ 1 ] )
                    || ( i[ ax_2 ] == 0
                        && geo.phybnd_flag[ ax_2 ][ 0 ] )
                    || ( i[ ax_2 ] == geo.n_ceff[ ax_2 ]
                        && geo.phybnd_flag[ ax_2 ][ 1 ] ) ) )
            {
                b_d.fld.j_edge.at( i )[ c ] = f2_t( 0 );
                continue;
            }

            f2_t j_e ( 0 );
            //  Curl by finite differences of B * ( dual
            //  edge length ).  At a PHYSICAL boundary the
            //  backward difference would read a *float_t*
            //  ghost ( the difference direction AND a
            //  tangential corner both in the ghost shell ),
            //  which the comm does not fill --  use a
            //  forward difference there ( same sign:  both
            //  are + d/ dx_ax ).   At an interface /
            //  periodic row the ghost IS comm - filled and
            //  the ghost - reading backward difference is
            //  the correct centred cross - seam form  ( the
            //  one - sided form is wrong there and corrupts
            //  the cross - interface current ).
            if( ! th_1 )
            {
                f2_t t;
                if( bc_e == 2 && i[ ax_1 ] == 0
                    && geo.phybnd_flag[ ax_1 ][ 0 ] )
                {
                    // Pseudo - vacuum ( B_t = 0 at the
                    // boundary plane ),  folded in -
                    // formula:  B_g = - B_i  =>
                    //  B_i W_i - B_g W_g = B_i ( W_i + W_g
                    //  ).
                    auto k( i );  -- k[ ax_1 ];
                    t =   b_d.fld.bf.at( i )[ ax_2 ]
                        * ( f_itg.edg_d( ax_2, i, geo )
                          + f_itg.edg_d( ax_2, k, geo ) );
                }
                else if( i[ ax_1 ] == 0
                         && geo.phybnd_flag[ ax_1 ][ 0 ] )
                {
                    auto k( i );  ++ k[ ax_1 ];
                    t  =   b_d.fld.bf.at( k )[ ax_2 ]
                         * f_itg.edg_d( ax_2, k, geo )
                         - b_d.fld.bf.at( i )[ ax_2 ]
                         * f_itg.edg_d( ax_2, i, geo );
                }
                else if( bc_e == 2
                         && i[ ax_1 ] == geo.n_ceff[ ax_1 ]
                         && geo.phybnd_flag[ ax_1 ][ 1 ] )
                {
                    // High side:  the corner sits on the
                    //  boundary plane; B( plane ) = -B( n-1
                    //  )
                    // in the backward difference.
                    auto k( i );  -- k[ ax_1 ];
                    t = - b_d.fld.bf.at( k )[ ax_2 ]
                        * ( f_itg.edg_d( ax_2, i, geo )
                          + f_itg.edg_d( ax_2, k, geo ) );
                }
                else
                {
                    auto k( i );  -- k[ ax_1 ];
                    t  =   b_d.fld.bf.at( i )[ ax_2 ]
                         * f_itg.edg_d( ax_2, i, geo )
                         - b_d.fld.bf.at( k )[ ax_2 ]
                         * f_itg.edg_d( ax_2, k, geo );
                }
                j_e += t;
            }
            if( ! th_2 )
            {
                f2_t t;
                if( bc_e == 2 && i[ ax_2 ] == 0
                    && geo.phybnd_flag[ ax_2 ][ 0 ] )
                {
                    auto k( i );  -- k[ ax_2 ];
                    t =   b_d.fld.bf.at( i )[ ax_1 ]
                        * ( f_itg.edg_d( ax_1, i, geo )
                          + f_itg.edg_d( ax_1, k, geo ) );
                }
                else if( i[ ax_2 ] == 0
                         && geo.phybnd_flag[ ax_2 ][ 0 ] )
                {
                    auto k( i );  ++ k[ ax_2 ];
                    t  =   b_d.fld.bf.at( k )[ ax_1 ]
                         * f_itg.edg_d( ax_1, k, geo )
                         - b_d.fld.bf.at( i )[ ax_1 ]
                         * f_itg.edg_d( ax_1, i, geo );
                }
                else if( bc_e == 2
                         && i[ ax_2 ] == geo.n_ceff[ ax_2 ]
                         && geo.phybnd_flag[ ax_2 ][ 1 ] )
                {
                    auto k( i );  -- k[ ax_2 ];
                    t = - b_d.fld.bf.at( k )[ ax_1 ]
                        * ( f_itg.edg_d( ax_1, i, geo )
                          + f_itg.edg_d( ax_1, k, geo ) );
                }
                else
                {
                    auto k( i );  -- k[ ax_2 ];
                    t  =   b_d.fld.bf.at( i )[ ax_1 ]
                         * f_itg.edg_d( ax_1, i, geo )
                         - b_d.fld.bf.at( k )[ ax_1 ]
                         * f_itg.edg_d( ax_1, k, geo );
                }
                j_e -= t;
            }
            //  Dual area:  the c-face area at the volume
            //  centre of the cell diagonally behind the
            //  corner.
            auto k( i );
            -- k[ ax_1 ];  -- k[ ax_2 ];
            f2_t a_d;
            if( c >= n_dm )
                a_d = f_itg.sfc_thin( c, k, geo );
            else
            {
                a_d  = geo.surface( k, c );
                ++ k[ c ];
                a_d += geo.surface( k, c );
                a_d *= 0.5;
            }
            b_d.fld.j_edge.at( i )[ c ] =
                ( a_d > 0 ? j_e / a_d : f2_t( 0 ) );
        }
    }
    return;
};

////////////////////////////////////////////////////////////
//  eta tensor at the cell centres,  for the edge-centred
//  implicit diffusion leg ( edge_mg ).
//
//      eta = eo I + ea ( I - b_hat b_hat )
//
//  i.e. Ohmic + ambipolar,  the SYMMETRIC part of the
//  diffusivity.  The Hall term is antisymmetric and cannot
//  enter a symmetric operator;  it stays on the explicit
//  emf leg ( [nonideal] ee_enable > 0 with hall != 0 is
//  refused in ni_t::init ).
//
//  The tensor is stored as DATA ( Voigt,  n_int = 6 ) so
//  that the solver's kernels never see a provider type.
//  Ambient limit:  with |B| -> 0 the anisotropic part has
//  no direction,  so eta -> eo I.
//  The per - cell tensor algebra lives in its own light
//  header ( eta_tensor.h ), so it can be reused without
//  pulling this TU's integrator in;  ni_eta_fill_k below
//  is the caller in this file.

template< class bdt_T, class eta_T >
__global__ void ni_eta_fill_k
( const bdt_T b_d, const eta_T f_eta )
{
    using  f2_t = type::float2_t;
    const auto & geo = b_d.geo;
    const type::idx_t   n  = geo.n_ceff;
    type::idx_t i( 0, int( blockIdx.x ), int( blockIdx.y ) );
    for( int j = threadIdx.x; j < n[ 0 ]; j += blockDim.x )
    {
        i[ 0 ] = j;
        type::float_t eo, eh, ea;
        f_eta.eval( b_d, i, eo, eh, ea );
        const auto * w = b_d.hyd.w.at( i );
        f2_t bh[ 3 ], et6[ 6 ];
        for( int a = 0; a < 3; ++ a )
        {
            bh[ a ] = f2_t( w[ idx::bcc + a ] );
        }
        ni_eta_tensor( f2_t( eo ), f2_t( ea ), bh, et6 );
        auto * et = b_d.fld.eta_cc.at( i );
        for( int q = 0; q < 6; ++ q )
        {
            et[ q ] = type::float_t( et6[ q ] );
        }
    }
    return;
};

////////////////////////////////////////////////////////////
//  Diffusion emf at the owned edges:  J_|| taken from
//  j_edge natively, the two transverse components
//  interpolated from the four adjacent edge lattices
//  ( clamped to the nearest stored row at the left
//  boundary corners ),  B and eta averaged from the
//  surrounding cells.  e_dif is stored ( for the energy
//  kernel ) and,  for the coupled scheme ( ACC = true ),
//  added into the total emf ee *before* the edge
//  exchange.  For the Godunov-split sweep ( ACC = false )
//  the owned rows of ee are WRITTEN directly to the emf
//  ( non-owned slots zeroed ) --  there is no separate
//  copy kernel --  and the edge exchange then fills the
//  +- boundary rows.
template< class bdt_T, class itg_T, class eta_T,
          bool ACC = true >
__global__
void ni_emf_k( const bdt_T b_d, const itg_T f_itg,
               const eta_T f_eta, const int bc_e,
               const type::float_t chyp
                   = type::float_t( 0 ),
               const type::float_t upw
                   = type::float_t( 0 ) )
{
    using  f2_t = type::float_t;
    const auto & geo    = b_d.geo;
    const int    n_dm   = geo.n_dim;
    const int    n_feff = blockDim.x +
        ( blockIdx.z == gridDim.z - 1 ? 1 : 0 );

    type::idx_t i = { 0, int( blockIdx.x ),
                       int( blockIdx.y ) };
    for( int j = threadIdx.x; j < n_feff;
             j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        bool th_skip( false );
        for( int a = n_dm; a < 3; ++ a )
            th_skip |= i[ a ] > 0;
        if( th_skip )
            continue;
        for( int a = 0; a < 3; ++ a )
        {
            if( i[ a ] > geo.n_ceff[ a ] )
                continue;
            const int ax_1 = utils::mod3( a + 1 );
            const int ax_2 = utils::mod3( a + 2 );
            const bool th_1 ( ax_1 >= n_dm );
            const bool th_2 ( ax_2 >= n_dm );

            //  B and eta:  average of the surrounding
            //  cells.   ( Before the current average:  the
            //  H3 upwind bias below reads b - hat here. )
            f2_t b_e[ 3 ] = { 0, 0, 0 };
            f2_t eo ( 0 ), eh ( 0 ), ea ( 0 );
            int  n_c ( 0 );
            for( int t1 = th_1 ? 0 : -1; t1 <= 0; ++ t1 )
            for( int t2 = th_2 ? 0 : -1; t2 <= 0; ++ t2 )
            {
                auto k( i );
                k[ ax_1 ] += t1;
                k[ ax_2 ] += t2;
                const auto * w = b_d.hyd.w.at( k );
                for( int n = 0; n < 3; ++ n )
                    b_e[ n ] += f2_t( w[ idx::bcc + n ] );
                f2_t to, th, ta;
                f_eta.eval( b_d, k, to, th, ta );
                eo += to;  eh += th;  ea += ta;
                ++ n_c;
            }
            for( int n = 0; n < 3; ++ n )
                b_e[ n ] /= n_c;
            eo /= n_c;  eh /= n_c;  ea /= n_c;

            //  The upwind Hall bias ( knob
            //  hall_upw,  off by default ).
            const f2_t eh_a ( __abs1( eh ) );
            auto fu_ax = [ & ] ( const int & ax )
            {
                if( upw == f2_t( 0 ) || eh_a == f2_t( 0 )
                    || ax >= n_dm )
                    return f2_t( 0 );
                const f2_t dx ( f2_t( geo.dx_f( ax, i ) ) );
                const f2_t fu ( upw * eh_a
                    * f2_t( b_d.dt ) / ( dx * dx ) );
                return f2_t( __min1( fu, f2_t( 0.45 ) ) );
            };

            //  Current components at the corner.
            f2_t j_c[ 3 ];
            j_c[ a ] = b_d.fld.j_edge.at( i )[ a ];
            for( int m = 1; m <= 2; ++ m )
            {
                const int ax = m == 1 ? ax_1 : ax_2;
                j_c[ ax ] = 0;
                int  n_s  = 0;
                //  ax-edges at a-faces { i[a], i[a]+1 }
                //  and ax-cells { i[ax]-1, i[ax] };
                //  thin directions collapse.  j_edge has
                //  no exchanged ghost rows ( the face field
                //  ghosts are only one row deep ),
                //  so the stencil falls back to the
                //  available one-sided rows at the edges
                //  of the owned box;  at a periodic seam
                //  the + boundary edge is owned by the
                //  neighbour and arrives via the emf
                //  exchange,  keeping the two seam sides
                //  consistent.
                const int s_hi = ( a  >= n_dm ||
                                   i[ a ]
                                   == geo.n_ceff[ a ] )
                               ? 0 : 1;
                const int t_lo = ( ax >= n_dm ||
                                   i[ ax ] == 0 )
                               ? 0 : -1;
                //  The upwind bias weights on the
                //  a / ax float_t loop ( s = 1 moves +e_a
                // --  the right edge of the a - face pair;
                //  t = 0 the +e_ax side ( k = i itself
                // )  --  each axis'  pair biased toward the
                //  b - hat side ).
                const f2_t fu_s ( fu_ax( a ) );
                const f2_t fu_t ( fu_ax( ax ) );
                const f2_t w_s_hi ( f2_t( 0.5 )
                    + ( b_e[ a ] > f2_t( 0 )
                          ?   fu_s : - fu_s ) );
                const f2_t w_t_hi ( f2_t( 0.5 )
                    + ( b_e[ ax ] > f2_t( 0 )
                          ?   fu_t : - fu_t ) );
                for( int s = 0; s <= s_hi; ++ s )
                for( int t = t_lo; t <= 0; ++ t )
                {
                    auto k( i );
                    k[ a  ] += s;
                    k[ ax ] += t;
                    f2_t wgt ( 1 );
                    if( s_hi > 0 )
                        wgt *= ( s == 1 ? w_s_hi
                                        : f2_t( 1 )
                                          - w_s_hi );
                    if( t_lo < 0 )
                        wgt *= ( t == 0 ? w_t_hi
                                        : f2_t( 1 )
                                          - w_t_hi );
                    j_c[ ax ] += wgt
                        * f2_t( b_d.fld.j_edge
                                       .at( k )[ ax ] );
                    ++ n_s;
                }
                j_c[ ax ] /= n_s;
            }


            //  Diffusion emf.
            f2_t b2 ( 0 );
            for( int n = 0; n < 3; ++ n )
                b2 += b_e[ n ] * b_e[ n ];
            //  Diffusion emf.  Ohmic part is always
            //  defined;  the Hall / ambipolar terms need
            //  the field direction,  undefined when
            //  |B| vanishes --  guard with a threshold
            //  ( also shields the fp traps from denormal
            //  roundoff fields ).
            const f2_t bn = __sqrt1( b2 );
            f2_t e_d = eo * j_c[ a ];
            if( bn > 1.e-15 )
            {
                f2_t jb ( 0 );
                for( int n = 0; n < 3; ++ n )
                    jb += j_c[ n ] * b_e[ n ];
                e_d +=   eh * (   j_c[ ax_1 ] * b_e[ ax_2 ]
                                - j_c[ ax_2 ]
                                * b_e[ ax_1 ] )
                              / bn
                       + ea * ( j_c[ a ]
                                - jb / b2 * b_e[ a ] );
                //  The fourth - order whistler
                //  hyper - resistivity ( knob hall_chyp ).
                if( chyp > f2_t( 0 ) )
                {
                    f2_t lap ( 0 );
                    for( int m = 0; m < 3; ++ m )
                    {
                        const int ax( m == 0 ? a
                                    : m == 1 ? ax_1
                                             : ax_2 );
                        if( ax >= n_dm )
                            continue;
                        const f2_t dx
                            ( f2_t( geo.dx_f( ax, i ) ) );
                        type::idx_t km( i ), kp( i );
                        const int na( geo.n_ceff[ ax ] );
                        if( i[ ax ] <= 0 )
                        {   ++ kp[ ax ];
                            ++ km[ ax ];   km[ ax ] -= 2; }
                        else if( i[ ax ] >= na )
                        {   -- km[ ax ];
                            -- kp[ ax ];   kp[ ax ] += 2; }
                        else
                        {   -- km[ ax ];   ++ kp[ ax ]; }
                        lap += (   b_d.fld.j_edge
                                       .at( km )[ a ]
                                 - f2_t( 2 )
                                 * b_d.fld.j_edge
                                       .at( i )[ a ]
                                 + b_d.fld.j_edge
                                       .at( kp )[ a ] )
                               / ( dx * dx );
                    }
                    f2_t dx2 ( f2_t( geo.dx_f( 0, i ) )
                             * geo.dx_f( 0, i ) );
                    for( int m = 1; m < n_dm; ++ m )
                    {
                        const f2_t d
                            ( f2_t( geo.dx_f( m, i ) ) );
                        dx2 = fmin( dx2, d * d );
                    }
                    e_d -= chyp * eh_a * dx2 * lap;
                }
            }
            // Essential BC ( bc_edge = 1 ):  n x E = 0 on
            // the boundary plane --  the in - plane edge
            // carries zero total emf there ( J was already
            // zeroed in ni_cur_k;  this covers the Hall /
            // ambipolar transverse terms ).
            if( bc_e == 1
                && (   ( i[ ax_1 ] == 0
                        && geo.phybnd_flag[ ax_1 ][ 0 ] )
                    || ( i[ ax_1 ] == geo.n_ceff[ ax_1 ]
                        && geo.phybnd_flag[ ax_1 ][ 1 ] )
                    || ( i[ ax_2 ] == 0
                        && geo.phybnd_flag[ ax_2 ][ 0 ] )
                    || ( i[ ax_2 ] == geo.n_ceff[ ax_2 ]
                        && geo.phybnd_flag[ ax_2 ][ 1 ] ) ) )
                e_d = f2_t( 0 );
            b_d.fld.e_dif.at( i )[ a ] = e_d;
            //  Coupled ( ACC ):  augment the ideal emf in
            //  place.   Split ( ! ACC ):  ee is the
            //  diffusion-emf working buffer.  BOTH store only
            //  the OWNED edges ( i[ a ] < n_ceff[ a ],
            //  matching calc_field ):  the +-boundary rows are
            //  delivered by the edge exchange ( act_flx ) and
            //  writing them here would leave un-reconciled
            //  local values that mix with the exchange.  A
            //  thin - axis COMPONENT ( a >= n_dm ) is NOT an
            //  unowned slot:  the poloidal faces' curl
            //  differences run along the NON - thin axes,  so
            //  they survive intg_field and read ee [ .. ][ a ]
            //  ( it drops only the terms whose DIFFERENCE
            //  direction is thin ). In 2.5D ee [ .. ][ 2 ] is
            //  therefore the whole poloidal evolution and must
            //  not be zeroed.
            if( ACC )
            {
                if( i[ a ] < geo.n_ceff[ a ] )
                    b_d.fld.ee.at( i )[ a ] += e_d;
            }
            else
                b_d.fld.ee.at( i )[ a ] =
                    ( i[ a ] < geo.n_ceff[ a ] )
                    ? e_d : f2_t( 0 );
        }
    }
    return;
};

////////////////////////////////////////////////////////////
//  Godunov-split sweep helpers.  The sweep runs once per
//  cycle,  after the ideal step,  and uses the slice pair
//  ( 0, 1 ):  0 holds the post-ideal state.  See ni_t.

//  Conserved -> primitive refresh for the sweep ( the
//  eos cons2prim has more dependencies;  the sweep only
//  needs w[ 0 ] for B and eta ).  The post-ideal state
//  is slice 0.  The cell-centred B is derived from the
//  CT faces ( as in eos cons2prim )  --  the conserved
//  u[ bcc ] is stale by the last field update.
template< class bdt_T >
__global__ void ni_hydro_k( const bdt_T b_d )
{
    using     f_t = typename decltype( b_d.hyd )::f_t;
    const auto & geo = b_d.geo;
    type::idx_t i = { 0, int( blockIdx.x ),
                       int( blockIdx.y ) };
    for( int j = threadIdx.x; j < blockDim.x;
             j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        if( i[ 0 ] >= geo.n_ceff[ 0 ] )
            continue;
        auto       w ( b_d.hyd.w.at( i, 0 ) );
        //  Only rho and the cell-centred B are read
        //  downstream ( ni_emf_k / ni_ene_k / hall-length
        //  eta ),  so only those are refreshed --  no eos
        //  gamma needed.  Matches the eos cons2prim cell
        //  B ( face average;  background b_l on thin
        //  axes ).
        w[ idx::rho ] = b_d.hyd.u.at( i, 0 )[ idx::rho ];
        for( int a = 0; a < 3; ++ a )
        {
            const f_t b_l ( b_d.fld.bf.at( i, 0 )[ a ] );
            if( geo.n_ceff[ a ] < 2 )
            {
                w[ idx::bcc + a ] = b_l;
                continue;
            }
            auto k ( i );
            ++ k[ a ];
            w[ idx::bcc + a ] = ( b_l
                + b_d.fld.bf.at( k, 0 )[ a ] ) * f_t( 0.5 );
        }
    }
    return;
};

////////////////////////////////////////////////////////////
//  The RKC1 ( damped - Chebyshev recursion )  sweep stages.
//  ni_rkc_comb_k:  slice 1 holds the Euler part
//  B_{j-1} + chi T( B_{j-1} )  from intg_field_only;
//  combine into  B_j = slice1 + ( mu - 1 ) B_{j-1}
//  + nu B_{j-2}   ( mu + nu == 1 ).   ni_rkc_roll_k:
//  bf_prev <- slice 0,  slice 0 <- slice 1.   Coverage
//  matches the CT update's rows ( field_intg 's face set ).
template< class bdt_T >
__global__ void ni_rkc_comb_k
( const bdt_T b_d, const type::float_t mu1,
  const type::float_t nu )
{
    const auto & geo ( b_d.geo );
    const int    n_dm ( geo.n_dim );
    const int    n_feff ( blockDim.x +
        ( blockIdx.z == gridDim.z - 1 ? 1 : 0 ) );
    type::idx_t i = { 0, int( blockIdx.x ),
                       int( blockIdx.y ) };
    for( int j = threadIdx.x; j < n_feff; j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        for( int a = 0; a < 3; ++ a )
        {
            if( a >= n_dm )
                continue;
            const int ax_1 ( utils::mod3( a + 1 ) );
            const int ax_2 ( utils::mod3( a + 2 ) );
            if( i[ a    ] >  geo.n_ceff[ a    ] ||
                i[ ax_1 ] >= geo.n_ceff[ ax_1 ] ||
                i[ ax_2 ] >= geo.n_ceff[ ax_2 ]  )
                continue;
            b_d.fld.bf.at( i, 1 )[ a ] +=
                  mu1 * b_d.fld.bf     .at( i, 0 )[ a ]
                + nu  * b_d.fld.bf_prev.at( i )[ a ];
        }
    }
    return;
};

template< class bdt_T >
__global__ void ni_rkc_roll_k( const bdt_T b_d )
{
    const auto & geo ( b_d.geo );
    const int    n_dm ( geo.n_dim );
    const int    n_feff ( blockDim.x +
        ( blockIdx.z == gridDim.z - 1 ? 1 : 0 ) );
    type::idx_t i = { 0, int( blockIdx.x ),
                       int( blockIdx.y ) };
    for( int j = threadIdx.x; j < n_feff; j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        for( int a = 0; a < 3; ++ a )
        {
            if( a >= n_dm )
                continue;
            const int ax_1 ( utils::mod3( a + 1 ) );
            const int ax_2 ( utils::mod3( a + 2 ) );
            if( i[ a    ] >  geo.n_ceff[ a    ] ||
                i[ ax_1 ] >= geo.n_ceff[ ax_1 ] ||
                i[ ax_2 ] >= geo.n_ceff[ ax_2 ]  )
                continue;
            b_d.fld.bf_prev.at( i )[ a ] =
                b_d.fld.bf     .at( i, 0 )[ a ];
            b_d.fld.bf     .at( i, 0 )[ a ] =
                b_d.fld.bf     .at( i, 1 )[ a ];
        }
    }
    return;
};

////////////////////////////////////////////////////////////
//  Energy update and heating diagnostics.  The diffusion
//  energy flux through the a-faces is the volume-centred
//  Poynting flux
//      S_a = E_ax1 B_ax2 - E_ax2 B_ax1,
//  with the edge emfs averaged over the two face edges
//  and B averaged over the two cells;  the conserved
//  energy receives its flux divergence.  The heating
//  powers are evaluated from the cell-centred primal
//  curl ( the same metric curl as the edge kernel,
//  centred on the cell ) and accumulated over each
//  cycle ( reset at the first substep ),  separately
//  for the Ohmic ( heat[ 0 ] )  and  ambipolar
//  ( heat[ 1 ] )  parts;  the Hall term does
//  not heat.
template< class bdt_T, class itg_T, class eta_T >
__global__
void ni_ene_k( const bdt_T b_d, const itg_T f_itg,
               const eta_T f_eta, const int step,
               const bool rst_heat )
{
    using  f2_t = type::float_t;
    const auto & geo  = b_d.geo;
    const int    n_dm = geo.n_dim;

    type::idx_t i = { 0, int( blockIdx.x ),
                       int( blockIdx.y ) };
    for( int j = threadIdx.x; j < blockDim.x;
             j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        bool th_skip( false );
        for( int a = n_dm; a < 3; ++ a )
            th_skip |= i[ a ] > 0;
        if( th_skip )
            continue;

        auto * u = b_d.hyd.u.at( i, step );
        //  The deposit itself is applied at the END, once
        //  the
        //  heating ledger below has the cell-centred J /
        //  eta
        //  ( form 2 needs them ).  See the branch there.

        //  Heating diagnostics:  the cell-centred
        //  current is the average of the cell's edge
        //  currents ( the dual curl of the operator
        //  itself ),  so the diagnostics are exactly
        //  consistent with the diffusion applied.
        f2_t j_cc[ 3 ];
        for( int c = 0; c < 3; ++ c )
        {
            const int ax_1 = utils::mod3( c + 1 );
            const int ax_2 = utils::mod3( c + 2 );
            j_cc[ c ] = 0;
            int  n_s  = 0;
            for( int t1 = 0; t1 <= ( ax_1 >= n_dm ? 0 : 1 );
                     ++ t1 )
            for( int t2 = 0; t2 <= ( ax_2 >= n_dm ? 0 : 1 );
                     ++ t2 )
            {
                auto k( i );
                k[ ax_1 ] += t1;
                k[ ax_2 ] += t2;
                j_cc[ c ] += f2_t( b_d.fld.j_edge
                                   .at( k )[ c ] );
                ++ n_s;
            }
            j_cc[ c ] /= n_s;
        }
        f2_t eo, eh, ea;
        f_eta.eval( b_d, i, eo, eh, ea );
        f2_t b2 ( 0 ), j2 ( 0 ), jp2 ( 0 );
        f2_t b_cc[ 3 ];
        const auto * w = b_d.hyd.w.at( i );
        for( int n = 0; n < 3; ++ n )
        {
            b_cc[ n ] = f2_t( w[ idx::bcc + n ] );
            b2 += b_cc[ n ] * b_cc[ n ];
            j2 += j_cc[ n ] * j_cc[ n ];
        }
        if( b2 > 1.e-30 )
        {
            f2_t jb ( 0 );
            for( int n = 0; n < 3; ++ n )
                jb += j_cc[ n ] * b_cc[ n ];
            jp2 = j2 - jb * jb / b2;
        }
        auto * ht = b_d.fld.heat.at( i );
        //  Heating accumulated over the current cycle:
        //  reset on the first ( sub- )step so the array
        //  needs no zero initialisation.   rst_heat is
        //  decoupled from step so the STS sweep can keep
        //  u on slice 0 while resetting only on its first
        //  Chebyshev sub-step.
        if( rst_heat )
        {
            ht[ 0 ] = b_d.dt * eo * j2;
            ht[ 1 ] = b_d.dt * ea * jp2;
        }
        else
        {
            ht[ 0 ] += b_d.dt * eo * j2;
            ht[ 1 ] += b_d.dt * ea * jp2;
        }

        //  THE DEPOSIT  ( [nonideal] mg_cc_ene ).
        //  2 ( default ): the METRIC-FREE measured form (
        //  see
        //      mg_ni_ene_mf_k ) --
        //  the field's own energy change, read from the
        //  updated
        //  faces with the eos's own face average, plus the
        //  same
        //      heating ledger just accumulated:
        //  u[ ene ] += ( |B*_cc|^2 - |B^n_cc|^2 ) / 2
        //  + dt ( eo |J|^2 + ea |J_perp|^2 ).
        //  B^n_cc is w[ bcc ] ( the cell-centred field the
        //  eos
        //  wrote at the top of the step ), B*_cc the face
        //  average
        //  of the updated bf -- exactly the two the eos
        //  will
        //      difference on the next cycle.
        if( f_itg.ene_dep == 2 )
        {
            f2_t b_s[ 3 ];
            for( int n = 0; n < 3; ++ n )
            {
                const bool thin( geo.n_ceff[ n ] < 2 );
                auto r( i );
                if( ! thin )
                    ++ r[ n ];
                const f2_t bl( b_d.fld.bf.at( i, step )[ n ] );
                const f2_t br( thin ? bl
                              : f2_t( b_d.fld.bf
                                      .at( r, step )[ n ] ) );
                b_s[ n ] = ( bl + br ) * f2_t( 0.5 );
            }
            f2_t dm( 0 );
            for( int n = 0; n < 3; ++ n )
                dm +=   b_s [ n ] * b_s [ n ]
                      - b_cc[ n ] * b_cc[ n ];
            u[ idx::ene ] += f2_t( 0.5 ) * dm
                + ( f_itg.ene_heat
                    ? b_d.dt * ( eo * j2 + ea * jp2 )
                    : f2_t( 0 ) );
        }
    }
    return;
};

//  Parabolic dt limit for the STS sizing,  with the Hall
//  stiffness included  ( parabolic form:  lam_max ~
//  ed k_max^2 ).
template< class bdt_T, class eta_T >
__global__
void ni_dtlim_k( const bdt_T b_d, const eta_T f_eta,
                 type::float_t * p_dt_dif )
{
    using f2_t = type::float_t;
    const auto & geo = b_d.geo;
    const int    n_dm = geo.n_dim;
    __dyn_shared__( char, var_sh );
    auto * dt = ( type::float_t * ) var_sh;
    const int  i_th = threadIdx.x;
    dt[ i_th ] = type::float_t( 1.e30 );
    type::idx_t i = { 0, int( blockIdx.x ),
                       int( blockIdx.y ) };
    for( int j = i_th; j < blockDim.x; j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        if( i[ 0 ] >= geo.n_ceff[ 0 ] )
            continue;
        f2_t eo, eh, ea;
        f_eta.eval( b_d, i, eo, eh, ea );
        //  Hall  ( whistler )  stiffness joins the sizing
        //  ( parabolic form:  lam_max ~ ed k_max^2 ).
        const f2_t ed = eo + ea + __abs1( eh );
        if( ed <= 0 )
            continue;
        f2_t d = geo.dx_cfl( 0, i );
        for( int a = 1; a < n_dm; ++ a )
            d = __min1( d, geo.dx_cfl( a, i ) );
        const f2_t fac = f2_t( 0.5 ) / n_dm;
        dt[ i_th ] = __min1( dt[ i_th ],
                    ( type::float_t )( fac * d * d / ed ) );
    }
    __syncthreads(  );
    utils::block_reduce< type::float_t, __min1 >
                       ( dt, blockDim.x );
    if( i_th == 0 )
        utils::atomic_min( p_dt_dif, dt[ 0 ] );
    return;
};

////////////////////////////////////////////////////////////
//  Integrator wrapper.  Enroll as
//      mod.enroll< dynamics::mhd::proxy_t,
//                  mhd_ee::ni::ni_t
//                      < dynamics::mhd::integrator::sph_t<>,
//                        mhd_ee::ni::const_eta_t > >();
//  with any geometry integrator as base_T.
template< class base_T = dynamics::mhd::integrator::rk2_t<>,
          class eta_T  = const_eta_t >
struct ni_t : base_T
{
    using  this_t = ni_t;
    using super_t = base_T;

    eta_T eta_p;
    bool    on  { false };
    //  Time-integration mode:  0 = coupled ( the diffusion
    //  emf rides the ideal ee through the RK2 step );
    //  1 = Godunov split ( a separate diffusion sweep after
    //  the ideal step ).   Read from [nonideal] mode.
    int     mode { 0 };
    //  Super-time-stepping ( split mode only ):  relax the
    //  parabolic dt cap and cover the diffusion with s
    //  Chebyshev-damped sub-steps ( classic STS,  as in
    //  Athena++ operator_split ).   sts_nu is the damping.
    bool                sts    { false };
    type::float_t      sts_nu { 0.01 };
    //  [nonideal] mg_cc_ene:  deposit the diffusion energy
    //  into u[ ene ]. This is REQUIRED: the field's energy
    //  change contains a transport part ( div S ) that must
    //  NOT be handed to the gas as heat, and the deposit
    //  removes it, leaving the gas with exactly dt eta J^2.
    //  0 = off,  2 = the METRIC-FREE measured form ( default ):
    //      u[ ene ] += ( |B*_cc|^2 - |B^n_cc|^2 ) / 2
    //  + dt ( eta_o |J|^2 + eta_a |J_perp|^2 ),
    //  cell-local, so the polar surface / volume metric is
    //  never divided.
    int                 ene_dep{ 1 };
    //  [nonideal] mg_cc_ene_heat ( DEFAULT 1 ): the form-2
    //  heating term dt ( eta_o |J|^2 + eta_a |J_perp|^2 ).
    //  0 = the measured field - energy change alone.
    int                 ene_heat{ 1 };
    type::float_t  *    p_dt_dif { nullptr };
    //  The explicit / split chain's boundary - edge rule:
    //  0 = free  ( one - sided ),  1 = essential  ( E_t = 0
    //  on boundary - plane edges ),  2 = pseudo - vacuum
    //  ( the in - formula antisymmetric closure ).
    int                 bc_edge { 0 };
    //  Whistler-cap coefficients:  hall_chyp weights the
    //  fourth - order hyper - resistivity,  hall_upw the
    //  upwind Hall bias ( 0 = off ).
    type::float_t       hall_chyp { 0.05 };
    type::float_t       hall_upw  { 0 };
    //  The face-centered Laplacian implicit solver (
    // mode=2; implicit.h --  the operations live in
    // implicit_t ).
    implicit_t<>        imp;
    //  The edge-centred implicit tensor-diffusion leg:  its
    //  device code lives in edge_mg/edge_mg.cpp,  so this
    //  member costs the pgen TU nothing but a few bytes.
    edge_mg::solver_t   ee_s;

    //  The wrapped geometry hooks are inherited;  the
    //  kernels above only use edg_f / edg_d / surface /
    //  volume through the wrapped integrator.

    __host__ virtual void init
    ( const input & args, const::dynamics::base_t & mod )
    {
        super_t::init( args, mod );
        eta_p.read( args );
        on = eta_p.on;
        //  The implicit curl - curl leg is not instantiated.
        //  Both keys below can now only make the surviving
        //  explicit leg do the WRONG thing silently:
        //  mg_cc > 0 asks for a module that does not exist,
        //  and mg_cc_eta_var was how that module took
        //  ownership of eta  ( switching the explicit sweep
        //  off,  so nobody supplies eta ).
        //  So: refuse,  and say what to do instead.
        if( args.get< int >( "nonideal", "mg_cc", 0 ) > 0 )
            throw std::runtime_error
            ( "mhd_ee::ni:  [nonideal] mg_cc > 0 --  this build has no"
              " implicit curl - curl MG leg,  so the key would be"
              " silently ignored."
              "  Set mg_cc = 0 and use the explicit coupled leg"
              " ( mode = 0 ),  the edge-centred implicit step"
              " ( ee_enable = 1 ),  or turn non-ideal off." );
        if( args.get< int >( "nonideal", "mg_cc_eta_var", 0 ) )
            throw std::runtime_error
            ( "mhd_ee::ni:  [nonideal] mg_cc_eta_var = 1 was how the"
              " implicit leg took ownership of eta;  that leg is not"
              " instantiated,  so nobody would supply eta --  set"
              " mg_cc_eta_var = 0 and put the diffusivity back into"
              " eta_o / eta_a ( or the prof_* pars )." );
        mode = args.get< int >( "nonideal", "mode", 0 );
        //  Edge-centred implicit tensor diffusion:  eta is the
        //  SYMMETRIC diffusivity,  so the Hall term has no home
        //  here and must not be silently dropped -- refuse
        //  instead.  The built-in providers read eta_h / hall_l
        //  from the par;  disk_eta sets eh = 0 unconditionally,
        //  which this check cannot see but which is by
        //  construction Hall-free.
        ee_s.read( args );
        if( ee_s.par.enable )
        {
            if( args.get< type::float_t >
                ( "nonideal", "eta_h", 0 ) != 0
             || args.get< type::float_t >
                ( "nonideal", "hall_l", 0 ) != 0 )
            {
                throw std::runtime_error
                ( "mhd_ee::ni:  [nonideal] ee_enable > 0 together"
                  " with eta_h / hall_l != 0.  The edge-centred"
                  " implicit leg solves with the SYMMETRIC"
                  " eta = eta_o I + eta_a ( I - b b ),  so the"
                  " Hall term cannot enter it,  and running the"
                  " explicit Hall sweep as well would"
                  " double count the Ohmic /"
                  " ambipolar part."
                  "  Set eta_h = hall_l = 0,  or use mode = 0." );
            }
            if( mode == 2 )
            {
                throw std::runtime_error
                ( "mhd_ee::ni:  [nonideal] ee_enable > 0 and"
                  " mode = 2 are two different implicit legs;"
                  "  use one." );
            }
            dynamics::mhd::fld_t::need_ni_ee = true;
        }
        sts = args.get< int >( "nonideal", "sts", 0 ) != 0;
        sts_nu = args.get< type::float_t >
                          ( "nonideal", "sts_nu", 0.01 );
        if( sts && ee_s.par.enable )
        {
            throw std::runtime_error
            ( "mhd_ee::ni:  [nonideal] sts = 1 and"
              " ee_enable = 1 are two different diffusion"
              " legs;  use one.  The STS leg relaxes the"
              " parabolic cap,  so running both would let"
              " the implicit solve take a step it was not"
              " configured for." );
        }
        ene_dep = args.get< int >
            ( "nonideal", "mg_cc_ene", 2 );
        if( ene_dep == 1 )
            throw std::runtime_error
            ( "mhd_ee::ni:  mg_cc_ene = 1 ( the Poynting - flux"
              " deposit ) is not available in this build:  the"
              " deposit is the metric - free measured form"
              " ( mg_cc_ene = 2,  the default )."
              "  Set mg_cc_ene = 2,  or 0 to turn the deposit off." );
        if( ene_dep < 0 || ene_dep > 3 )
            throw std::runtime_error
            ( "mhd_ee::ni:  mg_cc_ene must be 0 ( off ), 2"
              " ( the metric - free measured form,  default ) or 3"
              " ( the SBP face - flux form,  implicit leg only )." );
        //  [nonideal] mg_cc_ene = 3 ( the SBP face - flux
        //  form ) is implemented on the implicit curl - curl
        //  leg only,  which is not instantiated here,  so there
        //  is no owner to take the deposit -- refuse rather than
        //  silently falling back to form 2.
        if( ene_dep == 3 )
            throw std::runtime_error
            ( "mhd_ee::ni:  mg_cc_ene = 3 ( the SBP face - flux form )"
              " is implemented only on the implicit curl - curl leg,"
              " which this build does not instantiate."
              "  Use mg_cc_ene = 2 ( the measured form,  default )." );
        ene_heat = args.get< int >
            ( "nonideal", "mg_cc_ene_heat", 1 );
        bc_edge = args.get< int >
                          ( "nonideal", "bc_edge", 0 );
        hall_chyp = args.get< type::float_t >
            ( "nonideal", "hall_chyp", 0.05 );
        hall_upw  = args.get< type::float_t >
            ( "nonideal", "hall_upw", 0 );
        if( bc_edge < 0 || bc_edge > 2 )
            throw std::runtime_error
            ( "mhd_ee::ni:  bc_edge must be  0  ( free"
              " one - sided ),  1  ( essential )  or  2  ( pseudo-"
              "vacuum )." );
        if( sts )
        {
            //  STS needs the split sweep and a device
            // scalar for the global parabolic-dt reduction.
            mode = 1;
            p_dt_dif = mod.p_dev->malloc< type::float_t >
                ( 1, false );
        }
        if( on )
            dynamics::mhd::fld_t::need_ni = true;
        if( mode == 2 )
        {
            //  Implicit solve:  red-black Gauss-Seidel on the
            //  face-centered Laplacian ( implicit.h::solve ),
            //  which names this scalar  p_res --  a residual
            //  reduction buffer, NOT a CG dot-product one.
            //  The unknown is  cg_x;  the RHS is bf slice 0.
            dynamics::mhd::fld_t::need_ni_imp = true;
            if( ! p_dt_dif )
                p_dt_dif
                    = mod.p_dev->malloc< type::float_t >
                    ( 1, false );
        }
        return;
    };

    using dual_t = mesh::block::dual_t;

    //  calc_flx:  after the ideal emf is in ee, evaluate
    //  J on the edges and add the diffusion emf ( the
    //  subsequent edge exchange then acts on the total
    //  emf,  and the pole / axis averaging kernels see
    //  the total as well ).
    template< class bdt_T, class eos_T,
              class rec_T, class rie_T >
    __host__ void calc_flx
    ( const bdt_T & b_d, const dual_t & d,
      const eos_T & f_eos, const rec_T & f_rec,
      const rie_T & f_rie, const int & step,
      const device::base_t & dev )       const
    {
        super_t::calc_flx( b_d, d, f_eos, f_rec, f_rie,
                           step, dev );
        //  Coupled mode only;  split mode uses diffuse().
        if( ! on || mode != 0 )
            return;
        //  ni_cur_k reads the bf ghost shell,  so the edge
        //  exchange must have completed:  sync all streams
        //  before launching it.
        const_cast< device::base_t & >( dev )
            .sync_all_streams(  );
        const auto res = this->field_res( b_d );
        dev.launch( ni_cur_k< bdt_T, ni_t >,
                    res, d.stream, b_d, * this, bc_edge );
        return dev.launch
            ( ni_emf_k< bdt_T, ni_t, eta_T >,
              res, d.stream, b_d, * this, eta_p, bc_edge,
              type::float_t( hall_chyp ),
              type::float_t( hall_upw ) );
    };

    //  intg_flx:  after the hydro / field update, apply
    //  the diffusion energy flux and accumulate the
    //  heating diagnostics.
    template< class bdt_T, class eos_T >
    __host__ void intg_flx
    ( const bdt_T & b_d, const dual_t & d,
      const eos_T & f_eos, const int & step,
      device::base_t & dev )             const
    {
        super_t::intg_flx( b_d, d, f_eos, step, dev );
        //  Coupled mode only;  split mode uses diffuse().
        if( ! on || mode != 0 )
            return;
        const auto res = this->cons_res( b_d );
        return dev.launch
            ( ni_ene_k< bdt_T, ni_t, eta_T >,
              res, d.stream, b_d, * this, eta_p, step,
              step == 0 );
    };

    ////////// Godunov-split diffusion sweep //////////
    //  Run once per cycle,  after the ideal RK2 step,  and
    //  invoked from the pgen's module step --  the core
    //  integrator carries no hook.  It uses the ee buffer as
    //  the standalone diffusion-emf working array and reuses
    //  the act_flx edge exchange,  so dynamics.h is untouched.
    //  The ideal RK2 leaves the post-step state in slice 0;
    //  the sweep updates that slice.   No per-block sync is
    //  needed:  this is a separate phase,  fully ordered
    //  against the ideal step by the act_bnd / act_flx event
    //  machinery it reuses.
    __host__ void diffuse
    ( mesh::mesh_t & mesh, ::dynamics::base_t & mod )
    {
        if( ! on )
            return;
        //  One dt for the whole sweep.  ee_dt > 0 is the
        //  Forced-dt override:  it lets the implicit edge leg run
        //  at the same large dt as STS,  instead of the
        //  parabolic cap that ni_eos_t applies.
        const type::float_t dt_use
            ( ( ee_s.par.enable || sts )
              && ee_s.par.ee_dt > 0
              ? ee_s.par.ee_dt : mesh.p_cyc->dt(  ) );
        if( ee_s.par.enable )
        {
            //  The edge-centred leg.  It owns the whole
            //  diffusive update: the explicit / split sweep
            //  below must not also run ( it would add the
            //  same Ohmic + ambipolar emf a second time ).
            //  S0':  pack eta,  launch the divergence
            //  diagnostic,  leave the state untouched --
            //  the bit-identity check of foundations 9.
            auto & dev ( * mod.p_dev );
            dev.sync_all_streams(  );
            mod.p_bdk->act_bnd( mesh, 0 );
            dev.sync_all_streams(  );
            if( ee_s.par.solve )
            {
                int n_local = 0;
                for( auto & d : mod )
                {
                    ( void ) d;
                    ++ n_local;
                }
                if( n_local > 1 )
                {
                    throw std::runtime_error
                    ( "edge_mg:  implicit solve is"
                      " single-block only" );
                }
            }
            for( auto & d : mod )
            {
                auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                b_d.dt   = dt_use;
                dev.launch
                ( ni_eta_fill_k
                  < decltype( b_d ), eta_T >,
                  this->cons_res( b_d ), d.stream,
                  b_d, eta_p );
                auto v = edge_mg::make_view( b_d );
                ee_s.step( dev, d.stream, v );
                if( ee_s.last_it >= ee_s.par.ee_iter
                    && ee_s.last_res
                       > 10 * ee_s.par.ee_tol )
                {
                    throw std::runtime_error
                    ( "edge_mg: implicit solve did not"
                      " converge; reduce dt or use"
                      " PRECISION=2" );
                }
            }
            return dev.sync_all_streams(  );
        }
        if( mode == 2 )
        {
            //  Implicit ( face-centered Laplacian
            // Gauss-Seidel, implicit.h ).   Refresh the
            // primitives first ( the diffusivity reads w ).
            auto & dev ( * mod.p_dev );
            for( auto & d : mod )
            {
                auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                b_d.dt   = dt_use;
                dev.launch( ni_hydro_k< decltype( b_d ) >,
                            this->cons_res( b_d ),
                            d.stream, b_d );
            }
            dev.sync_all_streams(  );
            imp.solve( mod, mesh, dev, * this, eta_p,
                       dt_use, p_dt_dif );
            return;
        }
        if( mode != 1 )
            return;
        if( sts )
        {
            //  The RKC combine/roll still skips a >= n_dim,
            //  so a thin ( n_phi = 1 or any 2-D ) block would
            //  integrate B_phi with the wrong stage weights.
            //  Refuse instead of returning a silently wrong
            //  reference.
            for( auto & d : mod )
            {
                auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                if( b_d.geo.n_dim < 3 )
                {
                    throw std::runtime_error
                    ( "mhd_ee::ni:  sts = 1 is 3-D only;"
                      "  the RKC combine/roll does not"
                      " cover the thin-axis components yet." );
                }
            }
        }
        auto & dev ( * mod.p_dev );
        const int stp ( 0 );   //  post-ideal slice

        //  1.  Refresh the primitives ( slice 0 ) once
        // --  w is fixed during the sweep;  only the field
        // diffuses.
        for( auto & d : mod )
        {
            auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
            b_d.dt   = dt_use;
            dev.launch( ni_hydro_k< decltype( b_d ) >,
                        this->cons_res( b_d ),
                        d.stream, b_d );
        }

        //  Number of Chebyshev sub-steps.  Explicit ( no
        //  STS ):  one step with the full ( parabolic-
        //  limited ) dt.   STS:  the eos has relaxed the
        //  parabolic cap;  cover the diffusion with s
        //  damped sub-steps.
        int s ( 1 );
        if( sts )
        {
            type::float_t big ( 1.e30 );
            dev.cp( p_dt_dif, & big, 1 );
            for( auto & d : mod )
            {
                auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                const auto & geo = b_d.geo;
                const auto n_th = std::min
                    ( geo.n_ceff[ 0 ], this->n_th_lim );
                if( geo.n_ceff[ 0 ] % n_th != 0 )
                    throw std::runtime_error
                        ( "Indivisible stride, "
                          "mhd_ee::ni::ni_t ( STS )." );
                dev.launch( ni_dtlim_k< decltype( b_d ),
                                        eta_T >,
                    std::make_tuple
                    ( dim3( geo.n_ceff[ 1 ],
                            geo.n_ceff[ 2 ],
                            geo.n_ceff[ 0 ] / n_th ),
                      dim3( n_th, 1, 1 ),
                      n_th * sizeof( type::float_t ) ),
                    d.stream, b_d, eta_p, p_dt_dif );
            }
            dev.sync_all_streams(  );
            type::float_t dt_dif;
            dev.cp( & dt_dif, p_dt_dif, 1 );
            mesh.p_com->reduce_all( dt_dif, comm::min );
            //  Super - time - stepping  (  RKC1 ):  s is the
            //  number of damped sub-steps,  sized from the
            //  ratio  dt_use / dt_dif.
            const float_t ratio ( ( float_t ) dt_use
                                 / ( float_t ) dt_dif );
            s = ratio > 1.0 ? ( int ) std::ceil
                              ( std::sqrt( ratio ) ) : 1;
        }

        //  2-5. The RKC stages. Stage n ( = poly - degree
        //  j = n + 1 ): the tendency weight chi and ( for
        //  j >= 2 )  the recursion coefficients  mu / nu.
        float_t w0 ( 1 ), w1 ( 1.0 / s / s ), c_m2 ( 1 ),
               c_m1 ( 1 );
        if( sts && s > 1 )
        {
            w0 = 1.0 + ( float_t ) sts_nu / ( ( float_t ) s
                * s );
            const float_t phi ( std::acosh( w0 ) );
            w1 = std::cosh( s * phi ) * std::sinh( phi )
               / ( s * std::sinh( s * phi ) );
            c_m1 = w0;
        }
        for( int n = 0; n < s; ++ n )
        {
            type::float_t dts ( dt_use ), mu1 ( 0 ),
                          nu ( 0 );
            if( sts )
            {
                if( n == 0 )
                    dts = type::float_t( dt_use * w1 / w0 );
                else
                {
                    const float_t c_j
                        ( 2 * w0 * c_m1 - c_m2 );
                    mu1 = type::float_t
                        ( 2 * w0 * c_m1 / c_j - 1 );
                    nu  = type::float_t( - c_m2 / c_j );
                    dts = type::float_t
                        ( dt_use * 2 * w1 * c_m1 / c_j );
                    c_m2 = c_m1;   c_m1 = c_j;
                }
            }
            //  2.  Fill the field ghosts ( comm + phys BCs
            // ). Drain around the comm so ni_cur_k's ghost
            // reads are ordered ( see the CUDA race note
            // above ).
            dev.sync_all_streams(  );
            mod.p_bdk->act_bnd( mesh, stp );
            dev.sync_all_streams(  );
            //  3.  Edge current -> diffusion emf into ee.
            for( auto & d : mod )
            {
                auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                b_d.dt   = dts;
                const auto res = this->field_res( b_d );
                dev.launch
                    ( ni_cur_k< decltype( b_d ), ni_t >,
                      res, d.stream, b_d, * this, bc_edge );
                dev.launch( ni_emf_k< decltype( b_d ), ni_t,
                                      eta_T, false >,
                            res, d.stream, b_d, * this,
                            eta_p, bc_edge,
                            type::float_t( hall_chyp ),
                            type::float_t( hall_upw ) );
            }
            //  4.  Edge exchange reconciles the boundary
            //     diffusion emfs.
            dev.sync_all_streams(  );
            mod.p_bdk->act_flx( mesh, stp );
            dev.sync_all_streams(  );
            //  5.  The Euler part ( intg ) + heating;  then
            // the combine ( stages >= 2 )  and the roll.
            for( auto & d : mod )
            {
                auto b_d ( dynamics::mhd::proxy_t::d( d.d(  ) ) );
                b_d.dt   = dts;
                const auto cres = this->cons_res( b_d );
                this->intg_field_only( b_d, d, stp, dev );
                dev.launch( ni_ene_k< decltype( b_d ), ni_t,
                                      eta_T >,
                            cres, d.stream, b_d, * this,
                            eta_p, 0, n == 0 );
                if( sts && n > 0 )
                    dev.launch( ni_rkc_comb_k< decltype( b_d ) >,
                                this->field_res( b_d ),
                                d.stream, b_d, mu1, nu );
                dev.launch( ni_rkc_roll_k< decltype( b_d ) >,
                            this->field_res( b_d ),
                            d.stream, b_d );
            }
        }
        return dev.sync_all_streams(  );
    };

    //  Conserved-field launch resource:  one thread per
    //  cell.
    template< class bdt_T >
    __host__ std::tuple< dim3, dim3, int > cons_res
    ( const bdt_T & b_d )                        const
    {
        const auto & geo = b_d.geo;
        const auto n_th  = std::min( geo.n_ceff[ 0 ],
                                     this->n_th_lim );
        if( geo.n_ceff[ 0 ] % n_th != 0 )
            throw std::runtime_error
                ( "Indivisible stride, "
                  "mhd_ee::ni::ni_t." );
        return std::make_tuple
            ( dim3( geo.n_ceff[ 1 ], geo.n_ceff[ 2 ],
                    geo.n_ceff[ 0 ] / n_th ),
              dim3( n_th, 1, 1 ), 0 );
    };
};

////////////////////////////////////////////////////////////
//  EOS wrapper adding the parabolic time-step constraint
//  ( Athena++ NewFieldDiffusionDt ):
//      dt_oa = fac_oa * d^2 / ( eta_O + eta_A ),
//      dt_h  = fac_h  * d^2 / | eta_H |,
//  with fac_oa = 1 / ( 2 n_dim ), fac_h = 1 ( multi-D )
//  and d the CFL cell size min_a dx_cfl( a, i ).
template< class eta_T  = const_eta_t,
          class derived_T = crtp::dummy_t >
struct ni_eos_t : dynamics::mhd::eos::adiabatic_t
    < crtp::helper< ni_eos_t, eta_T, derived_T > >
{
    using  this_t = ni_eos_t< eta_T, derived_T >;
    using super_t = dynamics::mhd::eos::adiabatic_t
        < crtp::helper< ni_eos_t, eta_T, derived_T > >;

    eta_T eta_p;
    //  Super-time-stepping:  when on,  the Ohmic /
    // ambipolar ( true-diffusion )  cap is handled by the
    // STS sweep,  so only the Hall ( dispersive,  not
    // covered by STS ) cap is applied here.   Likewise the
    // implicit ( CG )  solve is unconditionally stable,  so
    // the cap is relaxed when mode == 2.
    bool    sts { false };
    bool    imp { false };
    //  ee_enable + ee_dt > 0 forces the dt:  the
    //  implicit edge solve gets the dt it was asked for,
    //  not the parabolic cap.
    bool          ee_on { false };
    type::float_t  ee_dt { 0 };

    __host__ virtual void read( const input & args )
    {
        super_t::read( args );
        eta_p.read( args );
        sts = args.get< int >( "nonideal", "sts", 0 ) != 0;
        imp = args.get< int >( "nonideal", "mode", 0 ) == 2;
        ee_on = args.get< int >
              ( "nonideal", "ee_enable", 0 ) != 0;
        ee_dt = args.get< type::float_t >
              ( "nonideal", "ee_dt", 0 );
        //  No cap - relaxation clause here:  the implicit
        //  curl - curl leg that would have needed one is not
        //  instantiated in this build.
        return;
    };

    template< class f_T, class bdt_T, class idx_T >
    __device__ f_T dt
    ( const f_T * v_p, const coord_t & bc_2,
      const coord_t & b_m, const f_T & gamma,
      const idx_T & i, const bdt_T & b_d )    const
    {
        const super_t & sup = * this;
        auto res = sup.dt( v_p, bc_2, b_m, gamma,
                           i, b_d );
        //  ee_dt is the forced-dt override:  the cycle dt and
        //  the diffusion dt must be the same number,  so return
        //  it here as well ( not just in ni_t::diffuse ).  The
        //  caller owns the ideal CFL when it forces this.
        if( ( ee_on || sts ) && ee_dt > 0 )
            return f_T( ee_dt );
        if( ! eta_p.on )
            return res;
        type::float_t eo, eh, ea;
        eta_p.eval( b_d, i, eo, eh, ea );
        const auto & geo = b_d.geo;
        f_T d ( geo.dx_cfl( 0, i ) );
        for( int a = 1; a < geo.n_dim; ++ a )
            d = __min1( d, geo.dx_cfl( a, i ) );
        const f_T fac_oa = f_T( 0.5 ) / geo.n_dim;
        //  The whistler cap tightens the parabolic limit here.
        const f_T fac_h  = f_T( 0.05 );
        if( ! sts && ! imp
            && !( ee_on && ee_dt > 0 )
            && eo + ea > 0 )
            res = __min1( res, fac_oa * d * d
                               / ( eo + ea ) );
        //  The whistler cap:  held for the plain explicit /
        //  implicit legs  ( the fallback;  tightened H2 );
        //  RELAXED under STS ( the RKC recursion with the
        //  Hall stiffness in  ni_dtlim_k's sizing  covers
        //  the near - Nyquist whistler spectrum ).
        if( eh != 0 && ! sts )
            res = __min1( res, fac_h * d * d
                               / __abs1( eh ) );
        return res;
    };
};

};  // namespace mhd_ee::ni
