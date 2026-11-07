#pragma once

#include "fld_data.h"

namespace rad::fld
{

////////////////////////////////////////////////////////////
//  FLD operator:  the scalar  ( I - dt div D grad + sig )
// stencil.   Only  weight( )  is overridden  ( n_fld = 1 );
// the base smooth / residual / defect / coarsen kernels
// pick it up automatically.
//  Stencil algebra, boundary closure, and the curvilinear
// metric factors:  docs/radiative_transfer/fld-method.md
// § 2.

struct fld_operation_t
    : multigrid::operation::operation_t< fld_operation_t >
{
    using super_t = multigrid::operation::operation_t
                    < fld_operation_t >;

    ////////// Data //////////
    bool                     var_coef;
    bool                       m1_on;   //  closure = 1
    bool                      vac_bnd; //  vacuum bnd
    bool                     sig_elim; //  LTE - elim
    bool                      newton;  //  LTE-Newton 5sig
    type::float_t            c_free4;  //  c / 4  ( Marshak )
    bool                       steady; //  drop E^n (eq.)
    type::float_t           elim_sig; //  the LTE mask
    type::float_t             d_unif;   //  D * dt
    type::float_t             s_unif;   //  sig
    //  Robin reflection g1 per axis / side  ( 0 = off ):
    int              robin[ 3 ][ 2 ];
    type::float_t    rbnd [ 3 ][ 2 ];

    ////////// Functions //////////
    __host__ fld_operation_t(  )
    {
        var_coef = false;
        m1_on    = false;
        vac_bnd  = false;
        sig_elim = false;
        newton   = false;
        c_free4  = type::float_t( 0 );
        steady   = false;
        elim_sig = type::float_t( 1 );
        d_unif   = type::float_t( 0 );
        s_unif   = type::float_t( 0 );
        for( int a = 0; a < 3; ++ a )
            for( int k = 0; k < 2; ++ k )
            {
                robin[ a ][ k ] = 0;
                rbnd [ a ][ k ] = type::float_t( 1 );
            }
        return;
    };

    //  Two - cell face mean of the frozen  co_arr[ 0 ]
    // ( the mirror of the nonideal  r_pair_face;  single -
    // block:  periodic wrap,  no interface masks ).
    //  The neighbour index along  ( a, k ):  the ghost
    // slot when out of range  (  SMR / periodic ghost
    // filled by the driver's  fill_co_ghost  /  the MG
    // comms;  the wrap of the single - block stage is
    // gone ).
    template< class idx_T >
    __device__ __forceinline__
    static int idx_nb
    ( const idx_T & i, const int & a, const int & k )
    {
        return k == 0 ? i[ a ] - 1 : i[ a ] + 1;
    };

    //  The M1 diagonal slot of axis  a  (  chi_aa ):
    //  the lower - triangle  xx, xy, xz, yy, yz, zz.
    static __device__ __forceinline__
    int chi_slot( const int & a )
    {
        return a == 0 ? 2 : ( a == 1 ? 5 : 7 );
    };

    template< class bdt_T, class idx_T >
    __device__ __forceinline__
    auto c_face
    ( const bdt_T & bdt, const idx_T & i,
      const int & a, const int & k ) const
    {
        //  The FACE coefficient at a  D / sig  jump (
        // tau_cell >> 1 across the disk / corona edge ):
        // the  arithmetic  mean  over - conducts  (  the
        // corona's  huge  D  dominates  and  short -
        // circuits  the  rim  to  the  boundary  --  the
        // 2026 - 10 - 14  convergence  audit ).   The
        // HARMONIC  mean  is  the  correct  series -
        // resistor  flux  (  D_face = 2 D_i D_j / ( D_i
        // + D_j )  --  the  flux  is  bottlenecked  by
        // the  opaque  side,  as  the  physics  demands ).
        const int n( bdt.geo.n_ceff[ a ] );
        if( m1_on )
        {
            const int s( chi_slot( a ) );
            if( n < 2 )
                return bdt.co_arr.at( i )[ s ];
            const int j( idx_nb( i, a, k ) );
            auto idx_j( i );
            idx_j[ a ] = j;
            const auto di( bdt.co_arr.at( i     )[ s ] );
            const auto dj( bdt.co_arr.at( idx_j )[ s ] );
            return type::float_t( 2 ) * di * dj
                / ( di + dj + type::float_t( 1e-30 ) );
        };
        if( n < 2 )
            return bdt.co_arr.at( i )[ 0 ];
        const int j( idx_nb( i, a, k ) );
        auto idx_j( i );
        idx_j[ a ] = j;
        const auto di( bdt.co_arr.at( i     )[ 0 ] );
        const auto dj( bdt.co_arr.at( idx_j )[ 0 ] );
        return type::float_t( 2 ) * di * dj
            / ( di + dj + type::float_t( 1e-30 ) );
    };

    //  The frozen per - cell absorption diagonal.
    template< class bdt_T, class idx_T >
    __device__ __forceinline__
    auto sig_i( const bdt_T & bdt, const idx_T & i ) const
    {
        return bdt.co_arr.at( i )[ 1 ];
    };

    //  Curvilinear metric  ( g_a,  c_a )  of one axis.
    //  The table and its derivation:  fld - method.md § 2.
    template< class f_T, class bdt_T, class idx_T >
    __device__ __forceinline__ void metric
    ( f_T & g_a, f_T & c_a, const bdt_T & bdt,
      const idx_T & i, const int & a ) const
    {
        g_a = f_T( 1 );  c_a = f_T( 0 );
        const auto & geo( bdt.geo );
        const bool cyl
        ( geo.geo_type == mesh::geo_t::cylindrical );
        const bool sph
        ( geo.geo_type == mesh::geo_t::spherical );
        if( ( cyl || sph ) && a == 0 )
        {
            const f_T r( geo.x_cc( 0, i[ 0 ] ) );
            c_a = sph ? f_T( 2 ) / r : f_T( 1 ) / r;
            return;
        };
        if( ! sph )
            return;
        const f_T r ( geo.x_cc( 0, i[ 0 ] ) );
        const f_T th( geo.x_cc( 1, i[ 1 ] ) );
        const f_T sn( sin( th ) > f_T( 1e-3 )
                    ? sin( th ) : f_T( 1e-3 ) );
        if( a == 1 )
        {
            g_a = f_T( 1 ) / ( r * r );
            c_a = cos( th ) / ( r * r * sn );
        }
        else if( a == 2 )
            g_a = f_T( 1 ) / ( r * r * sn * sn );
        return;
    };

    //  One axis's flux - form weights  ( the  g_a / c_a
    // metric folded into the asymmetric  wa0 / wa1 ).
    template< class f_T, class dat_T >
    __device__ __forceinline__ void wa_axis
    ( f_T & wa0, f_T & wa1, const f_T & g_a,
      const f_T & c_a, const dat_T & dat,
      const int & a, const bool & l_b,
      const bool & r_b ) const
    {
        const f_T dx_l( dat.dx[ a ][ 0 ] );
        const f_T dx_r( dat.dx[ a ][ 1 ] );
        const f_T drp ( dx_l + dx_r );
        wa0 = g_a * f_T( 2 ) / ( dx_l * drp ) - c_a / drp;
        wa1 = g_a * f_T( 2 ) / ( dx_r * drp ) + c_a / drp;
        if( l_b ) wa0 = f_T( 0 );
        if( r_b ) wa1 = f_T( 0 );
        return;
    };

    //  One ( axis, side )  stencil entry  ( the boundary
    // sides close through the given face flux -- the row
    // sum telescopes to  1 + sig;  fld - method.md § 2 ).
    template< class f_T, class bdt_T, class idx_T >
    __device__ __forceinline__ void w_side
    ( f_T & w_c, f_T & w_ak, const f_T & wa,
      const bdt_T & bdt, const idx_T & i,
      const int & a, const int & k,
      const bool & bnd ) const
    {
        const f_T c( var_coef
            ? f_T( this->c_face( bdt, i, a, k ) )
            : f_T( this->d_unif ) );
        if( bnd )
        {
            //  Vacuum  (  E_g = 0 )  coupling  (  the
            // lte_elim equilibrium needs OPEN physical
            // boundaries --  the flux escapes at the
            //  D E / dx  rate;   closed  ( w = 0 )
            // boundaries leave no sink ).   The  r_in
            // face  ( a=0, k=0 )  stays  CLOSED  in  the
            // disk  theta  range  --  it cuts the  dense
            // interior  (  an  unphysical  drain );
            // vacuum there only above  /  below the disk
            // ... keep it simple:  closed r_in entirely
            // ( the bench's star heating is volumetric,
            // no inflow needed ):
            w_ak = f_T( 0 );
            if( vac_bnd )
            {
                //  The Marshak outflow  (  free - streaming
                // escape ):   F_out = c E / 4  --  NOT  the
                // diffusive  D E / dx  (  the  rim - wall
                // audit  2026 - 10 - 14:  the  opaque  rim
                // cells'  diffusion  throttle  undercounts
                // the boundary escape by  tau_cell ~ 6800 ).
                //  NOTE:   wa  is  ZERO  at  physical  faces
                // (  wa_axis  zeroes  it )  --  the  Marshak
                // rate  is  c / 4  per  cell  --  face  area
                // --  the  cell  volume  ratio  in  the
                // operator's  units:  c_free4 / dx  where
                // dx  is  the  cell  size  along  the  axis ):
                //  The  TAU - ADAPTIVE  gate  (  2026 - 10 - 15
                // § 3.20  fix  ( 2a ) ):  the  Marshak  rate
                // is  the  FREE - STREAMING  escape  --  it  is
                // self - consistent  ONLY  for  optically
                // thin  boundary  cells.   A  THICK  cell
                // (  tau_cell  >  1 )  is  fed  by  the
                // throttled  interior  diffusion  flux;   the
                // c E / 4  outflow  overshoots  it  by  ~tau
                // and  SUCKS  the  cell  dry  (  the  r_out
                // midplane  6 . 5 K  vs  47 K  interior  --
                // only  the  midplane  columns  (  thick )
                // collapsed,  the  polar  cells  stayed
                // physical ).   Thick  cells  fall  back  to
                // the  zero - gradient  closure  (  the
                // outflow  matches  the  interior  diffusive
                // supply  --  w  =  0  boundary,   the  E
                // profile  continues  smoothly  out ):
                //  tau_cell  from  the  frozen  co_arr  in  the
                // ROW 's  OWN  units:   the  sig  slot  [ 1 ]
                // ( c  kappa  rho  dt  --  the  absorption
                // rate  of  the  row )  vs  the  Marshak
                // escape  rate  (  c_free4 / dxc  --  both
                // add  into  the  same  w_c ).   sig  >
                // escape  <=>  the  photon  is  absorbed
                // within  the  cell  before  it  reaches  the
                // free - streaming  boundary  <=>  thick :
                const auto & geo( bdt.geo );
                const auto dxc( geo.x_cc( a, 1 )
                    - geo.x_cc( a, 0 ) );
                const auto sig_t
                    ( var_coef
                      ? f_T( this->sig_i( bdt, i ) )
                      : f_T( this->s_unif ) );
                const auto esc
                    ( this->c_free4 / dxc );
                //  OUTFLOW  faces  only  (  2026 - 10 - 16
                // § 3.20:  the  r_in  face  ( a = 0,  k = 0 )
                // must  KEEP  its  Marshak  drain  even  when
                // thick  --  it  is  the  rim 's  escape
                // channel  into  the  cavity  (  blocking  it
                // with  the  gate  piled  the  rim  energy  to
                // 5763 K  in  the  pb9fl  audit );   the  inflow
                // side's  injection  is  the  rim_inflow
                // source's  own  Marshak - pair  closure ):
                const bool out_face
                    ( k == 1 || a != 0 );
                if( ! out_face || sig_t < esc )
                    w_c += esc;
            };
            return;
        };
        w_ak = - c * wa;
        w_c += c * wa;
        return;
    };

    ////////// The stencil //////////
    //  Base - class signature  ( the scalar Poisson form;
    //  n_fld = 1 ).   Uniform weights:  the ghost sits a
    // full  dx  away  ( NOT the Poisson half - distance
    // form -- that breaks the row - sum telescoping ).
    template< class f_T, class dat_T, class bdt_T,
              class idx_T >
    __device__ __forceinline__ void weight
    ( f_T & w_c, f_T ( & w )[ 3 ][ 2 ],
      const dat_T & dat, const bdt_T  & bdt,
      const idx_T & i ) const
    {
        const auto & geo = bdt.geo;
        const int  nd( geo.n_dim );
        //  The LTE - eliminated diagonal:  the closure
        // emission ( sig x a T_m^4 )  and  the  operator
        // absorption ( sig x E )  cancel EXACTLY  in  the
        // implicit system  (  the analytic elimination ):
        // ( I - dt div D grad + sig ) E  =  E^n + sig aT^4
        //   with aT^4 ~ E  ->  ( I - dt div D grad ) E
        //      = E^n + dt S   --  the  sig  diagonal  is
        // DROPPED  and the source carries only the true
        // external deposition.   One implicit solve then
        // propagates the equilibrium GLOBALLY  (  the
        // point of the implicit method --  the  frozen -
        // E  Picard  degeneration  of  the  lagged  form
        // never arises ):
        //  sig_elim is LOCAL:  the diagonal drops only
        // where LTE holds  (  sig  above  elim_sig  --
        // the dust - bearing disk interior;   the  disk
        // exterior / corona KEEPS its  absorption  (  it
        // is what bounds the  IR  conductivity  there;
        // dropping it there turns the corona into a
        // boundary - to - boundary  shortcut  pipe  (  the
        // 2026 - 10 - 14 audit ) ):
        const f_T sig_i
            ( var_coef ? f_T( this->sig_i( bdt, i ) )
                       : f_T( this->s_unif ) );
        //  The  LTE - Newton  diagonal  (  bench_close
        // newton mode ):  the  emission  linearization
        // moves  4 sig  to the LHS  (  +5 total );
        //  sig_elim  (  the  transport - only  mode )
        // drops it  (  corona  masked  by  elim_sig ):
        f_T diag( sig_i );
        if( newton )
            diag = f_T( 5 ) * sig_i;
        else if( sig_elim )
            diag = ( sig_i > elim_sig
                     ? f_T( 0 ) : sig_i );
        w_c = ( steady ? f_T( 0 ) : f_T( 1 ) ) + diag;
        for( int a = 0; a < nd; ++ a )
        {
            const bool l_b( geo.phybnd_flag[ a ][ 0 ]
                            && ( i[ a ] == 0 ) );
            const bool r_b
            ( geo.phybnd_flag[ a ][ 1 ]
              && ( i[ a ] == geo.n_ceff[ a ] - 1 ) );
            f_T g_a, c_a;
            metric( g_a, c_a, bdt, i, a );
            f_T wa0, wa1;
            wa_axis( wa0, wa1, g_a, c_a, dat,
                     a, l_b, r_b );
            for( int k = 0; k < 2; ++ k )
            {
                const bool bnd( k == 0 ? l_b : r_b );
                const f_T  wa ( k == 0 ? wa0 : wa1 );
                w_side( w_c, w[ a ][ k ], wa,
                        bdt, i, a, k, bnd );
            }
        }
        return;
    };
};

};                          // namespace rad::fld
