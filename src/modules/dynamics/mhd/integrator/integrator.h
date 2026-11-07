#pragma once

#include "../../hydro/integrator/integrator.h"
#include "data.h"
#ifdef DBG_CT_ID_DBG
#include "ct_id_dbg.h"
#endif

namespace dynamics::mhd::integrator
{
////////////////////////////////////////////////////////////
// Kernels for field calculations

template< class bdt_T, class itg_T > __global__ 
void field_calc( const bdt_T b_d, const itg_T f_itg )
{
    return f_itg.calc_field( b_d );
}

template< class bdt_T, class itg_T >
__global__ void field_intg
( const bdt_T b_d, const int step, const itg_T f_itg )
{
    return f_itg.intg_field( b_d, step );
}

////////////////////////////////////////////////////////////
// Integrator for hydrodynamics problems; using CRTP

using dual_t = mesh::block::dual_t;

template< class derived_T = crtp:: dummy_t >
struct base_t : hydro::integrator::base_t
              < crtp::helper< base_t, derived_T > >
{
    ////////// CRTP functions //////////
    __crtp_def_self__( base_t, derived_T );
    using super_t = hydro::integrator::base_t
                  < crtp::helper< base_t, derived_T > >;

    ////////// Resource //////////
    template< class  bdt_T >
    __host__ std:: tuple< dim3, dim3,  int >     resource
    ( const  bdt_T & b_d, const int & axis )        const;
    template< class  bdt_T >
    __host__ std:: tuple< dim3, dim3,  int >    field_res
    ( const  bdt_T & b_d )                          const;

    ////////// Field calculation //////////
    template< class bdt_T > __device__ void    calc_field
    ( const bdt_T & b_d )                           const;
    template< class bdt_T > __device__ void    intg_field
    ( const bdt_T & b_d, const int & step )         const;

    //  Curvilinear CT hooks.  edg_f: physical length
    //  of the edge "ax" at the corner "i" ( the edge
    //  runs along ax; i holds the face positions of
    //  the other two axes ).  sfc_thin: area of the
    //  thin-axis face family "a" ( a >= n_dim, where
    //  meshgen stores sfc only for the active axes ).
    //  Cartesian base: the coordinate spacing, resp.
    //  its in-plane product, which reproduces the
    //  exact face geometry.  Curvilinear integrators
    //  ( sph_t / cyl_t ) override both with the
    //  metric factors.
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto edg_f
    ( const int & ax, const idx_T & i,
      const geo_T & geo )                    const
    {
        return geo.dx_f( ax, i );
    };
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto sfc_thin
    ( const int & a, const idx_T & i,
      const geo_T & geo )                    const
    {
        const int ax_1 ( utils::mod3( a + 1 ) );
        const int ax_2 ( utils::mod3( a + 2 ) );
        return geo.dx_f( ax_1, i ) * geo.dx_f( ax_2, i );
    };

    //  Dual edge length for the non-ideal current ( the
    //  discrete adjoint of the CT curl ): the physical
    //  width along "ax" of the segment CENTRED ON THE
    //  FACE i[ ax ] ( i carries cell positions of the
    //  other two axes ).  Cartesian: the face-centred
    //  spacing; curvilinear integrators override with the
    //  metric factors evaluated at the cell positions.
    template< class idx_T, class geo_T >
    __device__ __forceinline__ auto edg_d
    ( const int & ax, const idx_T & i,
      const geo_T & geo )                    const
    {
        auto j( i ); -- j[ ax ];
        return ( geo.dx_f( ax, j ) + geo.dx_f( ax, i ) )
               / 2;
    };

    ////////// Device-side interface //////////
    template< class dat_T , class bdt_T, class eos_T >
    __device__ void operator (  )
    ( dat_T &, const bdt_T &, const eos_T & )  const;

    ////////// Host-side interface //////////
    template < class bdt_T,  class  eos_T > __host__ void
    intg_flx( const bdt_T &, const dual_t & , const eos_T &,
              const int   &, device::base_t & ) const;

    template< class bdt_T,   class  eos_T ,
              class rec_T,   class  rie_T > __host__ void
    calc_flx( const bdt_T &, const dual_t &, const eos_T &,
              const rec_T &, const  rie_T &, const int   &,
              const device::base_t  &  )     const;

    //  Singular-locus edge-emf regularisation,  applied
    //  after the E-field exchange and before the CT curl.
    //  Cartesian: no-op;  sph_t / cyl_t shadow this with
    //  the pole / axis ring average ( CRTP dispatch via
    //  get_self,  NOT virtual ).   Used by intg_flx and by
    //  the operator-split diffusion sweep.
    template< class bdt_T > __host__ void
    emf_fix( const bdt_T &, const dual_t &,
             device::base_t & )        const {  };

    //  Field-only CT update ( no hydro divergence ):  the
    //  non-ideal diffusion sweep advances ONLY the field
    //  by the curl of the ( exchanged,  emf_fix-treated )
    //  diffusion emf.
    template< class bdt_T > __host__ void
    intg_field_only( const bdt_T &, const dual_t &,
                     const int &, device::base_t & ) const;
};

////////////////////////////////////////////////////////////
// Resource of computing: allocation and dispatching

template< class derived_T >
template< class     bdt_T >  __host__
std::tuple<  dim3, dim3, int > base_t< derived_T >::resource
( const bdt_T & b_d, const int & axis  ) const
{
    const auto & geo    = b_d.geo;
    const auto & n_ceff = b_d. geo. n_ceff;
    const int    n_hvar = b_d. hyd. u.n_int;
    const int    n_th   = std::min( n_ceff  [ axis ],
                                    this->n_th_lim );
    if( n_ceff[ axis ] % n_th != 0 )
        throw std::runtime_error( "Indivisible stride: mhd:"
                                  ":integrator::base_t. " );
    const int     ax_1( ( axis + 1 ) % 3 );
    const int     ax_2( ( axis + 2 ) % 3 );
    int  n_bl =   n_ceff[ ax_1 ]     * n_ceff[ ax_2 ] ;
    n_bl += 2 * ( n_ceff[ ax_1 ] > 1 ? n_ceff[ ax_2 ] : 0 );
    n_bl += 2 * ( n_ceff[ ax_2 ] > 1 ? n_ceff[ ax_1 ] : 0 );

    const int & n_feff  = n_th   + 1;
    const int & n_ctot  = n_th   + 2 * b_d.geo.n_gh;
    const int & n_ftot  = n_ctot + 1;
    using  f_t = typename decltype( b_d.hyd )::f_t;
    int s_sh = ( n_ctot + 3 * n_feff  +   1 )
             * ( n_hvar + 3 ) * sizeof( f_t );
    if( ! geo.is_uniform[ axis ] )
        s_sh +=  n_ftot * 2   * sizeof( f_t );

    const dim3 n_bl_( n_bl, 1,   n_ceff[ axis ] / n_th );
    return std::make_tuple( n_bl_, n_th , s_sh );
}

template< class derived_T >
template< class     bdt_T >  __host__
std::tuple<  dim3, dim3, int > base_t< derived_T >::
field_res( const bdt_T & b_d ) const
{
    const int n_th = std:: min
        ( b_d.geo. n_ceff[ 0 ] , this->n_th_lim );
    if  ( b_d.geo. n_ceff[ 0 ] % n_th != 0 )
        throw std::runtime_error( "Indivisible stride: mhd"
                                  "::integrator::base_t." );
    const  dim3 n_bl ( b_d.geo.n_ceff[ 1 ] + 1,
                       b_d.geo.n_ceff[ 2 ] + 1,
                       b_d.geo.n_ceff[ 0 ] / n_th );
    return std::make_tuple( n_bl, dim3( n_th ), 0 );
}

////////////////////////////////////////////////////////////
// Field calculation and constrained-transport integration

template < class derived_T >
template < class     bdt_T > __device__ void
base_t< derived_T >::calc_field( const bdt_T & b_d ) const
{
    const int n_feff  = blockDim.x +
        ( blockIdx.z ==  gridDim.z - 1 ? 1 : 0 );
    const int n_dm    ( b_d.geo.n_dim );

    using f_t = typename decltype( b_d.fld )::f_t;
    type::idx_t i = { 0, blockIdx.x, blockIdx.y };
    f_t ee[ 3 ];
    f_t ec[ 2 ][ 2 ];
    auto f_ec = [ & ] ( const auto & j, const auto & ax )
    {
        bool is_in( true );
        for( int a = 0; a < 3 ; ++ a )
            is_in &= j[ a ] >= 0
                  && j[ a ] <  b_d.geo.n_ceff[ a ];
        if( is_in )
            return b_d.fld.ec.at( j )[ ax[ 0 ] ];
        const auto * b = b_d.hyd.w.at( j ) + idx::bcc;
        const auto * v = b_d.hyd.w.at( j ) + idx::mom;
        return b[ ax[ 1 ] ] * v[ ax[ 2 ] ]
             - b[ ax[ 2 ] ] * v[ ax[ 1 ] ];
    };

    type::idx_t  axis;
    int & ax   = axis[ 0 ];
    int & ax_1 = axis[ 1 ];
    int & ax_2 = axis[ 2 ];
    const auto & f   = b_d.hyd.f;
    for( int j = threadIdx.x; j < n_feff; j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j ;
        //  Thin axes ( >= n_dm, kept last by convention )
        //  carry a single edge layer; the +1 layer would
        //  read fluxes that do not exist, and nothing
        //  consumes it.
        bool th_skip( false );
        for( int a = n_dm; a < 3; ++ a )
            th_skip |= i[ a ] > 0;
        if( th_skip )
            continue;
        ee[ 0 ] = ee[ 1 ] = ee[ 2 ] = 0;
        for( ax = 0; ax < 3; ++ ax )
        {
            if( i[ ax ] >= b_d.geo.n_ceff[ ax ] )
                continue;
            ax_1 = utils :: mod3( 1  + ax );
            ax_2 = utils :: mod3( 2  + ax );
            //  Thin directions have coincident edge
            //  corners: skip the shifts so f_ec never
            //  reads a ghost layer that does not exist.
            const bool th_1 ( ax_1 >= n_dm );
            const bool th_2 ( ax_2 >= n_dm );
            ec [ 1 ][ 1 ] = f_ec( i, axis );
            if( ! th_1 )
            -- i [ ax_1 ] ;
            ec [ 1 ][ 0 ] = f_ec( i, axis );
            if( ! th_2 )
            -- i [ ax_2 ] ;
            ec [ 0 ][ 0 ] = f_ec( i, axis );
            if( ! th_1 )
            ++ i [ ax_1 ] ;
            ec [ 0 ][ 1 ] = f_ec( i, axis );
            if( ! th_2 )
            ++ i [ ax_2 ] ;
            ee   [ ax   ] = 0;

            //  These "para" and "perp" subscripts for a
            //  ("axis") denote the direction of
            //  derivatives.

            //  A thin axis has neither a sweep ( a_para )
            //  nor transverse flux lines ( a_perp ): the
            //  edge then takes the full weight from the
            //  surviving sweep, with no bracket average
            //  and no upwind corner correction.
            const f_t w_e ( ax_1 >= n_dm || ax_2 >= n_dm
                          ? 0.5f : 0.25f );
            for( const auto & a_para : { ax_1 , ax_2 } )
            {
                const auto a_perp
                    ( a_para == ax_1 ?   ax_2 : ax_1 ) ;
                const bool thn_p ( a_perp >= n_dm ) ;
                //  A thin sweep direction has no flux
                //  array at all ( hyd_data allocates f
                //  only for a < n_dim ): skip its whole
                //  contribution without touching f.
                if( a_para < n_dm )
                {
                    if( ! thn_p )
                        -- i[ a_perp ];
                    const auto ef_m = f[ a_para ].at
                                  ( i )[ idx::bcc + ax ];
                    if( ! thn_p )
                        ++ i[ a_perp ];
                    const auto ef_p = f[ a_para ].at
                                  ( i )[ idx::bcc + ax ];

                    ee[ ax ] += w_e * ( ef_m + ef_p ) ;

                    if( ! thn_p )
                    {
                        const auto frho_p = f[ a_perp ].at
                            ( i )[ idx::rho ];
                        -- i[ a_para ];
                        const auto frho_m = f[ a_perp ].at
                            ( i )[ idx::rho ];
                        ++ i[ a_para ];

                        const f_t de_p
                            ( frho_p > 0
                              ?  ( ec[ 0 ][ 1 ] - ef_m )
                              :  ( ec[ 1 ][ 1 ] - ef_p ) );
                        const f_t de_m
                            ( frho_m > 0
                              ? -( ec[ 0 ][ 0 ] - ef_m )
                              : -( ec[ 1 ][ 0 ] - ef_p ) );
                        ee[ ax ] += w_e * ( de_m - de_p ) ;
                    }
                }

                if( a_para != ax_1 )  // Transpose ec
                    continue;
                const auto swp = ec[ 0 ][ 1 ];
                ec[ 0 ][ 1 ]   = ec[ 1 ][ 0 ];
                ec[ 1 ][ 0 ]   = swp;
            }
        }
        auto * ee_g = b_d.fld.ee.at( i );
        //  Store only the components this block computes
        //  ( i[ a ] < n_ceff[ a ] ).  Components at the +-
        //  boundary edge rows are delivered by the E-field
        //  exchange; storing them here would clobber those
        //  slots with stale stack debris.
        for( int  a = 0; a < 3; ++  a )
            if( i[ a ] < b_d.geo.n_ceff[ a ] )
                ee_g[ a ] = ee[ a ];
        //  The boundary - plane advective edge EMF  ( the
        //  r_out field - anchoring fix ):  on an out / fre
        //  side  ( geo.bnd_out )  the boundary - plane
        //  edges would otherwise keep the init ZERO  ( no
        //  functor writes them )  -- a zero - E wall that
        //  piles the wound field up against the boundary
        //  ( the reverse - current sheet  ->  eta_A cap
        //  ->  anti - diffusion  ->  blow - up chain ).
        //  Pure advection,  first order,  NO ghost
        //  reads,  no corner terms:  the interior cell 's
        //  E = -( v x Bbar )  with  v  from  w  ( cons2prim
        //  - fresh )  and  Bbar  the cell 's face - average
        //  field.   Interface planes stay untouched  (
        //  act_flx owns them ).   Inflow sides  ( the
        //  outflow functor 's veto  zeroes the normal  v )
        //  degrade continuously to the old  E ~ 0.
        for( int  a = 0; a < 3; ++  a )
        {
            const int n( b_d.geo.n_ceff[ a ] );
            //  Only the HIGH boundary - plane edge rows
            //  ( i[ a ] == n )  are unwritten by the base
            //  stencil  ( the low plane 's  i == 0  rows
            //  are ordinary interior edges ).   No 2 -
            //  cell axes.
            if( i[ a ] != n || n < 2 )
                continue;
            const int s( 1 );
            if( ! b_d.geo.phybnd_flag[ a ][ s ]
                || ! b_d.geo.bnd_out[ a ][ s ] )
                continue;
            //  The interior cell behind the boundary plane.
            type::idx_t jc( i );   -- jc[ a ];
            const auto * w
                ( b_d.hyd.w.at( jc ) );
            const f_t irho( f_t( 1 )
                / fmax( f_t( w[ idx::rho ] ),
                        f_t( 1e-30 ) ) );
            f_t v_c[ 3 ], b_c[ 3 ];
            for( int q = 0; q < 3; ++ q )
            {
                v_c[ q ] = w[ idx::mom + q ] * irho;
                b_c[ q ] = w[ idx::bcc + q ];
            }
            //  The two tangential edge components  a1 /
            //  a2  at this  a - normal boundary plane:
            //      E_{a1} = v_a B_{a2} - v_{a2} B_a
            //      E_{a2} = v_{a1} B_a - v_a B_{a1}
            //  ( the  -( v x B )  cell - center value,
            //  the ec  sign convention:   B x v ).
            const int a1( utils::mod3( a + 1 ) );
            const int a2( utils::mod3( a + 2 ) );
            ee_g[ a1 ] =   v_c[ a  ] * b_c[ a2 ]
                         - v_c[ a2 ] * b_c[ a  ];
            ee_g[ a2 ] =   v_c[ a1 ] * b_c[ a  ]
                         - v_c[ a  ] * b_c[ a1 ];
        }
    }
    return;
}

template< class derived_T >
template< class     bdt_T > __device__
void base_t   < derived_T >::intg_field
( const bdt_T & b_d, const int & step ) const
{
    using  f_t = typename decltype( b_d.fld ):: f_t;
    using f2_t = typename decltype( b_d.fld )::f2_t;
    int i[ 3 ] ;
    i    [ 1 ] = blockIdx.x;
    i    [ 2 ] = blockIdx.y;

    const auto & self = get_self(  );
    const int  n_feff = blockDim.x +
        ( blockIdx. z == gridDim.z - 1 ? 1 : 0 );
    const auto & ee   = b_d.fld.ee;
    const auto & geo  = b_d.geo;
    const int  n_dm ( geo.n_dim );

    f2_t db_v[ 3 ];
    for( int j = threadIdx.x; j < n_feff; j += blockDim.x )
    {
        i[ 0 ] = blockDim.x * blockIdx.z + j;
        const int a_max( j == blockDim.x ? 1 : 3 );
        unsigned  msk( 0 );
        for(  int a = 0; a <  a_max;       ++  a )
        {
            const int  ax_1  = utils::mod3( a + 1 );
            const int  ax_2  = utils::mod3( a + 2 );
            if( i[ ax_1 ] >= geo.n_ceff[ ax_1 ] ||
                i[ ax_2 ] >= geo.n_ceff[ ax_2 ]  )
                continue;
            msk |= 1u << a;

            //  A thin axis ( >= n_dim, kept last by
            //  convention ) has no variation along it:
            //  the curl pair spanning that axis vanishes
            //  identically and its ee layer is never
            //  computed, so drop the pair instead of
            //  reading it. The thin face family's area
            //  is not in sfc0 ( meshgen fills only the
            //  active axes ); the inline dx_f product
            //  reproduces surface( ) exactly.
            const bool th_1 ( ax_1 >= n_dm );
            const bool th_2 ( ax_2 >= n_dm );
            auto & db( db_v[ a ] );
            if( ! th_2 )
                db  = ( f2_t )( ee.at( i )[ ax_1 ] *
                        self.edg_f ( ax_1, i, geo ) );
            else
                db  = ( f2_t )( 0 );
            if( ! th_1 )
            db -= ( f2_t )( ee.at( i )[ ax_2 ] *
                    self.edg_f ( ax_2, i, geo ) );
            ++  i [ ax_1 ];
            if( ! th_1 )
            db += ( f2_t )( ee.at( i )[ ax_2 ] *
                    self.edg_f ( ax_2, i, geo ) );
            --  i [ ax_1 ];
            ++  i [ ax_2 ];
            if( ! th_2 )
            db -= ( f2_t )( ee.at( i )[ ax_1 ] *
                    self.edg_f ( ax_1, i, geo ) );
            --  i [ ax_2 ];
            //  Faces of zero or negative area carry no
            //  flux: the polar theta-faces ( sin theta =
            //  0; the south pole goes negative in float )
            //  are gauge and keep their initial value.
            //  The BC owns them.
            const auto sfc ( a >= n_dm
                           ? self.sfc_thin ( a, i, geo )
                           : geo.surface ( i, a ) );
            db  *= ( sfc > 0 ? ( f2_t )( -b_d.dt / sfc )
                             : ( f2_t )( 0 ) );
        }
        self.set_field( b_d.fld, i, step, db_v, msk );
    }
    return;
}

////////////////////////////////////////////////////////////
// Device-side interface


template < class derived_T >
template < class dat_T, class bdt_T, class eos_T >
__device__ void base_t< derived_T >::operator(  )
( dat_T & d, const bdt_T & b_d, const eos_T & eos ) const
{
    __syncthreads(  );
    get_self(   ).save_flux( d, b_d );
    if( d.axis == 0 )
    {
        int  i[ 3 ];
        d.dispatch( i, b_d );
        if( i[ 1 ] >= 0 && i[ 1 ] < b_d.geo.n_ceff[ 1 ] &&
            i[ 2 ] >= 0 && i[ 2 ] < b_d.geo.n_ceff[ 2 ]  )
            get_self(  ).add_flux ( d,  b_d, eos );
    }
    return;
}

////////////////////////////////////////////////////////////
// Invoke the integrator

template< class derived_T >
template< class bdt_T, class eos_T,
          class rec_T, class rie_T >
__host__  void base_t< derived_T >::calc_flx
( const bdt_T &   b_d, const dual_t &     d,
  const eos_T & f_eos, const rec_T  & f_rec,
  const rie_T & f_rie, const int    &  step,
  const device::base_t & dev )        const
{
    const auto  & f_itg  = get_self(  );
    using   f_t = typename decltype( b_d.hyd )::  f_t  ;
    using  f2_t = typename decltype( b_d.hyd ):: f2_t  ;
    using dat_t = data_t < decltype( b_d ), f_t, f2_t >;
    for( int ax = b_d.geo.n_dim - 1; ax  >= 0 ; -- ax )
        dev.launch
        (  ::dynamics::integrator::flux_kernel  < dat_t >,
           f_itg.resource( b_d, ax ),   d.stream,   b_d  ,
           ax,   step, f_rec,  f_rie,   f_eos,    f_itg );
    return dev.launch( field_calc, f_itg.field_res( b_d ),
                       d.stream,   b_d,      f_itg      );
}

//  Thin - axis face sync.   A PERIODIC axis with a single
//  cell has no neighbour to exchange with:  its "wrap"
//  slots ( index 1 .. n_gh ) must BE the slot - 0 face.
//  Left stale they feed the corner EMFs with a phantom
//  face and the CT stops telescoping --  the  n_phi = 1
//  ( true 2.5D,  n_dim = 2 )  divergence blow - up of
//  2026-10-21.   Runs after every CT update.
template< class bdt_T >
__global__ void thin_axis_sync_k( const bdt_T b_d )
{
    const auto & geo( b_d.geo );
    const int t( blockIdx.x * blockDim.x + threadIdx.x );
    for( int a = 0; a < 3; ++ a )
    {
        if( geo.n_ceff[ a ] != 1 )
            continue;
        if(    geo.phybnd_flag[ a ][ 0 ]
            || geo.phybnd_flag[ a ][ 1 ] )
            continue;
        const int a1( utils::mod3( a + 1 ) );
        const int a2( utils::mod3( a + 2 ) );
        const int g ( geo.n_gh );
        const int N1( geo.n_ceff[ a1 ] + 2 * g );
        const int N2( geo.n_ceff[ a2 ] + 2 * g );
        if( t >= N1 * N2 )
            continue;
        type::idx_t i;
        i[ a  ] = 0;
        i[ a1 ] =   t % N1 - g;
        i[ a2 ] =   t / N1 - g;
        auto * d0( b_d.fld.bf.at( i ) );
        for( int k = -g; k <= g; ++ k )
        {
            if( k == 0 )
                continue;
            type::idx_t j( i );
            j[ a ] = k;
            auto * d1( b_d.fld.bf.at( j ) );
            d1[ 0 ] = d0[ 0 ];
            d1[ 1 ] = d0[ 1 ];
            d1[ 2 ] = d0[ 2 ];
        }
    }
    return;
};

template< class derived_T >
template< class     bdt_T, class eos_T >
__host__  void base_t< derived_T > ::intg_flx
( const bdt_T &  b_d, const   dual_t & d, const eos_T & eos,
  const int   & step, device::base_t & dev ) const
{
    super_t::intg_flx( b_d, d, eos, step, dev ) ;
    const auto & f_itg = get_self(  );
    dev.launch( field_intg, f_itg.field_res( b_d ),
                d.stream, b_d, step, f_itg );
#ifdef DBG_CT_ID_DBG
    //  Forward - CT telescoping probe  ( ct_id_dbg.h ):
    //  step 0 only,  after this step 's ee refresh.
    ct_id_dbg_run( b_d, f_itg, dev, d.stream, step );
#endif
    //  Full - domain grid:  field_res sizes for the face
    //  arrays,  not for the ( transverse x transverse )
    //  slot count.
    int n_t( 0 );
    for( int a = 0; a < 3; ++ a )
    {
        if( b_d.geo.n_ceff[ a ] != 1 )
            continue;
        if(    b_d.geo.phybnd_flag[ a ][ 0 ]
            || b_d.geo.phybnd_flag[ a ][ 1 ] )
            continue;
        const int g( b_d.geo.n_gh );
        n_t = std::max
            ( n_t, ( int( b_d.geo.n_ceff[ ( a + 1 ) % 3 ] )
                     + 2 * g )
                 * ( int( b_d.geo.n_ceff[ ( a + 2 ) % 3 ] )
                     + 2 * g ) );
    }
    if( n_t > 0 )
    return dev.launch( thin_axis_sync_k< bdt_T >,
                       std::make_tuple
                       ( dim3( ( n_t + 127 ) / 128 ),
                         dim3( 128 ), 0 ),
                       d.stream, b_d );
}

template< class derived_T >
template< class     bdt_T >
__host__  void base_t< derived_T > ::intg_field_only
( const bdt_T &  b_d, const   dual_t & d,
  const int   & step, device::base_t & dev ) const
{
    //  Field-only CT update:  regularise the singular-
    //  locus edge emf ( pole / axis )  then advance bf by
    //  -dt * curl( ee ).   No hydro divergence.
    const auto & f_itg = get_self(  );
    f_itg.emf_fix( b_d, d, dev );
    return dev.launch( field_intg, f_itg.field_res( b_d ),
                       d.stream, b_d, step, f_itg );
}

};              // namespace dynamics::mhd::integrator
