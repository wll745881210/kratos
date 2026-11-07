#include "edge_mg_driver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "../../../../../../src/device/device.h"
#include "../../../../../../src/io/args/input.h"
#include "../../../../../../src/utilities/functions.h"
#include "../../../../../../src/utilities/mapping/reduction.h"

////////////////////////////////////////////////////////////
//  The implicit step's host side:  every action of one
//  step,  and the setup-time jobs it calls.
//
//  The declarations are in edge_mg_driver.h;  the phase
//  methods are private members of driver_t there,  so the
//  order of the definitions below is:  the setup-time jobs
//  first ( they are what a phase calls ),  then the phases
//  in the order a step runs them,  then step( ) itself.
//
//  Method:  docs/mhd/nonideal/algorithm-spec-edge-implicit-
//  diffusion.md.  Engineering constraints ( why this file
//  and not the pgen,  why no template on the block data,
//  why the eta tensor is data ):  docs/mhd/nonideal/
//  edge-mg-foundations.md sections 3.3 and 4.1.

namespace mhd_ee::ni::edge_mg
{

////////////////////////////////////////////////////////////
//  The setup-time jobs

//  Build the complex of a block WITH ITS POLE FLAGS.
//
//  Every cplx_t of a block goes through here ( P1 of
//  docs/mhd/nonideal/HANDOVER-ETA-CPP-DEBUG.md ). The flags
//  are a property of the GEOMETRY, not of the caller: a
//  probe that built its own complex from the two-argument
//  setup() silently disagreed with the assembler about
//  where the axis is, so the zero-length phi edges on the
//  pole ring stayed in the system as ordinary degrees of
//  freedom and `M1 + dt K` was near singular there ( the
//  gate measured a non-converging 0.17 against 1.758 once
//  folded ). One entry point, so a new probe cannot forget.
__host__ void driver_t::build_complex
( cplx_t & c, const mesh::geo_t & geo, const idx_t & n_ceff,
  const bool ( & per )[ 3 ] ) const
{
    bool lo( false ), hi( false );
    pole_flags( geo, n_ceff, lo, hi );
    c.setup( n_ceff, per, lo, hi );
    return;
};

//  the six face areas of a cell, from the policy
template< class geo_T >
__host__ void driver_t::face_areas
( const geo_T & g, const idx_t & i,
  float2_t ( & area )[ 6 ] ) const
{
    for( int s = 0; s < 6; ++ s )
    {
        const int a = s / 2;
        const int b = ( a + 1 ) % 3;
        const int l = ( a + 2 ) % 3;
        float2_t acc( 0 );
        for( int qb = 0; qb < 3; ++ qb )
        {
            for( int qc = 0; qc < 3; ++ qc )
            {
                float2_t xi[ 3 ];
                xi[ a ] = float2_t( s % 2 );
                xi[ b ] = gq_x( qb );
                xi[ l ] = gq_x( qc );
                float2_t x[ 3 ], J[ 3 ][ 3 ], det;
                g.map( i, xi, x, J, det );
                float2_t ds[ 3 ];
                ds[ 0 ] = J[ 1 ][ b ] * J[ 2 ][ l ]
                        - J[ 2 ][ b ] * J[ 1 ][ l ];
                ds[ 1 ] = J[ 2 ][ b ] * J[ 0 ][ l ]
                        - J[ 0 ][ b ] * J[ 2 ][ l ];
                ds[ 2 ] = J[ 0 ][ b ] * J[ 1 ][ l ]
                        - J[ 1 ][ b ] * J[ 0 ][ l ];
                acc += sqrt( ds[ 0 ] * ds[ 0 ] + ds[ 1 ] * ds[ 1 ]
                           + ds[ 2 ] * ds[ 2 ] )
                     * gq_w( qb ) * gq_w( qc );
            }
        }
        area[ s ] = acc;
    }
    return;
};

template< class geo_T >
__host__ void driver_t::cells_dev_eta
( device::base_t & dev, const geo_T & g, const cplx_t & c,
  const float2_t * p_eta, const float2_t * p_deta,
  const float2_t * p_xc, cell_blk_t * p_dev, int * p_n_thr )
{
    blk_assemble_t< geo_T > op;
    op.g       = g;
    op.c       = c;
    op.p_blk   = p_dev;
    op.n_cell  = c.n_cell;
    op.p_n_thr = p_n_thr;
    op.p_eta   = p_eta;
    op.p_deta  = p_deta;
    op.p_xc    = p_xc;
    const auto [ n_bl, n_th, s_sh ] = op.resource( 256 );
    dev.launch( blk_assemble_k< blk_assemble_t< geo_T > >,
                n_bl, n_th, s_sh, nullptr, op, c.n_cell );
    dev.sync_all_streams(  );
    return;
};

////////////////////////////////////////////////////////////
//  PRODUCTION M12:  the sparse product.
//
//  This is the operator `a_from_bf` needs.  It USED to be
//  the gate's ref_rhs / ref_rhs_t.  P4.1 moved those
//  definitions to edge_mg_ref.cpp ( and P4.2 rewrites them
//  in closed form,  with the reference's OWN M12^T ),  so
//  the production path must not call them any more.  The
//  code is the same,  so the implicit step stays
//  bit-identical:  the split is about OWNERSHIP,  not about
//  the numbers.

__host__ void driver_t::prod_rhs_m12
( const mesh::geo_t & geo, const idx_t & n_ceff,
  const bool ( & per )[ 3 ], const cell_blk_t * p_blk,
  const float2_t * p_b, float2_t * p_r ) const
{
    cplx_t c;
    build_complex( c, geo, n_ceff, per );
    for( int e = 0; e < c.n_edge; ++ e )
    {
        p_r[ e ] = float2_t( 0 );
    }
    int k( 0 );
    for( int i2 = 0; i2 < int( n_ceff[ 2 ] ); ++ i2 )
    {
        for( int i1 = 0; i1 < int( n_ceff[ 1 ] ); ++ i1 )
        {
            for( int i0 = 0; i0 < int( n_ceff[ 0 ] ); ++ i0 )
            {
                idx_t i( i0, i1, i2 );
                int ge[ 12 ], gf[ 6 ];
                c.cell_edges( i, ge );
                c.cell_faces( i, gf );
                for( int e = 0; e < 12; ++ e )
                {
                    float2_t t( 0 );
                    for( int f = 0; f < 6; ++ f )
                    {
                        t += float2_t( p_blk[ k ].m12[ e * 6 + f ] )
                           * p_b[ gf[ f ] ];
                    }
                    p_r[ ge[ e ] ] += t;
                }
                ++ k;
            }
        }
    }
    return;
};

//  A7.1 representation IN,  once when no edge state is
//  available:
//
//      a <- M1^{-1} M12 b.
//
//  The face array stores B.n,  whereas M12 consumes a face
//  FLUX.  Both lower and upper owner faces must be filled;
//  the upper physical walls have no lower-cell owner.
//  There is deliberately NO representation OUT here:  the
//  face state is updated by the divergence-free increment.
__host__ bool driver_t::a_all_zero
( const std::vector< float_t > & a )
{
    for( const auto & v : a )
    {
        if( v != 0. )
        {
            return false;
        }
    }
    return true;
};

__host__ float_t driver_t::a_from_bf
( device::base_t & dev, const view_t & v, const cplx_t & c,
  const geo_sph_t & g, const bool ( & per )[ 3 ],
  const float_t & tol )
{
    read_arr( st.bf_h.size(  ), sizeof( float2_t ) );
    dev.cp( st.bf_h.data(  ), v.bf.dat, st.bf_h.size(  ) );
    dat_3d_t< float2_t > bh;
    bh = v.bf;
    bh.dat = st.bf_h.data(  );
    std::vector< float2_t > fb
        ( std::size_t( c.n_face ), float2_t( 0 ) );
    for( int ic = 0; ic < c.n_cell; ++ ic )
    {
        const idx_t i( ic % int( c.n[ 0 ] ),
                      ( ic / int( c.n[ 0 ] ) )
                      % int( c.n[ 1 ] ),
                       ic / int( c.n[ 0 ] * c.n[ 1 ] ) );
        int gf[ 6 ];
        c.cell_faces( i, gf );
        float2_t ar[ 6 ];
        face_areas( g, i, ar );
        for( int a = 0; a < 3; ++ a )
        {
            const int f = gf[ 2 * a ];
            fb[ std::size_t( f ) ]
                = bh.at( i )[ a ] * ar[ 2 * a ];
            idx_t h( i );
            ++ h[ a ];
            const int fh = gf[ 2 * a + 1 ];
            fb[ std::size_t( fh ) ]
                = bh.at( h )[ a ] * ar[ 2 * a + 1 ];
        }
    }
    std::vector< float2_t > r2
        ( std::size_t( c.n_edge ), 0. );
    prod_rhs_m12( v.geo, v.geo.n_ceff, per,
                  st.blk_h.data(  ), fb.data(  ),
                  r2.data(  ) );
    std::vector< float_t > rr
        ( std::size_t( c.n_edge ), 0. );
    for( int e = 0; e < c.n_edge; ++ e )
    {
        rr[ std::size_t( e ) ]
            = float_t( r2[ std::size_t( e ) ] );
    }
    std::vector< float_t > a_red
        ( std::size_t( st.n_compact ), 0. );
    reduce_rhs( c.n_edge, st.keep, st.comp, drop_t::compact,
                rr, a_red );
    std::vector< float_t > xa
        ( std::size_t( st.n_compact ), 0. );
    int it( 0 );
    const float_t res = cg( st.M1, a_red, xa, tol, 4000, it );
    std::vector< float_t > araw;
    expand_sol( c.n_edge, st.comp, drop_t::compact, xa,
                araw );
    dat_3d_t< float_t > ah;
    ah = v.a;
    ah.dat = st.a_h.data(  );
    for( int e = 0; e < c.n_edge; ++ e )
    {
        int   a_( 0 );
        idx_t nd( 0 );
        c.edge_key( e, a_, nd );
        ah.at( nd )[ a_ ] = araw[ std::size_t( e ) ];
    }
    write_arr( st.a_h.size(  ), sizeof( float_t ) );
    dev.cp( v.a.dat, st.a_h.data(  ), st.a_h.size(  ) );
    return res;
}

//////////////////////////////////////////////////////////////
//  The two invariants the matrix-free backend must satisfy
//  before a step is allowed to solve with it.
//
//  ( 1 ) Level by level,  `A_l x` from the aggregated per-cell
//  blocks must be `Sl[l] x` from the host Galerkin product the
//  assembled path solves.  The two are the same operator up to
//  the floating-point summation order ( A4.29 accepted
//  2.4e-16 ),  so anything above that means the aggregation
//  plan and the prolongation disagree -- and then nothing
//  downstream can be trusted ( dead-ends M79 ).  Level 0 is
//  checked too:  there the device applies the per-step cell
//  blocks and the host the assembled M1 + dt K.
//
//  ( 2 ) One V-cycle on a test residual must contract it
//  ( dead-ends M82 ).  The band is 0.5,  not 1.0:  "it
//  contracts at all" is too weak a claim for a preconditioner
//  the outer loop then relies on.  Every configuration
//  measured so far lands between 2.7e-03 ( 8x16x32 ) and
//  2.7e-01 ( polar,  n_smooth = 1 ),  so 0.5 is a tripwire,
//  not a tuning knob ( STATUS.md ( 90 ) ).
//
//  Both THROW,  not print:  a wrong aggregation or a
//  non-contracting cycle only shows up as an iteration count
//  in some gate's output otherwise,  and the small-mesh gates
//  pass either way.
//
//  The NUMBERS a human reads -- the per-level ladder,  the
//  Galerkin image,  the smoother's lambda -- are NOT printed
//  here.  They are a diagnostic,  they cost a device read-back
//  of lam,  and they live in the pgen that wants them
//  ( usr_ext/mhd_ee/ee_audit.cpp,  installed through audit( ) ).
//  This function measures the two scalars,  hands them to that
//  report,  and throws if either is out of band.
__host__ void driver_t::check_hierarchy
( device::base_t & dev, const hier_t & h,
  const std::vector< op_lvl_t > & op, mf_t & mf )
{
    const int n_max = mf.lv[ 0 ].n_compact;
    double worst_agg = 0.;
    double contraction = 0.;
    float_t * px = ( float_t * ) dev.f_malloc
                 ( sizeof( float_t ) * size_t( n_max ) );
    float_t * py = ( float_t * ) dev.f_malloc
                 ( sizeof( float_t ) * size_t( n_max ) );
    for( int l = 0; l < mf.n_lvl; ++ l )
    {
        const int n = mf.lv[ l ].n_compact;
        std::vector< float_t > xh( size_t( n ), 0. );
        for( int i = 0; i < n; ++ i )
        {
            xh[ size_t( i ) ]
                = float_t( std::sin( 0.7321 * double( i + 1 ) ) );
        }
        dev.f_cp( px, xh.data(  ), sizeof( float_t ) * size_t( n ) );
        dev.f_mset( py, 0, sizeof( float_t ) * size_t( n ), nullptr );
        mf.apply( l, px, py );
        std::vector< float_t > yd( size_t( n ), 0. );
        dev.f_cp( yd.data(  ), py, sizeof( float_t ) * size_t( n ) );
        std::vector< float_t > yh( size_t( n ), 0. );
        op[ size_t( l ) ].A.mul( xh, yh );
        double nd( 0. ), nh( 0. );
        for( int i = 0; i < n; ++ i )
        {
            const double d = double( yd[ size_t( i ) ] )
                           - double( yh[ size_t( i ) ] );
            nd += d * d;
            nh += double( yh[ size_t( i ) ] )
                * double( yh[ size_t( i ) ] );
        }
        const double rel = std::sqrt( nd / ( nh > 0. ? nh : 1. ) );
        if( !( rel <= worst_agg ) )   // NaN-aware,  like M76
        {
            worst_agg = rel;
        }
    }
    //  --- the cycle itself:  one V-cycle on a test residual
    //  must contract it.  v2 = A( B r ),  so
    //  | r - v2 | / | r | is the one-cycle contraction.
    {
        const int n = mf.lv[ 0 ].n_compact;
        std::vector< float_t > rh( size_t( n ), 0. );
        std::vector< float_t > ah( size_t( n ), 0. );
        for( int i = 0; i < n; ++ i )
        {
            rh[ size_t( i ) ]
                = float_t( std::cos( 0.3771 * double( i + 1 ) ) );
        }
        dev.f_cp( px, rh.data(  ), sizeof( float_t ) * size_t( n ) );
        const float_t * pz = mf.cycle_apply( 0, px );
        dev.f_mset( py, 0, sizeof( float_t ) * size_t( n ), nullptr );
        mf.apply( 0, pz, py );
        dev.f_cp( ah.data(  ), py, sizeof( float_t ) * size_t( n ) );
        double nd( 0. ), nh( 0. );
        for( int i = 0; i < n; ++ i )
        {
            const double d = double( rh[ size_t( i ) ] )
                           - double( ah[ size_t( i ) ] );
            nd += d * d;
            nh += double( rh[ size_t( i ) ] ) * double( rh[ size_t( i ) ] );
        }
        contraction = std::sqrt( nd / ( nh > 0. ? nh : 1. ) );
    }
    dev.f_free( px );
    dev.f_free( py );
    //  the gate's report,  with the contraction this function
    //  measured ( the report never runs the cycle twice )
    if( p_audit != nullptr )
    {
        p_audit( p_audit_ctx, dev, h, op, mf, contraction );
    }
    if( !( worst_agg <= double( 1.e-10 ) ) )
    {
        std::printf( "CXX mf-agg FAIL :  the worst per-level"
                     " operator mismatch is %.3e\n", worst_agg );
        throw std::runtime_error
            ( "mhd_ee::ni::edge_mg:  mf_t:  the device per-level"
              " operator does not match the assembled one ( see"
              " dead-ends M79 );  refusing to solve with it." );
    }
    if( !( contraction <= double( 0.5 ) ) )
    {
        std::printf( "CXX mf-cyc FAIL :  one V-cycle contracts"
                     " by only %.3e ( the band is 0.5 )\n",
                     contraction );
        throw std::runtime_error
            ( "mhd_ee::ni::edge_mg:  mf_t:  one V-cycle does not"
              " contract ( see dead-ends M82 );  refusing to solve"
              " with it." );
    }
    return;
};

////////////////////////////////////////////////////////////
//  Install the gate's hierarchy report.  The function is the
//  pgen's ( usr_ext/mhd_ee/ee_audit.cpp );  the driver only
//  keeps the pointer and calls it from check_hierarchy( ).

__host__ void driver_t::audit( void * ctx, hier_audit_t fn )
{
    p_audit_ctx = ctx;
    p_audit     = fn;
    return;
};

//////////////////////////////////////////////////////////
//  The meter,  the state's lifetime,  and the mirrors

__host__ void driver_t::read_arr
( const std::size_t & n, const std::size_t & el )
{
    if( tr_on )
    {
        ++ tr_n_read;
        tr_b_read += long( n * el );
    }
    return;
};

__host__ void driver_t::write_arr
( const std::size_t & n, const std::size_t & el )
{
    if( tr_on )
    {
        ++ tr_n_write;
        tr_b_write += long( n * el );
    }
    return;
};

__host__ void driver_t::traffic
( int & n_read, int & n_write, long & b_read,
  long & b_write ) const
{
    n_read  = tr_n_read;
    n_write = tr_n_write;
    b_read  = tr_b_read;
    b_write = tr_b_write;
    return;
};

__host__ bool driver_t::traffic_on(  ) const
{
    return tr_on;
};

__host__ void driver_t::release( device::base_t & dev )
{
    for( auto * p : { & st.p_blk, & st.p_lat, & st.p_gr, & st.p_xc } )
    {
        if( * p != nullptr )
        {
            dev.f_free( * p );
            * p = nullptr;
        }
    }
    if( st.mf_ready )
    {
        st.mf.free(  );
        st.mf_ready = false;
    }
    if( st.tail_ready )
    {
        rep_dev_free( dev, st.rep_dev );
        wb_dev_free( dev, st.wb );
        wb0_dev_free( dev, st.wb0 );
        if( st.p_rep_old != nullptr )
        {
            dev.f_free( st.p_rep_old );
            st.p_rep_old = nullptr;
        }
        if( st.p_rep_new != nullptr )
        {
            dev.f_free( st.p_rep_new );
            st.p_rep_new = nullptr;
        }
        st.tail_ready = false;
    }
    st.built = false;
    return;
};

__host__ driver_t::driver_t(  )
{
    return;
};

__host__ driver_t::~driver_t(  )
{
    //  The solver's finalize( ) frees through free( ) while the
    //  device is alive;  this is the guard for a driver that is
    //  destroyed without it.
    if( p_dev != nullptr )
    {
        release( * p_dev );
    }
    return;
};

__host__ void driver_t::free(  )
{
    if( p_dev != nullptr )
    {
        release( * p_dev );
    }
    return;
};

//  Read-only handles on the step state's host mirrors.  The
//  step never calls this:  test_mode 15 poisons the mirrors,
//  runs one step,  and asserts they are untouched -- the
//  non-fakeable form of the traffic meter.

__host__ void driver_t::mirror
( const int & what, void ** pp, int & n, int & on_dev )
{
    void *  p[ 9 ]
    { st.blk_h.data( ), st.lat_h.data( ), st.eta_h.data( ),
      st.a_h.data( ), st.bf_h.data( ), st.far.data( ),
      nullptr,
      st.mf.lv.empty( ) ? nullptr : st.mf.lv[ 0 ].p_x, nullptr };
    int     m[ 9 ]
    { int( st.blk_h.size( ) ), int( st.lat_h.size( ) ),
      int( st.eta_h.size( ) ), int( st.a_h.size( ) ),
      int( st.bf_h.size( ) ), int( st.far.size( ) ),
      0,
      st.mf.lv.empty( ) ? 0 : st.mf.lv[ 0 ].n_compact, 0 };
    const int k = ( what >= 0 && what <= 7 ) ? what : 8;
    * pp = p[ k ];
    n = m[ k ];
    on_dev = ( k == 7 ) ? 1 : 0;
    return;
};

////////////////////////////////////////////////////////////
//  The seven phases of one step

//  ---- 0. one allocation per shape
__host__ void driver_t::_allocate
( device::base_t & dev, const view_t & v, const cplx_t & c )
{
    const int ne = c.n_edge;
    const int nc = c.n_cell;
    const int nf = c.n_face;
    if( st.built && st.n_edge == ne && st.n_cell == nc )
    {
        return;
    }
    release( dev );
    st.cx     = c;
    st.n_edge = ne;
    st.n_cell = nc;
    st.n_face = nf;
    st.p_blk = dev.f_malloc( sizeof( cell_blk_t ) * std::size_t( nc ) );
    st.p_lat = dev.f_malloc( sizeof( float2_t ) * std::size_t( nc ) * 6 );
    st.p_gr  = dev.f_malloc( sizeof( float2_t ) * std::size_t( nc ) * 18 );
    st.p_xc  = dev.f_malloc( sizeof( float2_t ) * std::size_t( nc ) * 3 );
    st.blk_h.resize( std::size_t( nc ) );
    st.a_h.resize( std::size_t( v.a.n_ctot ) );
    st.bf_h.resize( std::size_t( v.bf.n_ctot ) );
    st.eta_h.resize( std::size_t( v.eta.n_ctot ) );
    st.far.assign( std::size_t( nc ) * 6, 0. );
    st.built = true;
    return;
};

//  ---- 1. the eta lattice,  on the device ( K1 )
__host__ void driver_t::_lattice
( device::base_t & dev, const view_t & v )
{
    ni_ie_eta_lat_dev
        ( dev, v.geo, v.eta, ( float2_t * ) st.p_lat, 256 );
    return;
};

//  ---- 2. the sub-cell gradient
__host__ void driver_t::_gradient
( device::base_t & dev, const geo_sph_t & g, const cplx_t & c,
  float2_t *& p_deta, float2_t *& p_xc )
{
    const int nc = c.n_cell;
    const int t = std::min( 256, nc > 0 ? nc : 1 );
    dev.launch( eta_grad_k,
                dim3( ( nc + t - 1 ) / t ), dim3( t ), 0, nullptr,
                g, c, ( const float2_t * ) st.p_lat,
                ( float2_t * ) st.p_gr );
    dev.sync_all_streams(  );
    ni_ie_eta_xc_dev
        ( dev, g, c.n, ( float2_t * ) st.p_xc, 256 );
    p_deta = ( float2_t * ) st.p_gr;
    p_xc   = ( float2_t * ) st.p_xc;
    return;
};

//  ---- 3. the blocks,  assembled on the device
//
//  A matrix-free step ( ee_method = 5 ) with the device
//  write-back consumes them IN PLACE:  level 0 reads them as
//  they are and every coarse level aggregates them ( K2 was
//  the last per-cell host mirror in a step ).  The read-back
//  below therefore happens ONCE,  when the device hierarchy
//  is extracted from the assembled one -- 13.11 MB of PREC2
//  blocks that a step no longer pays.
//
//  The matrix-free step's DEVICE TAIL is defined for both
//  write-back modes:  mode 1 forms bf from P_out( a_new ) -
//  P_out( a_old ) through the K6 bridge,  mode 0 ( the
//  shipped default ) forms the strong -dt d1( eta J )
//  increment from the edge state and the eta lattice.  Either
//  way the step copies no array.
__host__ void driver_t::_blocks
( device::base_t & dev, const geo_sph_t & g, const cplx_t & c,
  const float2_t * p_deta, const float2_t * p_xc )
{
    cells_dev_eta< geo_sph_t >
        ( dev, g, c, ( const float2_t * ) st.p_lat, p_deta, p_xc,
          ( cell_blk_t * ) st.p_blk, nullptr );
    //  The ONE read-back: the build step needs the host blocks
    //  for the reference operator of the G-AGG check ( and for
    //  nothing else ).  Every later step skips this.
    if( ! st.mf_ready )
    {
        read_arr( std::size_t( c.n_cell ), sizeof( cell_blk_t ) );
        dev.cp( st.blk_h.data(  ), ( const cell_blk_t * ) st.p_blk,
                std::size_t( c.n_cell ) );
    }
    return;
};

//  ---- 4a. the operator:  ONE matrix-free hierarchy,  built
//  ONCE.
//
//  That build is the only place the prolongation operators
//  and the 2:1 cell hierarchy are defined,  so reading the
//  TOPOLOGY off it is cheaper and much safer than deriving it
//  a second time ( M74 was a weight bug ).  What it also
//  leaves behind is the numbering ( keep / comp ) and the
//  level-0 M1,  which is all the matrix-free backend needs to
//  build its own device hierarchy.  After this block the
//  assembled operator exists only for setup-time jobs:  the
//  representation IN ( 4b ),  and the G-AGG invariant.
__host__ void driver_t::_build
( device::base_t & dev, view_t & v, const par_t & par,
  const geo_sph_t & g, const cplx_t & c,
  const bool ( & per )[ 3 ], const float_t & dt )
{
    const idx_t n = v.geo.n_ceff;
    const int cdir[ 3 ] = { ( par.ee_coarsen & 1 ) ? 1 : 0,
                            ( par.ee_coarsen & 2 ) ? 1 : 0,
                            ( par.ee_coarsen & 4 ) ? 1 : 0 };
    hier_t h;
    h.site_major = 1;
    h.bc_edge    = par.bc_edge;
    h.build( n, per, cdir, par.ee_nlevel,
             g.pole_lo, g.pole_hi );
    st.mf.n_smooth    = par.ee_n_smooth;
    st.mf.coarse_iter = par.ee_coarse_iter;
    st.mf.coarse_tol  = par.ee_coarse_tol;
    st.mf.build( dev, h, 256 );
    //  Level 0's apply needs the cell blocks,  and the coarse
    //  levels their aggregation:  one refill before the check,
    //  so the check sees exactly what a step sees.
    st.mf.refill( ( const cell_blk_t * ) st.p_blk, dt );
    st.keep      = h.L[ 0 ].keep;
    st.comp      = h.L[ 0 ].comp;
    st.n_compact = h.L[ 0 ].n_compact;
    std::vector< op_lvl_t > op;
    op_build( h, st.blk_h.data(  ), dt, op );
    st.M1 = op[ 0 ].M1;
    check_hierarchy( dev, h, op, st.mf );
    //  The representation IN ( M1 a = M12 bf ) and the DEVICE
    //  TAIL.  Both need only the setup-time numbers.
    _bridge_in( dev, v, par, g, c, per );
    _tail_setup( dev, v, par, g );
    st.tail_ready = true;
    st.mf_ready   = true;
    return;
};

//  ---- 4b. the representation IN ( M1 a = M12 bf ).  A
//  restart from bf arrives with a zero edge state, so this is
//  the one place that has to look at the host mirror of `a`
//  -- and it must run BEFORE the tail seeds rep_old from the
//  incoming state,  otherwise the first step's face increment
//  would count the bridged state twice ( dead-ends M83 ).
__host__ void driver_t::_bridge_in
( device::base_t & dev, const view_t & v, const par_t & par,
  const geo_sph_t & g, const cplx_t & c,
  const bool ( & per )[ 3 ] )
{
    if( par.ee_a_bridge == 0 || st.bridge_done )
    {
        return;
    }
    read_arr( st.a_h.size(  ), sizeof( float_t ) );
    dev.cp( st.a_h.data(  ), v.a.dat, st.a_h.size(  ) );
    if( a_all_zero( st.a_h ) )
    {
        const float_t bridge_res
            = a_from_bf( dev, v, c, g, per,
                         float_t( par.ee_tol ) );
        if( bridge_res > 10 * par.ee_tol )
        {
            throw std::runtime_error
            ( "edge_mg:  bf -> a representation-in did not"
              " converge" );
        }
    }
    st.bridge_done = true;
    return;
};

//  ---- 4c. the DEVICE TAIL.  The face write-back needs a
//  per-face table ( owner cell,  face-local side, owner area,
//  degenerate mask ),  all geometry:  K7 measured the
//  per-step _face_areas at ~5 ms a block,  so it is built
//  here,  once.  rep_old starts at the INCOMING edge state,
//  so the first step's increment is measured from it exactly
//  like the host path did.
__host__ void driver_t::_tail_setup
( device::base_t & dev, view_t & v, const par_t & par,
  const geo_sph_t & g )
{
    const idx_t n = v.geo.n_ceff;
    const cplx_t & c = st.cx;
    const int nc = c.n_cell;
    rep_dev_ensure( dev, c, st.rep_dev, 256 );
    for( int ic = 0; ic < nc; ++ ic )
    {
        const idx_t i( ic % int( n[ 0 ] ),
                      ( ic / int( n[ 0 ] ) ) % int( n[ 1 ] ),
                       ic / int( n[ 0 ] * n[ 1 ] ) );
        float2_t ar[ 6 ];
        face_areas( g, i, ar );
        for( int s_ = 0; s_ < 6; ++ s_ )
        {
            st.far[ std::size_t( ic ) * 6 + s_ ]
                = float_t( ar[ s_ ] );
        }
    }
    wb_dev_setup( st.wb, dev, v.geo, c, st.far.data(  ), v.bf );
    //  Mode 0's own table:  the cell -> edge CSR ( so the
    //  accumulation order is the host's ),  the cached d1 face
    //  rows and the wall-edge mask.
    wb0_dev_setup( st.wb0, dev, v.geo, c, st.far.data(  ),
                   v.a, v.bf );
    if( par.ee_writeback == 1 )
    {
        st.p_rep_old = ( float_t * ) dev.f_malloc
                     ( sizeof( float_t ) * size_t( c.n_face ) );
        st.p_rep_new = ( float_t * ) dev.f_malloc
                     ( sizeof( float_t ) * size_t( c.n_face ) );
        //  rep_old is built from the INCOMING edge state.  The
        //  bridge is indexed by GLOBAL edge id,  so the state is
        //  first gathered into a compact vector and expanded
        //  back into the raw per-edge array -- a device round
        //  trip,  once,  not per step.
        st.mf.gather( v.a, st.mf.lv[ 0 ].p_z );
        st.mf.expand( v.a, st.mf.lv[ 0 ].p_z, st.mf.p_xraw0 );
        int     it0( 0 );
        float_t res0( 0. );
        ni_ie_represent_dev_raw
            ( dev, c, ( const cell_blk_t * ) st.p_blk,
              st.mf.p_xraw0, st.rep_dev, st.p_rep_old,
              float_t( par.ee_tol ), par.ee_iter, it0, res0,
              256 );
    }
    return;
};

//  ---- 5. the solve.
//
//  The cell blocks are already on the device ( phase 3 ),  the
//  right-hand side is M1 applied to the gathered edge state,
//  and the coarse side of the V-cycle is refreshed from the
//  blocks -- so a step here copies NO per-cell or per-face
//  array,  only the O(1) scalars of the PCG.
__host__ void driver_t::_solve
( device::base_t & dev, const view_t & v, const par_t & par,
  const float_t & dt, solve_stat_t & out )
{
    ( void ) dev;
    st.mf.refill( ( const cell_blk_t * ) st.p_blk, dt );
    st.mf.gather( v.a, st.mf.lv[ 0 ].p_z );
    st.mf.rhs_m1( st.mf.lv[ 0 ].p_z, st.mf.lv[ 0 ].p_b );
    const auto ps = st.mf.solve
        ( st.mf.lv[ 0 ].p_x, st.mf.lv[ 0 ].p_b,
          float_t( par.ee_tol ), par.ee_iter, true );
    out.it     = ps.it;
    out.n_red  = ps.n_red;
    out.n_comm = ps.n_red;
    out.res    = ps.res;
    return;
};

//  ---- 6. the device tail:  expand the compact solution onto
//  the edge field,  then the representation bridge ( K6 ) and
//  the face write-back ( K7 ) run on the arrays the step
//  already owns.  Nothing here returns to the host but the two
//  O(1) counters of the M2 solve.
__host__ void driver_t::_tail
( device::base_t & dev, view_t & v, const par_t & par,
  const float_t & dt, solve_stat_t & out )
{
    int     r_it( 0 );
    float_t r_res( 0. );
    if( par.ee_writeback == 1 )
    {
        //  Mode 1:  bf += P_out( a_new ) - P_out( a_old ).
        st.mf.expand( v.a, st.mf.lv[ 0 ].p_x, st.mf.p_xraw0 );
        //  Warm start from the previous step's rep:  2-3 fewer
        //  M2 iterations,  and the scalar readback count does
        //  not change ( it is still one 8-float copy per
        //  iteration ).
        ni_ie_represent_dev_raw
            ( dev, st.cx, ( const cell_blk_t * ) st.p_blk,
              st.mf.p_xraw0, st.rep_dev, st.p_rep_new,
              float_t( par.ee_tol ), par.ee_iter, r_it, r_res,
              256, st.p_rep_old );
        if( std::getenv( "MF_TAIL_TRACE" ) != nullptr )
        {
            std::printf( "CXX k6-tail:  r_it %d  r_res %.3e\n",
                         r_it, double( r_res ) );
            std::fflush( stdout );
        }
        ni_ie_writeback_dev
            ( dev, st.wb, v.bf, st.p_rep_old, st.p_rep_new, 256 );
        std::swap( st.p_rep_old, st.p_rep_new );
    }
    else
    {
        //  Mode 0,  the shipped default:  the strong
        //  -dt d1( eta J ) increment,  formed on the device
        //  from the edge state and the eta lattice.
        st.mf.expand( v.a, st.mf.lv[ 0 ].p_x, nullptr );
        ni_ie_writeback0_dev
            ( dev, st.wb, st.wb0, v.bf, v.a,
              ( const float2_t * ) st.p_lat, dt, par.bc_edge, 256 );
    }
    out.it    = out.it + r_it;
    out.res   = std::max( out.res, r_res );
    return;
};

////////////////////////////////////////////////////////////
//  One implicit step of one block.

__host__ void driver_t::step
( device::base_t & dev, const void * stream, view_t & v,
  const par_t & par, solve_stat_t & out )
{
    ( void ) stream;
    p_dev = & dev;
    tr_on = true;
    tr_n_read  = 0;
    tr_n_write = 0;
    tr_b_read  = 0;
    tr_b_write = 0;
    const mesh::geo_t & geo = v.geo;
    const idx_t n = geo.n_ceff;
    bool per[ 3 ];
    for( int a = 0; a < 3; ++ a )
    {
        per[ a ] = !( geo.phybnd_flag[ a ][ 0 ]
                   || geo.phybnd_flag[ a ][ 1 ] );
    }
    geo_sph_t g;
    if( ! geo_sph_from( geo, g ) )
    {
        throw std::runtime_error
        ( "edge_mg:  ee_solve needs a uniform ( r, theta ) spherical block"
          " ( the cartesian policy has no mu parameterisation yet )." );
    }
    cplx_t c;
    build_complex( c, geo, n, per );
    const float_t dt = float_t( v.dt );
    _allocate( dev, v, c );
    _lattice( dev, v );
    float2_t * p_deta( nullptr );
    float2_t * p_xc( nullptr );
    if( par.subcell )
    {
        _gradient( dev, g, c, p_deta, p_xc );
    }
    _blocks( dev, g, c, p_deta, p_xc );
    if( ! st.mf_ready )
    {
        _build( dev, v, par, g, c, per, dt );
    }
    _solve( dev, v, par, dt, out );
    _tail( dev, v, par, dt, out );
    if( par.ee_verb > 0 )
    {
        std::printf
            ( "EE1 m=5 it=%d n_red=%d n_comm=%d res=%.3e"
              "  ( ne %d,  nc %d )\n",
              std::min( out.it, par.ee_iter ), out.n_red,
              out.n_comm, out.res, st.n_edge, st.n_cell );
    }
    tr_on = false;
    return;
};

};    // namespace mhd_ee::ni::edge_mg