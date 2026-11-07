#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../../../../../../src/device/device.h"
#include "../../../../../../src/mesh/geometry/geometry.h"
#include "../../../../../../src/mesh/types/dat_3d.h"
#include "../../../../../../src/types.h"

#include "edge_mg_cplx.h"

//  the __global__ layer of this file lives in the kernels/ subdirectory here.
#include "kernels/ni_ie_kernels.h"

////////////////////////////////////////////////////////////
//  P5 / K7 -- the DEVICE write-back:  edge solution -> face
//  field,  one thread per face.
//
//  solve_step's write-back ( edge_mg.cpp,  the loop after
//  the solve ) does two things every step:
//
//    (1) it REBUILDS the per-cell face areas,
//        `_face_areas( g, i, ar )` for every cell,  into
//        `st.far[ ic*6 + s ]`.  The areas are a property
//        of the BLOCK GEOMETRY alone:  eta,  the solution,
//        dt,  ee_subcell and the iterative refinement never
//        enter them.  Recomputing them each step is O(nc)
//        of 3x3 Gauss quadrature -- pure waste.
//    (2) it walks the faces and updates `bf`:
//
//          ar = far[ ic*6 + 2 a + s ]        ( owner cell )
//          if( ar > 0 && !degenerate )
//              flux = bf[ nd ][ a ] * ar + rep_new[ f ]
//                                          - rep_old[ f ]
//              bf[ nd ][ a ] = flux / ar
//
//        which is pure elementwise arithmetic on one face's
//        own three numbers.
//
//  This file owns both:  `wb_dev_setup` builds the STATIC
//  per-face table once ( owner cell,  side s,  axis a,
//  owner area,  bf slot,  degenerate flag ),  and
//  `ni_ie_writeback_dev` is one kernel launch,  one thread
//  per face,  NO ATOMICS.
//
//  NO ATOMICS is a theorem,  not a hope:  a face is
//  ( axis,  lower node ) and `cplx_t::face_key` is a
//  bijection onto the faces,  so every face owns exactly
//  one ( node,  component ) slot,  and two faces never
//  write the same `float2_t` component.  `wb_build` CHECKS
//  the injectivity of the slot map anyway ( it is O(1)
//  work per face plus one sort at setup time ),  because a
//  silent collision would be a data race that no gate could
//  see -- the last writer wins and the numbers look sane.
//
//  WHAT THE TABLE IS FOR ( and what it frees ):
//
//  * the per-step `st.far` fill disappears -- call
//    `wb_dev_fill_far` once per block,  from the per-face
//    areas `ref_face_area` already provides,  and the
//    per-cell 6-area array is no longer needed at all;
//  * the per-step HOST round trip of `bf` disappears --
//    the update runs where `bf` lives;
//  * the owner selection ( M66:  the canonical owner must
//    be the one the ASSEMBLY used ) is computed once,  by
//    the same `face_key` decomposition the assembler uses,
//    so it cannot drift from it.
//
//  WHEN THE TABLE MUST BE REBUILT ( the price of caching
//  the geometry -- see the K7 report for the full
//  argument ):
//
//  * the block's shape ( `c` ) changed -- regrid,  level
//    change,  refinement / coarsening of this block;
//  * the block's geometry policy changed -- a moved block,
//    a different `x_min/x_max`,  a different `sph` mode,
//    an SMR regrid that re-creates the block's geo;
//  * the `bf` array was re-created ( its `dat` / `n_gh` /
//    `n_cell` are baked into `slot` ).
//  It must NOT be rebuilt per step:  the mesh does not
//  move between steps of one solve,  and that is exactly
//  the O(nc) that this file removes.
//
//  TRAPS THIS FILE IS BUILT AROUND:
//
//  * M63 -- a probe that reads the WRONG array does not
//    crash,  it prints zero error.  Hence the gate checks
//    the frozen slots ( ghosts,  degenerate faces,  the
//    untouched components ) BIT FOR BIT against the
//    uploaded field,  not just the updated ones;
//  * M67 -- a degenerate pole face is ELIMINATED,  not
//    zeroed:  the write must be SKIPPED ( leaving `bf`
//    alone ),  never `flux / 0`.  `deg` is therefore
//    checked before `ar`,  and a gate must run with a real
//    pole ( the a5 wedge has none at theta = 0 );
//  * M66 -- a periodic axis has no high wall:  `nd[ a ]`
//    wraps,  so no face of a periodic axis is owned from
//    `s = 1`.  `wb_build` counts such faces and refuses to
//    build if the count is not zero ( the seam would then
//    be owned from the wrong side ).
////////////////////////////////////////////////////////////

namespace mhd_ee::ni::edge_mg
{

using type::    idx_t;
using type::  float_t;
using type:: float2_t;
using mesh::dat_3d_t;

////////////////////////////////////////////////////////////
//  One face,  flattened for the device.
//
//  `slot` is the ABSOLUTE offset into `bf.dat`,  in units
//  of one `float2_t` -- i.e. `bf.idx( nd ) + a`.  It is
//  the whole reason the kernel needs no `idx_t` arithmetic,
//  no ghost offsets and no `n_int` multiply:  everything
//  index-shaped is resolved once,  on the host.
//
//  There is NO sign / orientation field,  and that is not an
//  omission:  `bf.at( nd )[ a ]` holds `B_a` AT THE NODE
//  ( the host code stores `flux / ar`,  not `+- flux / ar` ),
//  so the face's own normal `a` fully determines the write.
//  The host loop has no sign bookkeeping either -- if one is
//  ever added it must be added here,  once,  next to `a`.

////////////////////////////////////////////////////////////
//  The static per-face table plus the counters a caller
//  ( or a gate ) wants printed.
////////////////////////////////////////////////////////////

struct wb_dev_t
{
    //  shape / the lattice the table was built for
    idx_t          n;   // cells along the ( block's ) axes
    int       n_face;   // faces in the table
    int       n_cell;   // owner cells ( diagnostics )
    int        n_deg;   // degenerate faces ( pole theta rings )
    int        n_ar0;   // faces with a non-positive owner area
    int n_written;      // faces the kernel actually updates
    int        n_dsup;  // distinct ( node, component ) slots
    //  the device table
    wb_face_t * p_face;
    //  the host copy ( diagnostics,  and the sabotage runs )
    std::vector< wb_face_t > face_h;

    __host__ wb_dev_t(  );
    __host__ bool ok(  ) const;
};

////////////////////////////////////////////////////////////
//  far[ ic*6 + 2a + s ] from the per-face areas.
//
//  `p_a[ f ]` is the area of face `f`,  in the canonical
//  decomposition `c.face_key( f, a, nd )` -- exactly what
//  `ref_face_area` fills.  The scatter takes the OWNER
//  cell of each face,  so the result is,  entry by entry,
//  the owner entry of the old `_face_areas( g, i, ar )`
//  array,  and the per-step loop that built it can go away.
//
//  Slots that no face owns are LEFT ALONE ( a periodic axis
//  has no high wall ),  so the caller should zero the
//  buffer once before the call -- both producers here do.
//  Returns the number of faces scattered.
////////////////////////////////////////////////////////////

__host__ int wb_dev_fill_far
( const cplx_t & c, const float2_t * p_a, float_t * far );

////////////////////////////////////////////////////////////
//  Build the host table.  No device traffic -- `wb_upload`
//  is separate so that a gate ( or a sabotage run ) can
//  inspect / perturb the table before it is sent.
//
//  `far_host` must be the per-cell 6-area array for THIS
//  block's lattice,  i.e. `wb_dev_fill_far`'s output ( or
//  the old `st.far` ).  `bf` supplies only `idx / n_gh /
//  n_cell / n_int`:  the table is tied to ONE array.
////////////////////////////////////////////////////////////

__host__ void wb_build
( wb_dev_t & w, const mesh::geo_t & geo, const cplx_t & c,
  const float_t * far_host, const dat_3d_t< float2_t > & bf );

////////////////////////////////////////////////////////////
//  Allocate / upload the table.  Idempotent:  a second call
//  frees the first table ( the sabotage runs rely on it ).
////////////////////////////////////////////////////////////

__host__ void wb_upload
( wb_dev_t & w, device::base_t & dev );

////////////////////////////////////////////////////////////
//  setup = build + upload ( one call for production ).
////////////////////////////////////////////////////////////

__host__ void wb_dev_setup
( wb_dev_t & w, device::base_t & dev, const mesh::geo_t & geo,
  const cplx_t & c, const float_t * far_host,
  const dat_3d_t< float2_t > & bf );

__host__ void wb_dev_free
( device::base_t & dev, wb_dev_t & w );

////////////////////////////////////////////////////////////
//  THE KERNEL -- one thread per face,  no atomics.
//
//  The update is elementwise in the face's OWN three
//  numbers,  so the result is bit-identical to the host
//  loop at every PRECISION:  the same expression in the
//  same order,  `float_t( bf )` downcast included ( at
//  PRECISION = 1 the host also downcasts the stored
//  float2_t to float_t before the arithmetic ).
//
//  A face that is degenerate ( M67 ) or has no positive
//  area is SKIPPED -- `bf` keeps its value bit for bit.
//  The `deg` test comes FIRST so that a stale / negative
//  area on an eliminated face can never divide.
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//  The launcher.
//
//      ni_ie_writeback_dev( dev, w, bf, rep_old, rep_new,
//                           n_th_lim );
//
//  `bf` is updated IN PLACE on the device;  `rep_old` and
//  `rep_new` are the per-face edge-solution arrays ( what
//  the host loop calls `st.rep` / `rep_new` ).  With
//  `ee_writeback == 1` the caller then swaps them,  exactly
//  as before -- this function never touches `rep`.
//
//  `n_th_lim` caps the block size ( 1 .. 1024 ),  so a
//  par-file knob can compare thread counts without a
//  rebuild.  A full barrier follows,  so the caller sees
//  the updated `bf` when this returns ( the write-back is
//  the end of the step's device work;  the alternative is
//  for every caller to remember the sync ).
////////////////////////////////////////////////////////////

__host__ void ni_ie_writeback_dev
( device::base_t & dev, const wb_dev_t & w,
  dat_3d_t< float2_t > & bf, const float_t * p_rep_old,
  const float_t * p_rep_new, const int & n_th_lim );

////////////////////////////////////////////////////////////
//  MODE 0 -- the STRONG write-back,  i.e. the SHIPPED
//  production default `ee_writeback = 0`:
//
//      inc[ f ] = -dt * sum_u sg[ u ] * ee[ ed[ u ] ]
//      ee[ e ]  = ( sum over the cells holding e of
//                   ( eta J ) . dl_e ) / cnt[ e ]
//      bf[ slot ] += inc[ f ] / ar_owner
//
//  with J = curl B / A per cell ( `j_from_curl` ) and the
//  Voigt tensor from the per-cell `p_lat` lattice.
//
//  THREE kernels,  THREE static tables,  NO atomics,  and
//  NO per-step host traffic ( dt and bc_edge are scalars ):
//
//    wb0_cell_k   pass 1.  Per cell: the six local face
//                 circulations from the cached d1 rows,
//                 the six cached areas,  `j_from_curl`
//                 ( the SAME function the host uses ),  the
//                 Voigt multiply.  One `float_t` triple a
//                 cell.
//    wb0_edge_k   pass 2.  Per EDGE,  gather the cells that
//                 hold it,  dot with the cached edge
//                 vector,  divide by the count ( the
//                 scatter normality trap that bit K4 /
//                 A4.31 ),  apply the `bc_edge == 1` wall
//                 mask.
//    wb0_face_k   pass 3.  Per face: the cached d1 row
//                 ( 4 edge indices + signs + nz ),  the
//                 owner area and the bf slot from the SAME
//                 `wb_face_t` table the mode-1 kernel uses.
//
//  WHY A GATHER AND NOT `atomicAdd`.  The host accumulates
//  `acc[ e ]` in a FIXED order -- cells ascending -- and the
//  gate's threshold is 1e-15.  An atomic scatter sums in
//  hardware order,  so the same input gives a different last
//  bit on every run: it would not be comparable to the host
//  bit for bit,  and `cnt` would need a second atomic.  The
//  CSR table is built once,  in ascending cell order,  so the
//  device sums in the host's order and the result is
//  reproducible AND equal.
//
//  WHY THE TENSOR IS THE RAW LATTICE.  The host `writeback`
//  takes `p_eta6` ( six numbers a cell,  from
//  `_eta_lat_host` ) and never looks at `p_deta` / `p_xc`.
//  `ee_subcell` shapes the OPERATOR's tensor,  not the
//  write-back's,  so this path reads `p_lat` and nothing
//  else:  it mirrors the reference instead of quietly
//  "fixing" it.  ( If the reference ever learns the sub-cell
//  tensor,  `wb0_cell_k` needs one extra per-cell
//  expansion -- the table already carries the cell index. )
//
//  THE DEGENERATE MASK.  A pole theta face has area EXACTLY
//  zero,  and `j_from_curl` tests `area > 0` on its own,  and
//  pass 3 tests `ar > 0` as well.  So on the meshes this line
//  runs,  dropping the degenerate mask alone is INVISIBLE --
//  the gate reports that as a finding,  not as a pass.  The
//  mask is still carried ( one flag a face,  the host has it
//  too ) because it is the first guard when `far` is stale or
//  negative,  and because the pass-3 skip is what makes the
//  frozen-slot check meaningful.
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//  The mode-0 static tables plus its device scratch.
//
//  Everything index-shaped is resolved once,  on the host:
//  the cell's six global faces,  the six per-cell areas (
//  the old `st.far`,  computed once instead of every step ),
//  the four ( edge, sign ) pairs of every face,  the CSR
//  cell list of every edge,  the edge vectors,  the
//  wall-tangent mask,  and the `a`-array slot of every edge.
////////////////////////////////////////////////////////////

struct wb0_dev_t
{
    //  shape
    idx_t          n;
    int       n_cell;   // cells
    int       n_face;   // faces
    int       n_edge;   // edges
    int        n_ctr;   // CSR entries ( one per ( cell, edge ) )
    int        n_deg;   // degenerate faces in the tables
    int       n_wall;   // wall-tangent edges ( bc_edge == 1 )
    int       n_cnt0;   // edges no cell holds ( ee = 0 )
    int     a_n_ctot;   // the `a` array the slots were built for
    int    bf_n_ctot;   // the `bf` array the slots were built for
    //  host mirrors ( diagnostics,  and the sabotage runs )
    std::vector< int          >     cf;   // nc*6  faces
    std::vector< float_t      >    ar6;   // nc*6  areas
    std::vector< int          >    fed;   // nf*4  edges
    std::vector< signed char  >    fsg;   // nf*4  signs
    std::vector< signed char  >    fnz;   // nf    count
    std::vector< signed char  >    fdg;   // nf    degenerate
    std::vector< int          >   eptr;   // ne+1  CSR
    std::vector< int          >  ecell;   // n_ctr cells
    std::vector< float_t      >    edl;   // ne*3  edge vector
    std::vector< signed char  >  ewall;   // ne    bc mask
    std::vector< int          >  aslot;   // ne    a slot
    //  the device tables
    int         * p_cf;
    float_t     * p_ar6;
    int         * p_fed;
    signed char * p_fsg;
    signed char * p_fnz;
    signed char * p_fdg;
    int         * p_eptr;
    int         * p_ecell;
    float_t     * p_edl;
    signed char * p_ewall;
    int         * p_aslot;
    //  per-step scratch ( allocated once,  reused )
    float_t     * p_ej;
    float_t     * p_ee;

    __host__ wb0_dev_t(  );
    __host__ bool ok(  ) const;
};

////////////////////////////////////////////////////////////
//  The flat cell index,  the same layout `st.far` uses:
//      ic = ( i2*n1 + i1 ) * n0 + i0
////////////////////////////////////////////////////////////

__host__ idx_t wb0_idx( const cplx_t & c, const int & ic );

////////////////////////////////////////////////////////////
//  Build the host tables.  `far_host` is the per-cell array
//  the old step loop rebuilt every step:
//
//      _face_areas( g, i, ar );  far[ ic*6 + s ] = ar[ s ];
//
//  i.e. the OWN parametrisation of the cell -- NOT the owner
//  side of its high faces.  That distinction is real: the
//  spherical policy maps `r = r0 + ( i + xi )*dr` and
//  `mu = ma + xi*( mb - ma )`,  so a cell's `s = 1` area and
//  the neighbour's `s = 0` area of the same face differ in
//  the last bits.  Pass 1 needs the per-cell array,  the bf
//  update needs the owner entry of the same array.
//
//  Degenerate faces are zeroed here,  exactly where the host
//  zeroes them ( before `j_from_curl` ).
////////////////////////////////////////////////////////////

template< class geo_T >
__host__ void wb0_build
( wb0_dev_t & w, const geo_T & g, const cplx_t & c,
  const float_t * far_host, const dat_3d_t< float_t > & a,
  const dat_3d_t< float2_t > & bf )
{
    if( far_host == nullptr )
    {
        throw std::runtime_error
            ( "wb0_build:  far_host is null." );
    }
    if( bf.n_int != 3 )
    {
        throw std::runtime_error
            ( "wb0_build:  bf must hold three components a node." );
    }
    //  the edge unknowns live in the THREE component
    //  slots of a node (`a.at( nd )[ a_ ]`,  the production
    //  `a_edge` is initialised with n_int = 3 ),  so the
    //  edge slot below is `a.idx( nd ) + a_` -- and `idx`
    //  already carries the component stride
    if( a.n_int != 3 )
    {
        throw std::runtime_error
            ( "wb0_build:  the edge array must hold three"
              " components a node." );
    }
    const int nc = int( c.n_cell );
    const int nf = int( c.n_face );
    const int ne = int( c.n_edge );
    if( nc <= 0 || nf <= 0 || ne <= 0 )
    {
        throw std::runtime_error
            ( "wb0_build:  an empty complex." );
    }
    w.n         = c.n;
    w.n_cell    = nc;
    w.n_face    = nf;
    w.n_edge    = ne;
    w.a_n_ctot  = int( a.n_ctot );
    w.bf_n_ctot = int( bf.n_ctot );
    w.n_deg     = 0;
    //  ---- the per-cell face list and the per-cell areas
    w.cf.assign( std::size_t( nc ) * 6, 0 );
    w.ar6.assign( std::size_t( nc ) * 6, float_t( 0 ) );
    int n_forced( 0 );
    for( int ic = 0; ic < nc; ++ ic )
    {
        const idx_t i( wb0_idx( c, ic ) );
        int gf[ 6 ];
        c.cell_faces( i, gf );
        for( int s = 0; s < 6; ++ s )
        {
            w.cf[ std::size_t( ic ) * 6 + s ] = gf[ s ];
            float_t ar = far_host[ std::size_t( ic ) * 6 + s ];
            if( c.degenerate_f( gf[ s ] ) )
            {
                if( ar != float_t( 0 ) )
                {
                    ++ n_forced;
                }
                ar = float_t( 0 );
            }
            w.ar6[ std::size_t( ic ) * 6 + s ] = ar;
        }
    }
    //  ---- the per-face d1 rows ( topology:  cache it,  the
    //  host `d1_row` is not device code )
    w.fed.assign( std::size_t( nf ) * 4, - 1 );
    w.fsg.assign( std::size_t( nf ) * 4, 0 );
    w.fnz.assign( std::size_t( nf ), 0 );
    w.fdg.assign( std::size_t( nf ), 0 );
    for( int f = 0; f < nf; ++ f )
    {
        if( c.degenerate_f( f ) )
        {
            w.fdg[ std::size_t( f ) ] = 1;
            ++ w.n_deg;
        }
        int ed[ 4 ], sg[ 4 ], nz( 0 );
        c.d1_row( f, ed, sg, nz );
        if( nz < 1 || nz > 4 )
        {
            throw std::runtime_error
                ( "wb0_build:  a d1 row must have 1..4 entries." );
        }
        w.fnz[ std::size_t( f ) ] = ( signed char ) nz;
        for( int u = 0; u < nz; ++ u )
        {
            if( ed[ u ] < 0 || ed[ u ] >= ne )
            {
                throw std::runtime_error
                    ( "wb0_build:  a d1 row points outside"
                      " the edge array." );
            }
            w.fed[ std::size_t( f ) * 4 + u ] = ed[ u ];
            w.fsg[ std::size_t( f ) * 4 + u ]
                = ( signed char ) ( sg[ u ] >= 0 ? 1 : - 1 );
        }
    }
    //  ---- the CSR cell list of every edge,  ascending in
    //  the cell index ( the host's accumulation order )
    std::vector< int > cnt( std::size_t( ne ) + 1, 0 );
    for( int ic = 0; ic < nc; ++ ic )
    {
        int ge[ 12 ];
        c.cell_edges( wb0_idx( c, ic ), ge );
        for( int q = 0; q < 12; ++ q )
        {
            if( ge[ q ] < 0 || ge[ q ] >= ne )
            {
                throw std::runtime_error
                    ( "wb0_build:  a cell edge is outside"
                      " the edge array." );
            }
            ++ cnt[ std::size_t( ge[ q ] ) ];
        }
    }
    w.eptr.assign( std::size_t( ne ) + 1, 0 );
    for( int e = 0; e < ne; ++ e )
    {
        w.eptr[ std::size_t( e ) + 1 ]
            = w.eptr[ std::size_t( e ) ] + cnt[ std::size_t( e ) ];
    }
    w.n_ctr = w.eptr[ std::size_t( ne ) ];
    w.ecell.assign( std::size_t( w.n_ctr ), - 1 );
    std::vector< int > cur( w.eptr.begin(  ),
                            w.eptr.end(  ) - 1 );
    w.n_cnt0 = 0;
    for( int e = 0; e < ne; ++ e )
    {
        if( cnt[ std::size_t( e ) ] == 0 )
        {
            ++ w.n_cnt0;
        }
    }
    for( int ic = 0; ic < nc; ++ ic )
    {
        int ge[ 12 ];
        c.cell_edges( wb0_idx( c, ic ), ge );
        for( int q = 0; q < 12; ++ q )
        {
            const int e = ge[ q ];
            w.ecell[ std::size_t( cur[ std::size_t( e ) ] ) ] = ic;
            ++ cur[ std::size_t( e ) ];
        }
    }
    //  ---- the edge vector and the a-array slot,  per edge
    w.edl.assign( std::size_t( ne ) * 3, float_t( 0 ) );
    w.aslot.assign( std::size_t( ne ), - 1 );
    w.ewall.assign( std::size_t( ne ), 0 );
    w.n_wall = 0;
    for( int e = 0; e < ne; ++ e )
    {
        int    a_( 0 );
        idx_t  nd( 0 );
        c.edge_key( e, a_, nd );
        //  the edge vector:  node_x( nd + e_a ) - node_x( nd ),
        //  the same expression `_dl` uses
        idx_t hd( nd );
        hd[ a_ ] += 1;
        float2_t p0[ 3 ], p1[ 3 ];
        g.node_x( nd, p0 );
        g.node_x( hd, p1 );
        for( int q = 0; q < 3; ++ q )
        {
            w.edl[ std::size_t( e ) * 3 + q ]
                = float_t( p1[ q ] - p0[ q ] );
        }
        //  the slot of the edge's unknown in the `a` array:
        //  production scatters x_raw[ e ] into
        //  a.at( nd )[ a_ ],  so read it back from there
        const int j = int( a.idx( nd ) );
        if( j < 0 || j + a_ >= int( a.n_ctot ) )
        {
            throw std::runtime_error
                ( "wb0_build:  an edge slot is outside the"
                  " edge array -- is `a` the same lattice?" );
        }
        w.aslot[ std::size_t( e ) ] = j + a_;
        //  the `bc_edge == 1` wall-tangent mask,  the host's
        //  own rule:  an edge that touches a non-periodic
        //  wall in a direction other than its own is zeroed
        for( int q = 0; q < 3; ++ q )
        {
            if( q == a_ || c.per[ q ] )
            {
                continue;
            }
            if( int( nd[ q ] ) == 0
             || int( nd[ q ] ) == int( c.n[ q ] ) )
            {
                w.ewall[ std::size_t( e ) ] = 1;
                ++ w.n_wall;
                break;
            }
        }
    }
    //  ---- the CSR must be ascending in the cell index
    for( int e = 0; e < ne; ++ e )
    {
        for( int q = w.eptr[ std::size_t( e ) ] + 1;
             q < w.eptr[ std::size_t( e ) + 1 ]; ++ q )
        {
            if( w.ecell[ std::size_t( q ) ]
                < w.ecell[ std::size_t( q ) - 1 ] )
            {
                throw std::runtime_error
                    ( "wb0_build:  the CSR is not ascending --"
                      " the device sum would not be the host's." );
            }
        }
    }
    return;
};

////////////////////////////////////////////////////////////
//  Upload ( frees a previous table first,  so it is safe to
//  re-upload the same struct -- the sabotage runs do ).
////////////////////////////////////////////////////////////

__host__ void wb0_dev_free
( device::base_t & dev, wb0_dev_t & w );

__host__ void wb0_upload
( wb0_dev_t & w, device::base_t & dev );

////////////////////////////////////////////////////////////
//  Build + upload in one call,  dispatching on the policy.
////////////////////////////////////////////////////////////

__host__ void wb0_dev_setup
( wb0_dev_t & w, device::base_t & dev,
  const mesh::geo_t & geo, const cplx_t & c,
  const float_t * far_host, const dat_3d_t< float_t > & a,
  const dat_3d_t< float2_t > & bf );

////////////////////////////////////////////////////////////
//  pass 1 -- eta ( curl B / A ) per cell.
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//  pass 2 -- the edge average,  in the host's order.
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//  pass 3 -- one update per GLOBAL face.
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//  The mode-0 launcher.
//
//      ni_ie_writeback0_dev( dev, w, w0, bf, a, p_lat,
//                            dt, bc_edge, n_th_lim );
//
//  `w` is the MODE-1 owner table ( slot + owner area +
//  degenerate flag ) -- pass 3 reuses it rather than a second
//  copy of the same 16 bytes a face.  `w0` is this file's
//  mode-0 table.  `p_lat` is the per-cell tensor lattice on
//  the device ( `st.p_lat`,  six `float2_t` a cell ).
//
//  Per step this costs three launches and ONE barrier:  no
//  array is copied,  the host touches nothing but `dt` and
//  `bc_edge`.  The table is rebuilt only when the block's
//  geometry,  the complex,  or the `a` / `bf` lattices
//  change -- never per step.
////////////////////////////////////////////////////////////

__host__ void ni_ie_writeback0_dev
( device::base_t & dev, const wb_dev_t & w,
  const wb0_dev_t & w0, dat_3d_t< float2_t > & bf,
  const dat_3d_t< float_t > & a, const float2_t * p_lat,
  const float_t & dt, const int & bc_edge,
  const int & n_th_lim );

};                              // namespace mhd_ee::ni::edge_mg
