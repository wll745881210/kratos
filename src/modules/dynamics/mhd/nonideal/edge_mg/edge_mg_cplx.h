#pragma once

#include <vector>

#include "../../../../../../src/types.h"

////////////////////////////////////////////////////////////
//  The cubical cell complex of ONE block, on the node set.
//
//  Built exactly as docs/mhd/nonideal/proto/p8_complex.py:
//  nodes are indexed by the node set,  edges by
//  (lower node, direction),  faces by (lower node, normal),
//  cells by the cell index,  and periodicity is the
//  identification of node indices modulo n.  d1 d0 = 0 and
//  d2 d1 = 0 are then integer identities,  not
//  approximations ( the boundary of a boundary is empty ).
//
//  Conventions ( identical to the Python side, so the local
//  blocks can be compared element by element ):
//
//    NN[ a ]   = n[ a ]  if periodic  else  n[ a ] + 1
//    edge (a)  : i_a in [0, n_a),  transverse in [0, NN)
//    face (a)  : i_a in [0, n_a  if periodic else n_a + 1),
//                transverse in [0, n)
//    cell      : all in [0, n)
//
//  Index order inside a direction:  axis 0 fastest,  as in
//  the rest of Kratos.
//
//  This class is host-only ( it holds no pointers, so it is
//  also copyable to the device if a kernel ever needs it ).

namespace mhd_ee::ni::edge_mg
{

using type::   idx_t;
using type:: float_t;

struct cplx_t
{
    ////////// Data //////////
    idx_t       n;          // cells per axis
    int      per[ 3 ];      // periodic flags
    idx_t      NN;          // nodes per axis
    int     n_node;         // totals
    int      n_edge;
    int      n_face;
    int       n_cell;
    int    off_e[ 3 ];      // edge index offset of direction a
    int    st_e [ 3 ][ 3 ]; // edge index strides
    int    dim_e[ 3 ][ 3 ]; // edge index extents
    int    off_f[ 3 ];      // face index offset of each normal
    int    st_f [ 3 ][ 3 ];
    int    dim_f[ 3 ][ 3 ];
    //  pole flags per theta end ( 1 when that end has NO
    //  wall, i.e. the
    //  domain reaches the axis ). Both zero by default, so
    //  every theta
    //  wedge is bit-for-bit unchanged.
    int      pole[ 2 ];

    ////////// Functions //////////
    __host__ __device__ int wrap
    ( const int & a, const int & q ) const
    {
        const int N = int( NN[ q ] );
        int v = a % N;
        return v < 0 ? v + N : v;
    };

    //  POLE FOLDING ( spec 5.7 ): on a pole ring every phi
    //  slot is the
    //  SAME physical node, so the index folds to k = 0. The
    //  node lattice
    //  keeps its size ( the folded slots simply become
    //  unreachable ), which
    //  keeps every offset table and the device arithmetic
    //  unchanged; the
    //  consumers that must know about degeneracy use
    //  degenerate_e() below.
    //
    //  pole[0] / pole[1] are set when the theta axis has NO
    //  wall at that
    //  end, i.e. when the domain reaches the axis. Both
    //  default to zero,
    //  so a theta wedge is bit-for-bit what it was before.
    __host__ __device__ int node( const idx_t & i ) const
    {
        int k = int( i[ 2 ] );
        if( ( pole[ 0 ] != 0 && int( i[ 1 ] ) == 0 )
         || ( pole[ 1 ] != 0 && int( i[ 1 ] ) == int( NN[ 1 ] ) - 1 ) )
        {
            k = 0;
        }
        return ( int( i[ 0 ] ) * int( NN[ 1 ] ) + int( i[ 1 ] ) )
             * int( NN[ 2 ] ) + k;
    };

    //  Edge along `a` whose LOWER node is `nd` ( the
    //  a-index
    //  of `nd` is the edge's own index; the transverse ones
    //  are wrapped when that axis is periodic ).
    __host__ __device__ int edge
    ( const int & a, const idx_t & nd ) const
    {
        int v = off_e[ a ];
        for( int q = 0; q < 3; ++ q )
        {
            int iq = nd[ q ];
            if( q == a )
            {
                iq = iq % int( n[ a ] );     // periodicity
            }
            else if( per[ q ] )
            {
                iq = wrap( iq, q );
            }
            v += iq * st_e[ a ][ q ];
        }
        return v;
    };

    //  Face normal to `a` whose lower node is `nd`.
    __host__ __device__ int face
    ( const int & a, const idx_t & nd ) const
    {
        int v = off_f[ a ];
        for( int q = 0; q < 3; ++ q )
        {
            int iq = nd[ q ];
            if( q == a )
            {
                const int na = int( n[ a ] );
                iq = per[ a ] ? ( iq % na ) : iq;
            }
            else
            {
                iq = iq % int( n[ q ] );
            }
            v += iq * st_f[ a ][ q ];
        }
        return v;
    };

    //  The 12 local edges of cell i,  in the Python's order
    //      ED = [ (a, sb, sl) for a in 0..2
    //  for sb in 0..1 for sl in 0..1 ]
    __host__ __device__ void cell_edges
    ( const idx_t & i, int ( & e )[ 12 ] ) const
    {
        int k = 0;
        for( int a = 0; a < 3; ++ a )
        {
            const int b = ( a + 1 ) % 3;
            const int l = ( a + 2 ) % 3;
            for( int sb = 0; sb < 2; ++ sb )
            {
                for( int sl = 0; sl < 2; ++ sl )
                {
                    idx_t nd( i );
                    nd[ b ] += sb;
                    nd[ l ] += sl;
                    e[ k ] = edge( a, nd );
                    ++ k;
                }
            }
        }
        return;
    };

    //  The 6 local faces,  FD = [ (a, s) for a in 0..2
    //                                  for s in 0..1 ]
    __host__ __device__ void cell_faces
    ( const idx_t & i, int ( & f )[ 6 ] ) const
    {
        int k = 0;
        for( int a = 0; a < 3; ++ a )
        {
            for( int s = 0; s < 2; ++ s )
            {
                idx_t nd( i );
                nd[ a ] += s;
                f[ k ] = face( a, nd );
                ++ k;
            }
        }
        return;
    };

    ////////// Setup //////////
    //  an edge whose two endpoints are the SAME physical
    //  node: a phi edge
    //  on a pole ring ( zero length ). Such an edge is not
    //  a degree of
    //  freedom of the 1-form space -- its line integral is
    //  identically zero
    //  -- so it must be dropped from the system ( keep = 0
    //  ), and the
    //  assembler must not scatter into it.
    __host__ __device__ bool degenerate_e
    ( const int & e ) const
    {
        int   a( 0 );
        idx_t nd( 0 );
        edge_key( e, a, nd );
        if( a != 2 )
        {
            return false;
        }
        idx_t hd( nd );
        hd[ 2 ] = ( int( hd[ 2 ] ) + 1 ) % int( NN[ 2 ] );
        return node( nd ) == node( hd );
    };

    __host__ void setup
    ( const idx_t & n_cell_eff, const bool ( & p )[ 3 ] );

    __host__ void setup
    ( const idx_t & n_cell_eff, const bool ( & p )[ 3 ],
      const bool & pole_lo, const bool & pole_hi );

    ////////// Incidence ( host,  for the gates ) //////////
    //  d0: edges -> nodes, +1 at the head, -1 at the tail.
    //  A thin axis ( NN == 1 ) gives an empty row ( the
    //  edge is a closed loop,  the discrete analogue of a
    //  degenerate polar phi-edge ).
    __host__ void d0_row
    ( const int & e, int ( & nd )[ 4 ], int ( & sg )[ 4 ],
      int & nz ) const;

    //  Decompose an edge index back into ( direction, lower
    //  node ). The node is the UNWRAPPED one: on a periodic
    //  axis the last edge of the row has its head at index
    //  n_a,  which is the same physical node as 0 but a
    //  different coordinate -- anything that turns an edge
    //  into a geometric vector ( the gates' linear-field
    //  probe,  the write-back metric ) must use this form.
    __host__ __device__ void edge_key
    ( const int & e, int & a, idx_t & nd ) const
    {
        int a_off( e );
        for( a = 2; a >= 0; -- a )
        {
            if( a_off >= off_e[ a ] )
            {
                break;
            }
        }
        a_off -= off_e[ a ];
        for( int q = 0; q < 3; ++ q )
        {
            nd[ q ] = ( a_off / st_e[ a ][ q ] ) % dim_e[ a ][ q ];
        }
        return;
    };

    //  A theta-normal face ON a pole ring is a line: its
    //  phi extent has
    //  collapsed, so its area is zero, its flux DOF is
    //  identically zero,
    //  and it must leave the 2-form space exactly as the
    //  zero-length phi
    //  edges leave the 1-form one. r-normal faces at the
    //  pole are the
    //  polar caps and are NOT degenerate ( they keep a
    //  finite area ).
    __host__ __device__ bool degenerate_f
    ( const int & f ) const
    {
        int   a( 0 );
        idx_t nd( 0 );
        face_key( f, a, nd );
        if( a != 1 )
        {
            return false;
        }
        if( pole[ 0 ] != 0 && int( nd[ 1 ] ) == 0 )
        {
            return true;
        }
        if( pole[ 1 ] != 0 && int( nd[ 1 ] ) == int( n[ 1 ] ) )
        {
            return true;
        }
        return false;
    };

    //  Decompose a face index back into ( normal, lower
    //  node ).
    __host__ __device__ void face_key
    ( const int & f, int & a, idx_t & nd ) const
    {
        int a_off( f );
        for( a = 2; a >= 0; -- a )
        {
            if( a_off >= off_f[ a ] )
            {
                break;
            }
        }
        a_off -= off_f[ a ];
        for( int q = 0; q < 3; ++ q )
        {
            nd[ q ] = ( a_off / st_f[ a ][ q ] ) % dim_f[ a ][ q ];
        }
        return;
    };

    //  d1: faces -> edges, as the boundary of the face's
    //  own
    //  corner cycle, walked counter-clockwise about +a in
    //  the
    //  (b, l) = (a+1, a+2) plane. The sign is +1 when the
    //  walk
    //  increases that coordinate.
    __host__ void d1_row
    ( const int & f, int ( & ed )[ 4 ], int ( & sg )[ 4 ],
      int & nz ) const;
};

//  A primal edge lying IN a physical wall: its direction is
//  tangential to that wall and its lower node is on either
//  wall.  For bc_edge = 2 ( n x B = 0 ) its line
//  integral is a homogeneous Dirichlet datum, not a solver
//  unknown.
__host__ __device__ inline bool wall_tangent_e
( const cplx_t & c, const int & e )
{
    int   a( 0 );
    idx_t nd( 0 );
    c.edge_key( e, a, nd );
    for( int q = 0; q < 3; ++ q )
    {
        if( q == a || c.per[ q ] )
        {
            continue;
        }
        if( int( nd[ q ] ) == 0
         || int( nd[ q ] ) == int( c.n[ q ] ) )
        {
            return true;
        }
    }
    return false;
};

//  The live-dof mask of a 1-form space: a DEGENERATE edge (
//  zero length,
//  on a pole ring ) is not a degree of freedom, so keep = 0
//  there and comp
//  compacts the rest. Everything that scatters into a
//  global vector must go
//  through comp -- otherwise the degenerate rows are
//  structurally zero and
//  the operator is singular.
//  The compact numbering. It is a CONVENTION: every object
//  that enters the
//  operator ( M1, K, G, C, P, the nodewise sites )
//  addresses the compact space
//  through `comp`, so a PERMUTATION here is transparent to
//  all of them.
//
//  `site_major` permutes it so that the up-to-three edges
//  of one NODE get
//  CONSECUTIVE compact indices. Why: the nodewise smoother
//  then reads three
//  CONSECUTIVE rows of the CSR instead of three scattered
//  ones, which is the
//  only remaining limiter of that kernel ( measured: thread
//  count and the
//  dependent-load chain both matter not, foundations 9z-x
//  ).
__host__ void make_keep
( const cplx_t & c, std::vector< int > & keep,
  std::vector< int > & comp, int & n_compact,
  const bool & site_major = false, const int & bc_edge = 0 );

//  the same mask for the 2-form ( face ) space
__host__ void make_keep_face
( const cplx_t & c, std::vector< int > & keep,
  std::vector< int > & comp, int & n_compact );

};  // namespace edge_mg
