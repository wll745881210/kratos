#pragma once
//
//  edge_mg_hier.h -- the GEOMETRIC half of the multigrid hierarchy.
//
//  A matrix-free V-cycle does not need an assembled operator, but it
//  does need the geometry:  which cells the coarsening keeps,  the
//  canonical prolongation P ( fine rows x coarse columns ),  and the
//  compact numbering that every level shares with the device.  Those
//  are topology plus the three per-direction weight rules,  so they
//  live here,  where both the production step and the reference-side
//  gates can build them without touching a solver.
//
//  What is NOT here ( and is gone from the line as of P6/P7,  2026-11-06 ):
//  the assembled operators,  the HX auxiliary spaces,  the site blocks,
//  the Chebyshev workspaces,  the dense coarsest factorisation and the
//  host V-cycle.  Those were the `mg_t` half;  method 5 replaced them.
//
//  The three rules of build_P ( spec 5.2 ) are written out one by one,
//  because getting one wrong is the 143x trap ( dead-ends M74:  the
//  uncoarsened own axis must NOT be halved ).

#include <vector>
#include <stdexcept>

#include "edge_mg_cplx.h"
#include "edge_mg_csr.h"

namespace mhd_ee::ni::edge_mg
{
using type::float_t;


////////////////////////////////////////////////////////////
//  One level:  the cell complex,  the compact numbering,  and
//  the prolongation to the next level.  No operator.

struct mg_lvl_t
{
    idx_t                  n_ceff;
    cplx_t                      c;
    std::vector< int >       keep;   // 1 = live dof
    std::vector< int >       comp;   // dof -> compact
    int                 n_compact;
    std::vector< int >       keep_f;   // the 2-form ( face ) space
    std::vector< int >       comp_f;
    int               n_compact_f;
    csr_t                      P;   // fine x coarse ( compact ),  l -> l+1
};

inline csr_t build_P
( const cplx_t & cc, const cplx_t & cf,
  const std::vector< int > & keep_c,
  const std::vector< int > & keep_f,
  const std::vector< int > & comp_c,
  const std::vector< int > & comp_f,
  const int ( & cdir )[ 3 ], const int & n_fine,
  const int & n_coarse )
{
    const int NNc[ 3 ] = { int( cc.NN[ 0 ] ), int( cc.NN[ 1 ] ),
                           int( cc.NN[ 2 ] ) };
    bool coarse[ 3 ];
    for( int q = 0; q < 3; ++ q )
    {
        coarse[ q ] = ( cdir[ q ] != 0 )
                   && ( int( cc.NN[ q ] ) != int( cf.NN[ q ] ) );
    }
    std::vector< int > rows, cols;
    std::vector< float_t > vals;
    for( int ef = 0; ef < cf.n_edge; ++ ef )
    {
        if( ! keep_f[ ef ] )
        {
            continue;
        }
        int a( 0 );
        idx_t base( 0 );
        cf.edge_key( ef, a, base );
        int    oi[ 3 ][ 2 ];
        float_t ow[ 3 ][ 2 ];
        int    on[ 3 ];
        for( int q = 0; q < 3; ++ q )
        {
            const int i = base[ q ];
            if( ! coarse[ q ] )
            {
                //  An axis the coarsening does NOT touch is the identity
                //  in both index and weight -- including the edge's own
                //  axis.  Halving here ( the pre-P3 form did,  for q == a
                //  regardless of `coarse[q]` ) folds a pair of fine
                //  edges onto one coarse edge: P then maps into the
                //  pair-constant subspace while the coarse complex keeps
                //  the full axis,  so d1 P != P_f d1_c ( gate C7 ) and
                //  the Galerkin coarse operator no longer belongs to the
                //  stored coarse complex.
                oi[ q ][ 0 ] = i % NNc[ q ];
                ow[ q ][ 0 ] = 1.;
                on[ q ] = 1;
            }
            else if( q == a )
            {
                oi[ q ][ 0 ] = ( i / 2 ) % NNc[ q ];
                ow[ q ][ 0 ] = 1.;
                on[ q ] = 1;
            }
            else if( ( i % 2 ) == 0 )
            {
                oi[ q ][ 0 ] = ( i / 2 ) % NNc[ q ];
                ow[ q ][ 0 ] = 1.;
                on[ q ] = 1;
            }
            else
            {
                oi[ q ][ 0 ] = ( i / 2     ) % NNc[ q ];
                oi[ q ][ 1 ] = ( i / 2 + 1 ) % NNc[ q ];
                ow[ q ][ 0 ] = 0.5;
                ow[ q ][ 1 ] = 0.5;
                on[ q ] = 2;
            }
        }
        const float_t fac = coarse[ a ] ? 0.5 : 1.;
        for( int c0 = 0; c0 < on[ 0 ]; ++ c0 )
        {
            for( int c1 = 0; c1 < on[ 1 ]; ++ c1 )
            {
                for( int c2 = 0; c2 < on[ 2 ]; ++ c2 )
                {
                    idx_t nd( oi[ 0 ][ c0 ], oi[ 1 ][ c1 ],
                              oi[ 2 ][ c2 ] );
                    const int ec = cc.edge( a, nd );
                    if( ec < 0 || ec >= cc.n_edge || ! keep_c[ ec ] )
                    {
                        continue;
                    }
                    rows.push_back( comp_f[ ef ] );
                    cols.push_back( comp_c[ ec ] );
                    vals.push_back( fac * ow[ 0 ][ c0 ]
                                        * ow[ 1 ][ c1 ]
                                        * ow[ 2 ][ c2 ] );
                }
            }
        }
    }
    csr_t P;
    //  the COLUMN count is the coarse compact size, given
    //  explicitly:
    //  a coarse edge that no fine edge touches would
    //  otherwise make the
    //  pattern-derived count too small, and P^T would then
    //  return a
    //  SHORT vector ( that is what made the V-cycle index
    //  out of range
    //  in the first ncol attempt,  2026-10-29 ).
    //  the column count is the COARSE COMPACT size, not the
    //  coarse edge
    //  count: with the pole rings dropped those differ, and
    //  using the
    //  edge count made P's columns run past the compact
    //  operator.
    P.build( n_fine, n_coarse, rows, cols, vals );
    return P;
};

////////////////////////////////////////////////////////////
//  Coarsening ( E1 ).  n_lvl is a COLLECTIVE quantity in a
//  multi-block run ( reduce it across blocks );  the single
//  block gates do not need the reduction.


inline int mg_can_coarsen( const idx_t & n, const int ( & cdir )[ 3 ] )
{
    for( int q = 0; q < 3; ++ q )
    {
        if( cdir[ q ] != 0 && int( n[ q ] ) > 1 && int( n[ q ] ) < 3 )
        {
            return 0;
        }
    }
    return 1;
};

inline idx_t mg_coarsen( const idx_t & n, const int ( & cdir )[ 3 ] )
{
    idx_t r( n );
    for( int q = 0; q < 3; ++ q )
    {
        if( cdir[ q ] != 0 && int( n[ q ] ) > 1 )
        {
            r[ q ] = std::max( 2, int( n[ q ] ) / 2 );
        }
    }
    return r;
};

////////////////////////////////////////////////////////////
//  A_c = P^T A P


inline csr_t galerkin( const csr_t & A, const csr_t & P,
                       const int & nc_in )   // nc = P's COLUMN count
{
    int nc = nc_in;
    //  LEFT-LOOKING, ROW BY ROW: A_c = P^T A P, one COARSE
    //  ROW at a
    //  time, accumulated into a dense workspace and emitted
    //  already
    //  sorted. No global triplet list, so no O(M log M)
    //  sort over all
    //  the products -- that sort was 15 s of one-off build
    //  at
    //  (8,16,32) ( measured 2026-10-29 ).
    //
    //  The pattern is STRUCTURAL: entries whose value
    //  happens to be zero
    //  are emitted too, so P^T M1 P and P^T K P end up with
    //  identical
    //  patterns and can be combined element-wise per dt.
    //
    //  Row c1 of A_c is
    //  sum_{ (f1,p1) in P[:,c1] } sum_{ f2 } A[f1,f2]
    //  P[f2,:] * p1
    //  so the outer loop walks the COLUMN view of P.
    const int nf = P.n;
    int nc_p = P.ncol > 0 ? P.ncol : nc;
    for( std::size_t q = 0; q < P.idx.size(  ); ++ q )
    {
        if( P.idx[ q ] + 1 > nc_p )
        {
            std::printf( "  [mg] WARNING: transfer column %d exceeds the"
                         " declared ncol %d -- enlarging the workspace\n",
                         P.idx[ q ], nc_p );
            nc_p = P.idx[ q ] + 1;
        }
    }
    if( nc_p > nc )
    {
        nc = nc_p;      //  placeholder, replaced below
    }
    std::vector< int > cstart( nc_p + 1, 0 );
    for( int q = 0; q < int( P.idx.size(  ) ); ++ q )
    {
        ++ cstart[ P.idx[ q ] + 1 ];
    }
    for( int c = 0; c < nc_p; ++ c )
    {
        cstart[ c + 1 ] += cstart[ c ];
    }
    std::vector< int    >  prow( P.idx.size(  ) );
    std::vector< float_t >  pval( P.val.size(  ) );
    {
        std::vector< int > cur( cstart.begin(  ), cstart.end(  ) - 1 );
        for( int f = 0; f < nf; ++ f )
        {
            for( int q = P.ptr[ f ]; q < P.ptr[ f + 1 ]; ++ q )
            {
                const int c = P.idx[ q ];
                const int t = cur[ c ] ++;
                prow[ t ] = f;
                pval[ t ] = P.val[ q ];
            }
        }
    }
    csr_t out;
    out.n    = nc;
    out.ncol = nc;
    out.ptr.assign( nc + 1, 0 );
    std::vector< float_t > w( nc_p, 0. );
    std::vector< int > touched;
    touched.reserve( 64 );
    for( int c1 = 0; c1 < nc; ++ c1 )
    {
        touched.clear(  );
        for( int u = cstart[ c1 ];
             u < cstart[ c1 + 1 ] && c1 < nc_p; ++ u )
        {
            const int    f1 = prow[ u ];
            const float_t p1 = pval[ u ];
            for( int qa = A.ptr[ f1 ]; qa < A.ptr[ f1 + 1 ]; ++ qa )
            {
                const int    f2 = A.idx[ qa ];
                const float_t a12 = A.val[ qa ] * p1;
                for( int qb = P.ptr[ f2 ]; qb < P.ptr[ f2 + 1 ]; ++ qb )
                {
                    const int c2 = P.idx[ qb ];
                    if( w[ c2 ] == 0. )
                    {
                        touched.push_back( c2 );
                    }
                    w[ c2 ] += a12 * P.val[ qb ];
                }
            }
        }
        std::sort( touched.begin(  ), touched.end(  ) );
        for( std::size_t t = 0; t < touched.size(  ); ++ t )
        {
            const int c2 = touched[ t ];
            if( t > 0 && touched[ t ] == touched[ t - 1 ] )
            {
                continue;
            }
            out.idx.push_back( c2 );
            out.val.push_back( w[ c2 ] );
            w[ c2 ] = 0.;
        }
        out.ptr[ c1 + 1 ] = int( out.idx.size(  ) );
    }
    return out;
};

////////////////////////////////////////////////////////////
//  Site-block data for block Chebyshev.
//
//  A SITE is the set of edges that START at one node -- up
//  to three, one per direction.  The 3x3 diagonal block of
//  those dofs is inverted once and applied as a block
//  Jacobi preconditioner inside the Chebyshev polynomial.
//
//  The eight buckets are only a STORAGE partition ( parity
//  of the node indices ).  Block Chebyshev does not sweep
//  them sequentially and never updates a residual between
//  buckets: the name `col` below is historical.  The true
//  8-colour Gauss-Seidel was removed with `ee_smoother=1`.
//
//  The representative of an edge is its LOWER node, which
//  is exactly what cplx_t::edge_key returns.


inline void compact_csr
( const csr_t & A, const std::vector< int > & comp, const int & nc,
  csr_t & out )
{
    std::vector< int > rows, cols;
    std::vector< float_t > vals;
    for( int r = 0; r < A.n; ++ r )
    {
        if( comp[ r ] < 0 )
        {
            continue;
        }
        for( int q = A.ptr[ r ]; q < A.ptr[ r + 1 ]; ++ q )
        {
            if( comp[ A.idx[ q ] ] < 0 )
            {
                continue;
            }
            rows.push_back( comp[ r ] );
            cols.push_back( comp[ A.idx[ q ] ] );
            vals.push_back( A.val[ q ] );
        }
    }
    out.build( nc, rows, cols, vals );
    return;
};

////////////////////////////////////////////////////////////
//  diag( B^T S B ) and the Gershgorin radius of its rows,
//  WITHOUT forming
//  the product.
//
//  B has at most four nonzeros per row ( d0: 2, d1: 3 ), so
//  the incident
//  list E( i ) of a space index is tiny and
//
//      (B^T S B)_ii = sum_{a,b in E(i)} B_a B_b S_{j_a j_b}
//  radius_i = sum_{a != b} |B_a B_b S_{j_a j_b}|
//
//  -- the same triangle inequality gershgorin_max uses, so
//  the two agree by
//  construction ( A4.24 checks it ). Each S entry is a
//  binary search in a
//  row whose columns are sorted ( csr_t::build guarantees
//  it ).

////////////////////////////////////////////////////////////
//  The hierarchy:  levels,  numberings,  transfers.  Built once
//  per shape;  a step only refills the per-cell blocks.

struct hier_t
{
    std::vector< mg_lvl_t > L;
    //  the compact numbering of level 0 ( `make_keep` may permute it
    //  with site_major ),  kept here so the step and the gates agree
    int                site_major { 1 };
    int                   bc_edge { 0 };

    __host__ void build
    ( const idx_t & n0, const bool ( & per )[ 3 ],
      const int ( & cdir )[ 3 ], const int & n_lvl_max,
      const bool & pole_lo = false, const bool & pole_hi = false );
};

}   //  namespace mhd_ee::ni::edge_mg
