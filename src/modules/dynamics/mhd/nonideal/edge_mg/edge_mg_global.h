#pragma once

#include <vector>

#include "edge_mg_cplx.h"
#include "edge_mg_csr.h"
#include "edge_mg_data.h"

////////////////////////////////////////////////////////////
//  FROM PER-CELL BLOCKS TO THE BLOCK OPERATOR.
//
//  One implementation,  so that every consumer assembles the
//  same operator:  scattered copies drift apart,  and the
//  operator is the thing every claim is about.
//
//  A is assembled in the SAME pass as M1 and K, so the
//  three
//  share their sparsity BY CONSTRUCTION. That is what makes
//      A = M1 + dt K
//  a fact rather than an assumption: the MG forms its S the
//  same way (`S.val += dt * K.val` over the same pattern),
//  and the two are compared on the compact numbering.
//
//  THE NUMBERING IS THE CALLER'S,  AND THERE ARE TWO:
//
//  drop_t::compact the degenerate edges ( zero-length phi
//  edges on a pole ring ) are OMITTED and
//  the rest are renumbered through `comp`.
//  This is the MG's `keep/comp` numbering,
//  so the MG and the flat CG see the SAME
//                      vector.
//
//  drop_t::identity they stay in the matrix with a 1 on the
//  diagonal. The raw edge numbering is kept.
//
//  In BOTH cases the RIGHT-HAND SIDE must follow: a dropped
//  DOF is not a degree of freedom, and nothing may drive it
//  ( the identity row alone reads a = f + l, because the
//  source and the load still scatter into it ).
//  `reduce_rhs` / `expand_sol` below are that rule,  in one
//  place,  for both policies.

namespace mhd_ee::ni::edge_mg
{
using type::float_t;


enum class drop_t : int
{
    compact  = 0,
    identity = 1,
    //  RAW: no filter, no identity rows, the raw edge
    //  numbering. This is
    //  what the hierarchy builder expects -- it owns the compaction
    //  and MUST NOT be
    //  handed an already compact matrix, because with no
    //  poles the two
    //  sizes are equal and it would compact the compact one
    //  through `comp`,
    //  a permutation, i.e. index garbage.
    raw      = 2,
};

//  M1,  K  and  A = M1 + dt K  of one block.
//
//  `p_blk` is in the flat cell order ( i fastest ), the
//  order
//  blk_assemble_k writes and cell_edges expects.
//  `keep` / `comp` come from make_keep;  `n_compact` is its
//  output size ( ignored by the identity policy ).
__host__ void assemble_m1k_a
( const cplx_t & c, const cell_blk_t * p_blk,
  const std::vector< int > & keep, const std::vector< int > & comp,
  const int & n_compact, const drop_t & mode, const float_t
      & dt,
  csr_t & M1, csr_t & K, csr_t & A );

//  A right-hand side given on the RAW edges -> the
//  numbering the operator
//  uses, with the dropped entries REMOVED ( compact ) or
//  ZEROED
//  ( identity ).
__host__ void reduce_rhs
( const int & n_edge, const std::vector< int > & keep,
  const std::vector< int > & comp, const drop_t & mode,
  const std::vector< float_t > & b_raw, std::vector< float_t
      > & b );

//  A solution in the operator's numbering -> the RAW edges.
//  A dropped DOF
//  gets 0., which is its exact value ( its line integral is
//  identically
//  zero ),  so nothing has to be special-cased downstream.
template< class x_T, class xr_T >
__host__ inline void expand_sol
( const int & n_edge, const std::vector< int > & comp,
  const drop_t & mode, const std::vector< x_T > & x,
  std::vector< xr_T > & x_raw )
{
    x_raw.assign( std::size_t( n_edge ), 0. );
    if( mode == drop_t::compact )
    {
        for( int e = 0; e < n_edge; ++ e )
        {
            if( comp[ e ] >= 0 && comp[ e ] < int( x.size(  ) ) )
            {
                x_raw[ e ] = x[ comp[ e ] ];
            }
        }
        return;
    }
    for( int e = 0; e < n_edge && e < int( x.size(  ) ); ++ e )
    {
        x_raw[ e ] = x[ e ];
    }
    return;
};

};  // namespace edge_mg
