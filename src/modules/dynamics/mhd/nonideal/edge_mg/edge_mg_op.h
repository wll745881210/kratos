#pragma once
//
//  edge_mg_op.h -- the ASSEMBLED operators of a hierarchy,  built
//  from per-cell blocks for the one purpose that still wants them:
//  checking the matrix-free operator against an independently
//  assembled one ( the G-AGG invariant in the P5 gates ).
//
//  Nothing in the production step solves with these;  method 5 only
//  ever applies `A_l` through the device blocks.  This is the same
//  arithmetic `mg_t::setup` used to do for the host V-cycle,  minus
//  the auxiliary spaces,  the site blocks and the coarsest
//  factorisation.

#include <vector>

#include "edge_mg_hier.h"
#include "edge_mg_global.h"

namespace mhd_ee::ni::edge_mg
{
using type::float_t;


////////////////////////////////////////////////////////////
//  One level's assembled triple

struct op_lvl_t
{
    csr_t                   M1, K;   // compact space
    csr_t                      A;   // M1 + dt K
};


//  M1 + dt K,  in the same pattern as M1 ( the buffers are not
//  shared,  so this is a plain elementwise blend ).
inline csr_t _add_dt
( const csr_t & M1, const csr_t & K, const float_t & dt )
{
    if( M1.n != K.n || M1.ptr.size(  ) != K.ptr.size(  )
        || M1.idx != K.idx )
    {
        throw std::runtime_error
        ( "op_build:  the M1 and K patterns differ -- they are built by"
          " one assembly pass and must not." );
    }
    csr_t A;
    A.n = M1.n;
    A.ncol = M1.ncol;
    A.ptr = M1.ptr;
    A.idx = M1.idx;
    A.val.resize( M1.val.size(  ) );
    for( std::size_t i = 0; i < M1.val.size(  ); ++ i )
    {
        A.val[ i ] = M1.val[ i ] + dt * K.val[ i ];
    }
    return A;
}

////////////////////////////////////////////////////////////
//  Build every level from the blocks and the transfers.
//
//  Level 0 is assembled straight from the blocks ( `drop_t::compact`
//  -- the numbering is `h.L[0]`'s ),  every coarse level is the
//  Galerkin product of the one below it,  with M1 and K carried
//  SEPARATELY so `dt` is never baked into the transfers.

inline void op_build
( const hier_t & h,  const cell_blk_t * p_blk,  const float_t & dt,
  std::vector< op_lvl_t > & op )
{
    op.clear(  );
    op.resize( h.L.size(  ) );
    {
        csr_t & A0 = op[ 0 ].A;
        assemble_m1k_a( h.L[ 0 ].c, p_blk, h.L[ 0 ].keep, h.L[ 0 ].comp,
                        h.L[ 0 ].n_compact, drop_t::compact, dt,
                        op[ 0 ].M1, op[ 0 ].K, A0 );
    }
    for( int l = 0; l + 1 < int( h.L.size(  ) ); ++ l )
    {
        const int nc = h.L[ size_t( l + 1 ) ].n_compact;
        op[ size_t( l + 1 ) ].M1
            = galerkin( op[ size_t( l ) ].M1, h.L[ size_t( l ) ].P, nc );
        op[ size_t( l + 1 ) ].K
            = galerkin( op[ size_t( l ) ].K,  h.L[ size_t( l ) ].P, nc );
        op[ size_t( l + 1 ) ].A
            = _add_dt( op[ size_t( l + 1 ) ].M1, op[ size_t( l + 1 ) ].K, dt );
        if( l == 0 )
        {
            //  level 0's A came out of the assembly with dt already in,
            //  which is fine -- but re-form it the same way so all the
            //  levels are built by one code path.
            op[ 0 ].A = _add_dt( op[ 0 ].M1, op[ 0 ].K, dt );
        }
    }
}

}   //  namespace mhd_ee::ni::edge_mg
