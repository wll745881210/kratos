#include "edge_mg_hier.h"

#include <vector>

////////////////////////////////////////////////////////////
//  The host builder of the level hierarchy.
//
//  `build` derives the per-level complex,  the transfer
//  stencils and the per-dof cell counts ONCE per setup;  the
//  device then runs from those tables and never rebuilds
//  them.  The `hier_t` layout and its data stay in the
//  header.

namespace mhd_ee::ni::edge_mg
{

__host__ void hier_t::build
( const idx_t & n0, const bool ( & per )[ 3 ],
  const int ( & cdir )[ 3 ], const int & n_lvl_max,
  const bool & pole_lo, const bool & pole_hi )
{
    L.clear(  );
    {
        mg_lvl_t lv;
        lv.n_ceff = n0;
        lv.c.setup( n0, per, pole_lo, pole_hi );
        make_keep( lv.c, lv.keep, lv.comp, lv.n_compact,
                   site_major != 0, bc_edge );
        make_keep_face( lv.c, lv.keep_f, lv.comp_f, lv.n_compact_f );
        L.push_back( lv );
    }
    while( int( L.size(  ) ) < n_lvl_max
        && mg_can_coarsen( L.back(  ).n_ceff, cdir ) )
    {
        mg_lvl_t lv;
        lv.n_ceff = mg_coarsen( L.back(  ).n_ceff, cdir );
        if( lv.n_ceff == L.back(  ).n_ceff )
        {
            break;
        }
        lv.c.setup( lv.n_ceff, per, pole_lo, pole_hi );
        make_keep( lv.c, lv.keep, lv.comp, lv.n_compact,
                   site_major != 0, bc_edge );
        make_keep_face( lv.c, lv.keep_f, lv.comp_f, lv.n_compact_f );
        L.push_back( lv );
    }
    for( int l = 0; l + 1 < int( L.size(  ) ); ++ l )
    {
        L[ l ].P = build_P( L[ l + 1 ].c, L[ l ].c,
                            L[ l + 1 ].keep, L[ l ].keep,
                            L[ l + 1 ].comp, L[ l ].comp,
                            cdir, L[ l ].n_compact,
                            L[ l + 1 ].n_compact );
    }
};

};    // namespace mhd_ee::ni::edge_mg
