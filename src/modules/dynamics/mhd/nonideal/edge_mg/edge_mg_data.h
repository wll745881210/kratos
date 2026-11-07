#pragma once

#include "../../../../../../src/types.h"
#include "../../../../../../src/mesh/types/dat_3d.h"
#include "../../../../../../src/mesh/geometry/geometry.h"

////////////////////////////////////////////////////////////
//  Edge-centred implicit tensor diffusion: the data the
//  solver works on.
//
//  Everything here is NON-TEMPLATE and trivially copyable
//  ( rule C2 of docs/mhd/nonideal/edge-mg-foundations.md
//  section 3.3 ):  the device code never sees a pgen
//  specific block-data type,  so the pgen TU instantiates
//  no kernel of this module.

namespace mhd_ee::ni::edge_mg
{

using type::   idx_t;
using type:: float_t;
using type::float2_t;
using mesh:: dat_3d_t;
using mesh::   geo_t;

////////////////////////////////////////////////////////////
//  Per-block view.
//
//  Lattices ( all on the NODE box of the block,  i.e. one
//  plane more than the cell box in each direction,  exactly
//  like the mhd module's  ee / j_edge / e_dif ):
//
//    a     the edge 1-form,  n_int = 3,  (i,j,k) = the
//          LOWER node of the edge,  [a] = its direction;
//    eta   the symmetric tensor eta at the cell centre,
//          n_int = 6,  the Voigt order of  eta_cc_t
//          { xx, yy, zz, xy, xz, yz }.
//
//  The solver reads bf / w and writes nothing but a / eta
//  at S0' ( the identity step of section 9 ).

struct eta_cc_t
{
    float_t xx, yy, zz, xy, xz, yz;
};

struct view_t
{
    geo_t                 geo;
    dat_3d_t< float2_t >   bf;
    dat_3d_t< float_t  >    w;
    dat_3d_t< float_t  >    a;
    dat_3d_t< float_t  >  eta;
    float_t                 dt;
};

////////////////////////////////////////////////////////////
//  Level-0 bookkeeping.  The coarse levels arrive with the
//  MG ( S2 );  this is the shape they will hang on:  one
//  grid per level,  the level's own n_ceff,  and the
//  per-level coefficient stencils ( fixed length,  see
//  spec section 4.2 ).

////////////////////////////////////////////////////////////
//  The per-cell cache slot:  ONE struct,  ONE array
//  ( dat_3d_t< cell_blk_t > ),  so a cell's whole local
//  assembly is contiguous ( 1600 B,  128 B aligned ).
//
//  Only m1 / m2 / m12 / detj are needed for the gates;  k
//  is the eta-dependent half of the operator and is the one
//  that gets rebuilt when eta or dt changes
//  ( edge-mg-foundations.md 4.0:  dt is NEVER baked in ).

struct cell_blk_t
{
    ////////// Data //////////
    float_t   m1[ 144 ];      // 12 x 12,  row-major
    float_t    k[ 144 ];
    float_t   m2[  36 ];      // 6 x 6
    float_t  m12[  72 ];      // 12 x 6
    float_t detj;
    float_t  pad[  3 ];       // -> 400 floats = 1600 B
};

struct level_t
{
    ////////// Data //////////
    idx_t     n_ceff;
    int          n_gh;
    int       n_level;
};

////////////////////////////////////////////////////////////

struct wb_face_t
{
    int         slot;   // bf.idx( nd ) + a  ( idx carries n_int )
    float_t       ar;   // owner-cell area,  st.far[ ic*6+2a+s ]
    int            a;   // face normal  ( the bf component )
    int            s;   // 0 = low side,  1 = high wall
    int           ic;   // owner cell flat index ( diagnostics )
    int          deg;   // 1 = degenerate pole-ring theta face
};

};  // namespace edge_mg
