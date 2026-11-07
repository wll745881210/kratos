#pragma once

namespace multigrid
{
////////////////////////////////////////////////////////////
// Enumerate types

enum   rb_t{ red   =  0,           black   = 1 };
// Red-black stages of Gauss-Siedel iterations

enum dlvl_t{ finer = -1, same = 0, coarser = 1 };
// Level-change directions

enum opt_flag_t : int
{ output_defect = ( 1 << 0 ) , subtract_src  = ( 1 << 1 ),
  subtract_fld  = ( 1 << 2 ) } ;

};
