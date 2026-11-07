#pragma once

namespace multigrid::kernel
{
////////////////////////////////////////////////////////////
// Generic kernel templates

template  < class bdt_T, class opt_T >     __global__ void
refine( const bdt_T b_c, const bdt_T  b_f, const opt_T opt )
{
    return opt.refine( b_c,   b_f );
}
template  < class bdt_T, class opt_T >     __global__ void 
coarse( const bdt_T b_c, const bdt_T  b_f, const opt_T opt )
{
    return opt.coarsen ( b_c,   b_f );
}
template  < class bdt_T, class opt_T >     __global__ void
smooth( const bdt_T b_d, const rb_t stage, const opt_T opt )
{  
    return opt.smooth  ( b_d, stage );
}
template< class bdt_T, class opt_T > __global__
void residual( const bdt_T b_d, const opt_T opt )
{  
    return opt.residual( b_d );
}
template< class bdt_T, class opt_T > __global__
void calc_ave( const bdt_T b_d, const opt_T opt )
{  
    return opt.calc_ave( b_d );
}
template< class bdt_T, class opt_T > __global__
void sub_ave( const bdt_T b_d, const opt_T opt )
{  
    return opt.sub_ave( b_d );
}
template< class bdt_T, class opt_T > __global__
void defect ( const bdt_T b_d, const opt_T opt )
{  
    return opt.defect( b_d );
}
};                           // namespace multigrid::kernel
