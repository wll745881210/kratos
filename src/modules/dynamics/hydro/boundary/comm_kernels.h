#pragma once

namespace dynamics::hydro::boundary
{
////////////////////////////////////////////////////////////
// Kernels for hydrodynamics boundary launches
// #warning "Can we put the function name in the template?"

template< class bdt_T, class bnd_T >
__global__ void ker_to_buf
( const bdt_T bdt, const bnd_T bnd, const int step )
{
    if     ( bnd.d_lvl == 0 )
        bnd.to_same    ( bdt, step );
    else if( bnd.d_lvl >  0 )
        bnd.to_finer   ( bdt, step );        
    else
        bnd.to_coarser ( bdt, step );
    return;
}

template< class bdt_T, class bnd_T >
__global__ void ker_from_buf
( const bdt_T bdt, const bnd_T bnd, const int step )
{
    bnd.from_buf    ( bdt, step );
    return;
}

template< class bdt_T, class bnd_T > __global__ void
ker_flux( const bdt_T bdt, const bnd_T bnd, const int step )
{
    if( bnd.d_lvl < 0 )
        bnd.flux_to_buf  ( bdt, step );
    else 
        bnd.flux_from_buf( bdt, step );
    return;
}

};  // namespace dynamics::hydro::boundary
