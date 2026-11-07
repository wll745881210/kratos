#pragma once

namespace particle::cic::boundary
{
////////////////////////////////////////////////////////////
// Types

using  namespace    mesh;
using  holder_c_t = mesh::boundary::holder_c_t;
using  holder_c_w = std ::reference_wrapper< holder_c_t >;

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

};                     // namespace particle::cic::boundary
