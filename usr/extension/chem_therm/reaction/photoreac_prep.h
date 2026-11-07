#pragma once

#include "reaction_list.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
//

template< class bdt_T, class eos_T >
__device__ void list_t::prep_rad
( const bdt_T & bdt, const eos_T & eos ) const
{
    const int i_th( threadIdx.x );
    const int n_hv( bdt.rad.n_fld(  ) );

    int i[ 3 ];
    i[ 0 ] = blockIdx.x;
    i[ 1 ] = blockIdx.y;
    i[ 2 ] = blockIdx.z;
    auto * mfpa_i = bdt.rad.mfpa_i.at( i );
    auto * u      = bdt.hyd.u     .at( i ) + eos.n_hyd;

    __dyn_shared__  ( char, sh );
    auto * var_sh = ( type::float_t * ) ( sh );
    auto *   p_sh = var_sh + i_th * n_hv;
    for( int n = 0; n < n_hv; ++ n )
        p_sh[ n ] = 0;
    __syncthreads(  );

    auto * shield( bdt.rad.shield.at( i ) );
    const int n_bin = n_hv * phot_reac.size;
    int i_hv, i_ph;
    for( int j = i_th; j < n_bin; j += blockDim.x )
    {
        utils ::divmod( i_ph, i_hv, j, n_hv );
        const auto & r( phot_reac [ i_ph  ] );
        const auto i_r( r.i_chem  [ 0 ]     );
        const auto   n( eos.unit.rho0 / phys::cgs::ma
                      * type::float_t ( u[ i_r ] ) );
        type::float_t ext_mod( 1 );
        if( r.i_sh >= 0 )
        {
            ext_mod = r.ext_mod( i_hv, i, bdt, eos );
            if( i_hv == 0 )
                shield[ r.i_sh ] = ext_mod;
        }
        p_sh[ i_hv ] += r.pars [ i_hv ] * n * ext_mod;
    }
    __syncthreads(  );
    if( i_th < n_hv )
    {
        for( int j = 1; j < blockDim.x; ++ j )
            var_sh[ i_th ] += var_sh[ j * n_hv + i_th ];
        mfpa_i[ i_th ] = var_sh[ i_th ] * eos.unit.l0;
    }   // From CGS to code unit ( mfp_i is the inverse )
    return;
}

};                     // namespace extension::chemistry
