#pragma once

#include "../solver/dat.h"

namespace chemistry::nuclear
{
////////////////////////////////////////////////////////////
//

template < class f_T, int n_aux = 7 >
struct dat_t : ::chemistry::solver::dat_t< f_T, n_aux >
{
    using super_t
        = ::chemistry::solver::dat_t< f_T, n_aux >;
    using super_t::dim;    
    using super_t::idx;
    using super_t::  x;    
    using super_t::  y;
    using super_t::  e;
    using super_t::rho_cgs;
    using super_t::i_en;

    __device__ dat_t( const int  & dim,  char * p,
                      const bool & cache_f ) :
        super_t( dim, p, cache_f ) {  };
    
    template < class bdt_T, class eos_T >  __device__
    void load( const bdt_T & bdt, const eos_T & eos )
    {
        auto * u_src = bdt.hyd.u.at( idx );
        const f_T rho( u_src[ idx::rho ] );
        rho_cgs = rho * eos. unit. rho0;
        f_T e_cgs( 0 );
        if( threadIdx.x == 0 )
        {
            e_cgs = eos.e_g_cgs( u_src );
            y [ i_en ] = e_cgs / rho_cgs;
        }   // Specific internal energy ( in erg / g )

        this->t  = bdt.dt_0 * eos.unit.t0;
        this->t0 = this->t;
        this->h  = this->t;
        u_src   += eos.n_hyd;
        for( int l = threadIdx.x; l < dim; l += blockDim.x )
             if( l != i_en )
                 y[ l ]  = u_src[ l ] / rho
                    / ( eos.mcgs( l ) * phys::cgs::NA );
        __syncthreads(  );
        this->e =  e_cgs / eos.unit.ene0;
        return; // Partial mass density to Molar abundance
    };
    template < class bdt_T, class eos_T >  __device__
    void save( const bdt_T & bdt, const eos_T & eos ) const
    {
        __syncthreads(  );

        auto & g = x[ 2 ];
        for( int l = threadIdx.x; l < dim; l += blockDim.x )
             if( l != i_en )
                 g[ l ]  = y[ l ] * eos.mcgs( l );
        
        utils::block_reduce
             < f_T, utils::sum >( g, dim - 1 ) ;
        if( __isnan( g[ 0 ] ) || __isinf( g[ 0 ] ) )
            return;
        auto * u_tgt  = bdt.hyd.u.at( idx );
        const  f_T rho( u_tgt[ idx::rho ] );

        if( threadIdx.x ==  0 )
        {
            bdt.T_hyd( idx ) = this->T;
            u_tgt  [ idx::ene ] += -e + __max1
                ( y[ i_en ] * rho_cgs / eos.unit.ene0, 0 );
        }
        u_tgt += eos.n_hyd;
        const auto & n_conv( rho / g[ 0 ] );
        for( int l = threadIdx.x; l < dim; l += blockDim.x )
             if( l != i_en )
                 u_tgt[ l ] = y[ l ] * n_conv
                     * eos.mcgs( l );
        return;
    };    
};

};
