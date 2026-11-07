#pragma once

#include "../../../../mesh/types/dat_1d.h"
#include "../../hydro/integrator/data.h"

namespace dynamics::mhd::integrator
{
////////////////////////////////////////////////////////////
// Default data holder for hydrodynamic Godunov integrators

template< class bdt_T, class f_T, class f2_T >
struct data_t : hydro::integrator::data_t
              < bdt_T, f_T, f2_T >
{
    using super_t
        = hydro::integrator::data_t< bdt_T, f_T, f2_T >;
    using super_t::   q;
    using super_t::axis;
    using super_t::step;

    __device__ data_t(  ) = default;
    template < class eos_T > __device__
    data_t( const bdt_T & b_d_, const int & axis,
            const int   & step, void  *   shared,
            const eos_T & eos );

    ////////// Dispatching and Loading //////////
    template< class  idx_T > __device__
    void dispatch( idx_T & i, const bdt_T & b ) const;
    __device__ void load_primative( const bdt_T & b );
};

////////////////////////////////////////////////////////////
// Constructor

template< class bdt_T, class f_T, class f2_T >
template< class eos_T >
__device__ data_t< bdt_T, f_T, f2_T >::data_t
( const bdt_T & b_d, const int & axis_, const int & step_,
  void  *    shared, const eos_T & eos )
{
    this->axis  = axis_;
    this->step  = step_;
    this->init( b_d,  b_d.hyd.w.n_int, shared );
    this->load_geometry ( b_d );
    this->load_primative( b_d );
    return;
}

////////////////////////////////////////////////////////////
// Functions

template < class  bdt_T, class  f_T, class f2_T >
template < class  idx_T  >
__device__ void  data_t< bdt_T, f_T, f2_T >::dispatch
( idx_T & i_3d, const bdt_T & b_d )  const
{
    const auto n_ceff( b_d.geo.n_ceff );
    const int  a_1   ( utils::mod3 ( this->axis + 1 ) );
    const int  a_2   ( utils::mod3 ( this->axis + 2 ) );
    const int  n_cc  ( n_ceff[ a_1 ] *  n_ceff[ a_2 ] );

    i_3d[ axis ] = 0;
    int i_bl ( blockIdx.x );
    if( i_bl < n_cc )
        utils::divmod( i_3d  [ a_2 ], i_3d[ a_1 ] , 
                       i_bl,        n_ceff[ a_1 ] );
    else
    {
        //  Mirror resource(  ): a thin transverse axis
        //  contributes no ghost lines, so its block band
        //  is absent from the grid.
        i_bl -= n_cc ;
        const int n_gr ( 2 * ( n_ceff[ a_1 ] > 1
                             ? n_ceff[ a_2 ] : 0 ) );
        if( i_bl < n_gr )
        {
            const  int s = i_bl  < n_ceff[ a_2 ] ? 0 :  1;
            i_3d [ a_1 ] = s > 0 ? n_ceff[ a_1 ]     : -1;
            i_3d [ a_2 ] = i_bl  - n_ceff[ a_2 ] * s ;
        }
        else
        {
            i_bl        -= n_gr ;
            const  int s = i_bl  < n_ceff[ a_1 ] ? 0 :  1;
            i_3d [ a_2 ] = s > 0 ? n_ceff[ a_2 ]     : -1;
            i_3d [ a_1 ] = i_bl  - n_ceff[ a_1 ] * s ;
        }
    }
    return;
}

template< class  bdt_T, class  f_T, class f2_T >
__device__ void data_t< bdt_T, f_T, f2_T >::load_primative
( const bdt_T & b_d )
{
    idx_t i;
    this->dispatch( i, b_d );
    const int i_th  =     threadIdx.x;
    const int i_bf  = idx::bcc + axis;
    const int n_var = b_d.hyd.w.n_int;
    const int n_tot = this->q.n_spa * n_var;
    //  Ghost-line guard: on a transverse ghost line next
    //  to a cross-level face, the face-centered normal-B
    //  slots are never written ( no common faces across
    //  the level boundary -- mode > 0 holders carry no bf
    //  transfer by design ).  The flux they feed is later
    //  reflux-overwritten, so read the ( always written )
    //  cell-centered value from w instead.
    const auto & geo ( b_d.       geo );
    const int    a_1 ( utils::mod3( axis + 1 ) );
    const int    a_2 ( utils::mod3( axis + 2 ) );
    bool  cross_ghost ( false );
    for( int t = 0; t < 2; ++ t )
    {
        const int a ( t == 0 ? a_1 : a_2 );
        if( i[ a ] < 0 )
            cross_ghost |= geo.refine_flag[ a ][ 0 ]
                        || geo.coarse_flag[ a ][ 0 ];
        else if( i[ a ] >= geo.n_ceff[ a ] )
            cross_ghost |= geo.refine_flag[ a ][ 1 ]
                        || geo.coarse_flag[ a ][ 1 ];
    }
    for(  int j = i_th; j < n_tot; j += blockDim.x )
    {
        int i_s, n;
        utils::divmod( i_s, n, j, n_var );
        i_s      -= q.n_gh;        
        i[ axis ] = i_s + this->blk_shift;
        if( n != i_bf )
            q( i_s, n ) = b_d.hyd.w.at( i )[ n ];
    }
    for( int j = i_th; j < this->n_feff; j += blockDim.x )
    {
        i[ axis ]  = j + this->blk_shift;
        q( j, i_bf ) = cross_ghost
                     ? b_d.hyd.w.at( i )[ idx::bcc + axis ]
                     : b_d.fld.bf.at( i, step )[ axis ];
    }
    return;
}

};              // namespace dynamics::mhd::integrator
