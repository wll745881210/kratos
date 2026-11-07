#pragma once

#include "../../../mesh/mesh.h"
#include "../../../io/args/input.h"
#include "../../../utilities/types/crtp.h"
#include "../../../utilities/mapping/mapping.h"
#include <utility>

namespace multigrid::operation
{
////////////////////////////////////////////////////////////
// Base class of operator

using mesh::mod_base_t;

struct base_t
{
    enum phy_bnd_t { fixed, free } ;
    phy_bnd_t    phy_bnd;
    base_t(  ) : phy_bnd(  fixed ) {  };
    
    __host__ virtual void init    ( const input & ) {  };
    __host__ virtual void update  ( mod_base_t  & ) {  };
    __host__ virtual void finalize( mod_base_t  & ) {  };
};

////////////////////////////////////////////////////////////
// Default operator

template< class derived_T = crtp::dummy_t >
struct operation_t : public base_t
{
    ////////// Type //////////
    __crtp_def_self__( operation_t,  derived_T );

    ////////// Data //////////
    int                    flag;
    int                   n_fld;
    type::float_t *    defect_d;
    type::float_t *    defect_h;
    type::float_t * field_max_d;
    type::float_t * field_max_h;    
    type::float_t *   src_ave_c;
    type::float_t *   src_ave_d;
    type::float_t *   src_ave_h;

    ////////// Initialization //////////
    __host__ virtual void finalize( mod_base_t  & );

    ////////// For the Seeliger Paradox //////////
    __host__ void              setup_mem( mod_base_t & );
    __host__ type::float_t   stat_defect( mod_base_t & );
    __host__ void           stat_src_ave( mod_base_t & );

    ////////// Resource //////////
    template< class bdt_T > __host__
    auto    rsc_refine( const bdt_T & bdt, int n_th ) const;
    template< class bdt_T > __host__
    auto    rsc_coarse( const bdt_T & bdt, int n_th ) const;
    template< class bdt_T > __host__
    auto    rsc_smooth( const bdt_T & bdt, int n_th ) const;
    template< class bdt_T > __host__ __device__
    int    n_sh_smooth( const bdt_T & bdt           ) const;
    template< class bdt_T > __host__ __device__
    int    n_sh_refine( const bdt_T & bdt           ) const;
    template< class bdt_T > __host__ __device__
    int    n_sh_coarse( const bdt_T & bdt           ) const;

    ////////// Dispatcher //////////
    template   < class idx_T , class bdt_T >
    __device__ __forceinline__ bool  dispatch_lvlchg
    ( idx_T & i, const bdt_T & bdt                  ) const;
    template   < class idx_T , class bdt_T >
    __device__ __forceinline__ bool  dispatch_smooth
    ( idx_T & i, const bdt_T & bdt, const int & stg ) const;
    template   < class idx_T >
    __device__ __forceinline__ bool  dispatch
    ( idx_T & i, const idx_T & n_c, const int & i_t ) const;
    template   < class idx_T >
    __device__ __forceinline__ bool  muldim
    ( idx_T & j, const int   & i_t, const int & n_d ) const;

    ////////// Algorithms //////////
    template< class   f_T, class dat_T ,
              class bdt_T, class idx_T > __device__    void
    weight( f_T   & w_c, f_T w [  ][ 2 ], const dat_T & dat,
            const bdt_T  & bdt,   const idx_T & i )   const;

    template< class bdt_T > __device__ __forceinline__ void
    refine  ( const bdt_T & b_c, const bdt_T & b_f )  const;
    template< class bdt_T > __device__ __forceinline__ void
    refine_s( const bdt_T & b_c, const bdt_T & b_f )  const;
    template< class bdt_T > __device__ __forceinline__ void
    coarsen ( const bdt_T & b_c, const bdt_T & b_f )  const;
    template< class bdt_T > __device__ __forceinline__ void
    smooth  ( const bdt_T & bdt, const  rb_t & stg )  const;
    template< class bdt_T > __device__ __forceinline__ void
    residual( const bdt_T & bdt                    )  const;
    template< class bdt_T > __device__ __forceinline__ void
    calc_ave( const bdt_T & bdt                    )  const;
    template< class bdt_T > __device__ __forceinline__ void
    defect  ( const bdt_T & bdt                    )  const;
};

};                         // namespace multigrid::operation
