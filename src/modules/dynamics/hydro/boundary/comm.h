#pragma once

#include "../../../../types.h"
#include "../../../../utilities/functions.h"
#include "../../../../mesh/mesh.h"
#include "../../../../mesh/boundary/boundary.h"
#include "comm_kernels.h"
#include <tuple>

namespace dynamics::hydro::boundary
{
////////////////////////////////////////////////////////////
// Types

using mesh::            dat_3d_t;
using mesh::block::       dual_t;
using mesh::boundary::holder_c_t;

////////////////////////////////////////////////////////////
// Communication boundary

template< class prx_T, class derived_T = crtp::dummy_t >
struct comm_t : public mesh::boundary  ::  comm_base_t
{
    ////////// Types and CRTP functions //////////
    __crtp_def_self__( comm_t, derived_T, prx_T );
    using f_t = type::float_t;

    ////////// Data buffer and geometry //////////
    int              order;   // Order of spatial accuracy
    dat_3d_t< f_t > b_send;
    dat_3d_t< f_t > b_recv;
    dat_3d_t< f_t > b_fcom;   // Flux buffer

    ////////// Memory utilities //////////
    template< class  d3_T > __host__
    void alloc_one(  d3_T & a0,  const holder_c_t & h );
    __host__ virtual void free ( const holder_c_t & h );

    ////////// Setups //////////
    __host__ virtual void set_geo
    ( holder_c_t &, const neighbor_t &, mod_base_t & );    
    __host__ virtual void set_mem
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    __host__ virtual void set_comm
    ( holder_c_t &, const neighbor_t &, mod_base_t & );    
    __host__ virtual void set_local
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    __host__ virtual void set_launch
    ( holder_c_t &, const neighbor_t &, mod_base_t & );

    ////////// Job dispatching //////////
    __device__       void     dispatch  ( idx_t & i ) const;
    template< class fun_T > __device__ __forceinline__ void
    map_refine( const fun_T & f,  const int & n_dim ) const;

    ////////// Resource //////////
    __host__  std::tuple <  int,  int >
    resource_thbl        (  const idx_t  & n_cell )   const;
    __host__  std::tuple < dim3, dim3, int >  resource_pack
    ( const holder_c_t & h, const neighbor_t  & n )   const;
    __host__  std::tuple < dim3, dim3, int >  resource_unpk
    ( const holder_c_t & h, const neighbor_t  & n )   const;
    __host__  std::tuple < dim3, dim3, int >  resource_flux
    ( const holder_c_t & h, const neighbor_t  & n )   const;

    ////////// Device-side actuator //////////
    template < class bdt_T > __device__ void        to_same
    ( const bdt_T  & b_d, const int & step )          const;
    template < class bdt_T > __device__ void       to_finer
    ( const bdt_T  & b_d, const int & step )          const;
    template < class bdt_T > __device__ void     to_coarser
    ( const bdt_T  & b_d, const int & step )          const;
    template < class bdt_T > __device__ void       from_buf
    ( const bdt_T  & b_d, const int & step )          const;
    template < class bdt_T > __device__ void    flux_to_buf
    ( const bdt_T  & b_d, const int & step )          const;
    template < class bdt_T > __device__ void  flux_from_buf
    ( const bdt_T  & b_d, const int & step )          const;
};

};  // namespace dynamics::hydro::boundary
