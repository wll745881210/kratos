#pragma once

#include "../../../utilities/types/crtp.h"
#include "../../../mesh/mesh.h"
#include "../types/dual_data.h"

namespace multigrid::boundary
{
////////////////////////////////////////////////////////////
// Types

using  type::               idx_t;
using  mesh::            dat_3d_t;
using  mesh::          mod_base_t;
using  mesh::          neighbor_t;
using  mesh::boundary::holder_c_t;
using  ::multigrid:: data::dual_t;

////////////////////////////////////////////////////////////
// Kernels

template< class prx_T, class bnd_T >    __global__
void ker_same ( const prx_T prx, const bnd_T bnd )
{
    return bnd.to_same( prx );
}

template< class prx_T, class bnd_T >    __global__
void ker_finer( const prx_T prx, const bnd_T bnd )
{
    return bnd.to_finer( prx );
}

template< class prx_T, class bnd_T >    __global__
void ker_coarser( const prx_T prx, const bnd_T bnd )
{
    return bnd.to_coarser( prx );
}

template< class prx_T, class bnd_T >    __global__
void ker_finer_finest( const prx_T prx, const bnd_T bnd )
{
    return bnd.to_finer_finest( prx );
}

template< class prx_T, class bnd_T > __global__
void ker_from_buf( const prx_T prx, const bnd_T bnd )
{
    return bnd.from_buf( prx );
}

template< class prx_T, class bnd_T > __global__
void ker_from_buf_finest( const prx_T prx, const bnd_T bnd )
{
    return bnd.from_buf_finest( prx );
}

////////////////////////////////////////////////////////////
// Communication boundary

template< class prx_T, class derived_T = crtp::dummy_t >
struct comm_t : mesh::boundary::comm_base_t
{
    ////////// Types and CRTP functions //////////
    __crtp_def_self__( comm_t, derived_T, prx_T );
    using     f_t = type::              float_t;
    using super_t = mesh::boundary::comm_base_t;

    ////////// Data buffer and geometry //////////
    bool             just_restrict;
    int                        lvl;
    int                      n_fld;
    int                      m_dim;
    using       super_t::    order;
    using       super_t:: n_ceff_s;    
    using       super_t:: n_ceff_r;
    using       super_t::same_rank;
    mesh::dat_3d_t< f_t >   b_send;
    mesh::dat_3d_t< f_t >   b_recv;

    ////////// Con-/Destructor and initialization //////////
    template    < class  d_T  >
    __host__ void alloc( d_T  & , const holder_c_t & );
    __host__ virtual void free  ( const holder_c_t & );
    __host__ virtual void set_mem
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    __host__ virtual void set_launch
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    
    ////////// Level-based resource allocation  //////////
    __host__ virtual std::tuple< int, int >
    resource_thbl( const  idx_t & ) const ;
    __host__  std::tuple< dim3, dim3, int >
    resource_pack( const holder_c_t &, const int & );
    __host__  std::tuple< dim3, dim3, int >
    resource_unpk( const holder_c_t &, const int & );
    __host__ virtual    void  adjust_lvl
    ( const        int & lvl, holder_c_t &   h ,
      const neighbor_t &  nb, mod_base_t & mod );

    ////////// Device-side actuator //////////
    __device__ bool dispatch
    ( idx_t &, const idx_t & )   const;
    template < class bdt_T >   __device__
    void  from_buf             ( const bdt_T & ) const;
    template < class bdt_T >   __device__
    void  to_same              ( const bdt_T & ) const;
    template < class bdt_T >   __device__
    void  to_finer             ( const bdt_T & ) const;
    template < class bdt_T >   __device__
    void  to_coarser           ( const bdt_T & ) const;
    template < class bdt_T >   __device__
    void to_finer_finest       ( const bdt_T & ) const;    
    template < class bdt_T >   __device__
    void from_buf_finest       ( const bdt_T & ) const;
    template < class bdt_T >   __device__
    void from_buf_finest_refine( const bdt_T & ) const;
};

};  // namespace multigrid::boundary

////////////////////////////////////////////////////////////
// Implementations of the functions

#include "comm_functions_host.h"
#include "comm_functions_device.h"
#include "comm_functions_finest.h"


