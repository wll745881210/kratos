#pragma once

#include "../../hydro/boundary/comm.h"

namespace dynamics::mhd::boundary
{
////////////////////////////////////////////////////////////
// Types

using namespace             mesh;
using mesh::boundary::holder_c_t;

////////////////////////////////////////////////////////////
// Communication boundary for MHD

template< class prx_T, class  derived_T = crtp::dummy_t >
struct comm_t : hydro::boundary::comm_t
      < prx_T , crtp ::helper< comm_t, prx_T, derived_T > >
{
    ////////// Types and CRTP functions //////////
    __crtp_def_self__( comm_t, derived_T, prx_T );
    using      f_t   = type ::   float_t;
    using     f2_t   = type ::  float2_t;
    using  super_t   = hydro::boundary::comm_t
    < prx_T , crtp ::helper< comm_t, prx_T, derived_T > >;

    ////////// Data //////////
    idx_t            offset_br; // Offsets for B-fields
    idx_t            offset_bs;
    idx_t            offset_bc;
    idx_t            offset_er; // Offsets for E-fields
    idx_t            offset_es; 
    idx_t            offset_ec;     
    dat_3d_t<  f2_t >   b_bsnd; // Buffers for E-M fields
    dat_3d_t<  f2_t >   b_brcv;
    dat_3d_t<   f_t >   b_ecom; 
    using   super_t::   b_send; // Buffers for hydro
    using   super_t::   b_recv;
    using   super_t::   b_fcom; 
    using   super_t:: offset_s; // Offsets for hydro vars
    using   super_t:: offset_r;
    using   super_t:: offset_c;
    using   super_t::same_rank;
    using   super_t::    d_lvl;
    using   super_t::     axis;
    using   super_t::     mode;
    using   super_t::      rhs;

    ////////// Memory utilities //////////
    template< class d30_T , class d31_T >
    __host__   void alloc_dual
    ( d30_T & a0,   d31_T & a1,  const holder_c_t & h );
    __host__ virtual void free ( const holder_c_t & h );
    
    ////////// Setups //////////
    __host__ virtual void set_geo
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    __host__ virtual void set_mem
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    __host__ virtual void set_local
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    __host__ virtual void set_launch
    ( holder_c_t &, const neighbor_t &, mod_base_t & );
    template< class par_T, class str_T, class evt_T >
    __host__ void set_fpck
    ( holder_c_t & h, const par_T & par,
      const str_T & str_dest, const evt_T & ev_dest );

    ////////// Resource //////////
    using super_t::resource_thbl;
    __host__  std::tuple < dim3, dim3, int > resource_pack
    ( const holder_c_t & h, const neighbor_t & n )   const;
    __host__  std::tuple < dim3, dim3, int > resource_unpk
    ( const holder_c_t & h, const neighbor_t & n )   const;
    __host__  std::tuple < dim3, dim3, int > resource_flux
    ( const holder_c_t & h, const neighbor_t & n )   const;
    __host__  std::tuple < dim3, dim3, int > resource_edge
    ( const holder_c_t & h, const neighbor_t & n )   const;

    ////////// Device-side actuator //////////
    using super_t::  dispatch;
    using super_t::map_refine;
    template < class bdt_T > __device__ void       to_same
    ( const bdt_T  & b_d,  const  int & step )       const;
    template < class bdt_T > __device__ void      to_finer
    ( const bdt_T  & b_d,  const  int & step )       const;
    template < class bdt_T > __device__ void    to_coarser
    ( const bdt_T  & b_d,  const  int & step )       const;
    template < class bdt_T > __device__ void      from_buf
    ( const bdt_T  & b_d,  const  int & step )       const;
    template < class bdt_T > __device__ void   flux_to_buf
    ( const bdt_T  & b_d,  const  int & step )       const;
    template < class bdt_T > __device__ void flux_from_buf
    ( const bdt_T  & b_d,  const  int & step )       const;
    template < class bdt_T > __device__ void   edge_efield
    ( const bdt_T  & b_d,  const  int & step )       const;
};
};                    // namespace dynamics::mhd::boundary
