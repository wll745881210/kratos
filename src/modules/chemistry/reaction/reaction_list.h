#pragma once

#include "../../../utilities/types/var_list.h"
#include "../../../utilities/types/crtp.h"
#include "../../../utilities/phys/constants.h"
#include "../../../mesh/mesh.h"
#include "../types.h"
#include "reaction.h"
#include <fstream>

namespace chemistry::reaction::list
{
////////////////////////////////////////////////////////////
// Lists for reactions

template< class rec_T,  class derived_T >
struct  std_t : public ::chemistry::reaction::list::base_t
{
    ////////// Type //////////
    __crtp_def_self__( std_t, derived_T, rec_T );

    ////////// Data //////////
    utils::const_list_t< rec_T > reac;

    ////////// Functions //////////
    template< class itg_T, class bdt_T, class eos_T >
    __host__ __forceinline__ void prep_host
    ( const itg_T & , const bdt_T &,
      const eos_T & ) const {  };
    
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void prep_dev
    ( dat_T &, const bdt_T &, const eos_T & ) const {  };
    
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void jac_single
    ( dat_T       &   d, const int   &   i ,
      const bdt_T & bdt, const eos_T & eos ) const
    {
        return reac[ i ].jac( d, bdt, eos, * this );
    };
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void rate_single
    ( dat_T       &   d, const int   &   i ,
      const bdt_T & bdt, const eos_T & eos ) const
    {
        reac[ i ].rate( d, bdt, eos, * this );
    };
    __device__ __forceinline__ int size(  ) const
    {
        return reac.size;
    };

    template< class dat_T, class bdt_T , class eos_T >
    __device__ void jac  ( dat_T &   d , const bdt_T & bdt,
                           const eos_T & eos ) const
    {
        __syncthreads(  );        
        const  auto & self( get_self(  ) );
        d.jf_memset_mt(   );    // With __syncthreads(  );
        eos.temp      ( d );    // With __syncthreads(  );
        const int i_th( threadIdx.x );
        const int n_th(  blockDim.x );
        for(  int l = i_th; l < self.size(  ); l += n_th )
            self.jac_single( d, l, bdt, eos );
        d.f_reduce    (  );        
        __syncthreads (  );
        if( d.i_en >= 0  )     // Heat-capacity related
        {
            for( int i = i_th ; i < d.dim; i += blockDim.x )
                for( int j = 0; j < d.dim; ++ j )
                {
                    if( j == d.i_en )
                        continue;
                    d.J( i, j ) -= eos.deps_dn( d, j )
                         * phys::cgs::NA * d.J( i, d.i_en );
                }
            __syncthreads(  );
        }
        return;
    };
    template< class dat_T, class bdt_T,  class eos_T >
    __device__ void rate ( dat_T &   d,  const bdt_T & bdt,
                           const eos_T & eos ) const
    {
        __syncthreads(  );
        const auto & self( get_self(  ) );
        d  .f_memset_mt(   );     // With __syncthreads(  );
        eos.temp       ( d );     // With __syncthreads(  );
        const int i_th ( threadIdx.x );
        const int n_th (  blockDim.x );
        for( int l = i_th; l < self.size(  ); l += n_th )
            self.rate_single( d, l, bdt, eos );
        d.f_reduce   (  );
        __syncthreads(  );
        return;
    };

    ////////// Host-side functions //////////
    __host__ virtual void print
    ( const ::chemistry::sp_vec_t & sps )
    {
        std::cout << "=======================\n# reacions: "
                  << reac.size << '\n';
        for( size_t i = 0; i < reac.size; ++ i )
            reac[ i ].print( sps );
        std::cout << "=======================" << std::endl;
        return;
    };

    __host__ virtual void init
    ( const input & args, ::chemistry::base_t & chem )
    {
        input rarg( args.get< std::string >
                  ( "chemistry", "reaction_file", "" ) );
        rarg.set_comm( chem.p_mesh->p_com );
        if( ! rarg.read(  ) )
        {
            if( chem.p_mesh->p_com->is_root (  ) )
                std::cout << "Skipping std reactions\n";
            return;
        }            
        std::vector < rec_T > reac_h;
        for( const auto & r : rarg.get_prefixes(  ) )
            if( ! reac_h.emplace_back(  ).parse
                ( r, rarg, chem ) )
                reac_h.pop_back(  );
        reac.init( reac_h.size (  ) );
        for( size_t i = 0; i < reac_h.size(  ); ++ i )
            reac[ i ] = reac_h.at ( i );
        if( args.get< bool >( "chemistry", "print", true ) )
            if( chem.p_mesh->p_com->is_root(  ) )
                print  (   chem.species );

        const auto on_const = args.get< bool >
            ( "chemistry", "reac_on_const",  true );
        reac.regularize( * chem.  p_dev, on_const );
        init_extra     (   args,             chem );
        return;
    };

    __host__ virtual void free( device::base_t & f_dev )
    {
        return reac.free( f_dev );
    };    
};
};                 // namespace chemistry::reaction::list
