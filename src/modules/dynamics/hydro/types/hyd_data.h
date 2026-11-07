#pragma once

#include "../../../../mesh/block/block_data.h"

namespace dynamics::hydro
{
////////////////////////////////////////////////////////////
// Hydrodynamic quantities

using namespace mesh;

struct hyd_t
{
    ////////// Types //////////
    using  f_t = type:: float_t;
    using f2_t = type::float2_t;

    ////////// Data //////////
    dat_3d_t< f2_t > u;
    dat_3d_t<  f_t > w;
    dat_3d_t<  f_t > f[ 3 ]; // fluxes

    ////////// Host-side functions //////////    
    __host__ __device__ __forceinline__ int & order(  )
    {
        return w.n_gh[ 0 ];
    };
    __host__ __device__ __forceinline__
    const int & order(  ) const
    {
        return w.n_gh[ 0 ];
    };

    __host__ __device__ __forceinline__
    const    int & n_fld(  ) const
    {
        return u.n_int;
    };
    __host__ int & n_fld(  )
    {
        return u.n_int;
    };

    __host__ virtual int & n_stp(         ) const;
    __host__ virtual std::string prefix(  ) const;
    __host__ virtual void setup ( const   f_new_t & f_n  ,
                                  const     geo_t & geo );
    __host__ virtual void free  ( const  f_free_t & f_f );
    __host__ virtual void read  ( const  f_read_t & f_r );
    __host__ virtual void write ( const f_write_t & f_w );

    __host__ virtual void copy_to
    ( const f_cp_t & f_c, hyd_t & tgt ) const;    
};

};                            // namespace dynamics::hydro
