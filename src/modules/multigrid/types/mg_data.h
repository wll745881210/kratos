#pragma once

#include "../../../mesh/mesh.h"

namespace multigrid::data
{
////////////////////////////////////////////////////////////
// Data type

using namespace mesh;

struct base_t
{
    ////////// Type //////////
    using f_t = type::float_t;

    ////////// Data //////////
    int                d_lvl;
    dat_3d_t< f_t >      fld;
    dat_3d_t< f_t >  src_dat;

    ////////// Device-side interfaces (default) //////////
    __host__ __device__ __forceinline__ int & order(  )
    {
        return fld.n_gh[ 0 ];
    };
    __host__ __device__ __forceinline__
    const int & order(  ) const
    {
        return fld.n_gh[ 0 ];
    };
    __host__ __device__ const int & n_fld(  ) const
    {
        return fld.n_int;
    };
    __host__                  int & n_fld(  )
    {
        return fld.n_int;
    };

    template  < class idx_T >
    __host__ __device__ __forceinline__
    auto & src( const idx_T & i, const int & n = 0 ) const
    {
        return src_dat.at ( i ) [ n ];
    };

    ////////// Host-side functions  //////////

    __host__ virtual void setup ( const   f_new_t & f_n  ,
                                  const     geo_t & geo );
    __host__ virtual void free  ( const  f_free_t & f_f );
    __host__ virtual void read
    ( const  f_read_t & f_r, const std::string & prefix );
    __host__ virtual void write
    ( const f_write_t & f_w, const std::string & prefix );
    __host__ virtual void null  ( const  f_null_t & f_n );

    __host__ virtual void copy_to
    ( const f_cp_t & f_c, base_t & tgt ) const;
};
};                            // namespace multigrid::data
