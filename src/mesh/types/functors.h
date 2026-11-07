#pragma once

//  This header uses std::function,  so it must say so:  a TU
//  whose first include lands here ( e.g.  edge_mg.cpp via
//  dat_3d.h ) otherwise fails under nvcc,  which does not
//  get <functional> from the HIP-CPU runtime header that
//  the g++ path force-includes.
#include <functional>

namespace mesh
{
using f_new_t  = std::function< void * ( const size_t & ) >;
using f_free_t = std::function< void   ( void         * ) >;
using f_null_t = std::function
    < void( void *,                      const size_t & ) >;
using f_cp_t   = std::function
    < void( void *, const       void *,  const size_t & ) >;

using f_read_t  = std::function
    < int ( void *,              const   std::string  & ) >;
using f_write_t = std::function
    < int ( void *, const size_t &,      const size_t & ,
                                 const   std::string  & ) >;
};                              // namespace mesh
