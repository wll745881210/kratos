#pragma once

#include "../../../../mesh/block/block_proxy.h"
#include "hyd_data.h"

namespace dynamics::hydro
{
////////////////////////////////////////////////////////////
// Block data holder for hydrodynamics

using namespace mesh;

struct block_data_t : mesh::block::base_data_t
{
    ////////// Type and Data //////////    
    hyd_t      hyd;

    ////////// Memory operations //////////
    __host__ virtual void setup( const f_new_t  & f_n )
    {
        return hyd.setup( f_n, geo );
    };
    __host__ virtual void free ( const f_free_t & f_f )
    {
        return hyd.free( f_f );
    };
    
    ////////// Data transfer //////////
    __host__ virtual void copy_input
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return hyd.copy_to
        ( f_cp, static_cast< block_data_t & >( tgt ). hyd );
    };
    __host__ virtual void copy_output
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return copy_input( f_cp, tgt );
    };    
    __host__ virtual void read ( const f_read_t  & f_r )
    {
        return hyd.read( f_r );
    };
    __host__ virtual void write( const f_write_t & f_w )
    {
        return hyd.write( f_w );
    };
};

using proxy_t = mesh::block::proxy_t< block_data_t >;

};                            // namespace dynamics::hydro
