#pragma once

#include "../../hydro/types/block_data.h"
#include "fld_data.h"

namespace dynamics::mhd
{
////////////////////////////////////////////////////////////
// Block data holder for hydrodynamics

using namespace mesh;

struct block_data_t : ::dynamics::hydro::block_data_t
{
    ////////// Data and Types //////////
    using   super_t = ::dynamics::hydro::block_data_t;
    fld_t       fld ;

    ////////// Memory operations //////////
    __host__ virtual void setup( const  f_new_t & f_n )
    {
        fld.setup( f_n, geo );
        super_t::setup( f_n );
    };
    __host__ virtual void free ( const f_free_t & f_f )
    {
        super_t  ::free( f_f );
        return fld.free( f_f );
    };

    ////////// Data transfer //////////
    __host__ virtual void copy_input
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        super_t::copy_input( f_cp, tgt );
        return fld. copy_to( f_cp, static_cast
                           < block_data_t & >( tgt ). fld );
    };
    __host__ virtual void copy_output
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return copy_input( f_cp, tgt );
    };
    __host__ virtual void read ( const f_read_t  & f_r )
    {
        super_t  ::read( f_r );
        return fld.read( f_r );
    };
    __host__ virtual void write( const f_write_t & f_w )
    {
        super_t  ::write( f_w );
        return fld.write( f_w );
    };
};

using proxy_t = mesh::block::proxy_t< block_data_t >;

};                              // namespace dynamics::mhd
