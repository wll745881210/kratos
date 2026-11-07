#pragma once

#include "mg_data.h"

namespace multigrid::data
{
////////////////////////////////////////////////////////////
// Block data holder for multigrid

using namespace mesh;

struct       dual_t ;

struct block_data_t : public mesh::block::base_data_t
{
    ////////// Type and Data //////////
    using super_t = mesh::block::base_data_t;
    data:: base_t                        mgd;
   
    ////////// Memory operations //////////
    __host__ virtual void setup( const f_new_t  & f_n )
    {
        return mgd.setup( f_n, geo );
    };
    __host__ virtual void free ( const f_free_t & f_f )
    {
        return mgd.free( f_f );
    };
    __host__ virtual
    std::shared_ptr< block_data_t > clone(  ) const
    {
        return std::make_shared< block_data_t > (  );
    };

    ////////// Data transfer //////////
    __host__ virtual std::string prefix(  ) const
    {
        return "mg_";
    };
    __host__ virtual void copy_input
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return  mgd.copy_to
        ( f_cp, dynamic_cast< block_data_t & >( tgt ).mgd );
    };
    __host__ virtual void copy_output
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return copy_input( f_cp, tgt );
    };    
    __host__ virtual void read ( const f_read_t  & f_r )
    {
        return mgd.read ( f_r, prefix(  ) );
    };
    __host__ virtual void write( const f_write_t & f_w )
    {
        return mgd.write( f_w, prefix(  ) );
    };

    __host__ virtual void set_lvl( dual_t &, const int & );
};
};                              // namespace multigrid::data
