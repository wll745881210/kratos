#pragma once

#include "../../../mesh/block/dual_data.h"
#include "block_data.h"
#include <vector>

namespace multigrid::data
{
////////////////////////////////////////////////////////////
// Block type

struct dual_t : public mesh::block::dual_t
{
    ////////// Initialize and finalize //////////
    using   mesh::block::dual_t::dual_t;
    virtual void free (  );
    virtual void setup(  );

    ////////// Multiple-level block data //////////
    std::vector< std::shared_ptr< block_data_t >  > p_d_arr;
    int                                               lvl_0;
    virtual int                           n_lvl(  )   const;
    virtual block_data_t & d_mg( const int    &   )   const;

/*
    virtual void write( binary_io::base_t & bio )
    {
        auto fw = [ & ]( void * p, const size_t & c, const
                         size_t & u, const std::string & t )
        {    return bio.write( p, c, u, prefix(  ) + t ); };

        auto & bdt = dynamic_cast< block_data_t & >( *p_h );
        for( size_t i = 1; i < p_d_arr.size(  ); ++ i )
        {
            auto & mgd( p_d_arr[ i ]->mgd );
            mgd.write
                ( fw, "mg_" + std::to_string( i ) + "_" );
        }
        return bdt.mgd.write( fw, "mg_" );
    };
*/
};                              // struct      block_t
};                              // namespace multigrid
