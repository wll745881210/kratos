#pragma once

#include "../../../mesh/block/block_data.h"
#include "../../../utilities/mapping/loop.h"

namespace particle::cic
{
////////////////////////////////////////////////////////////
// Block data holder for hydrodynamics

using namespace mesh;

struct cic_t
{
    ////////// Data //////////
    mesh::dat_3d_t< type::float_t > den;
    mesh::dat_3d_t< type::float_t > acc;    
    
    ////////// Dual-side functions //////////
    __host__ __device__ __forceinline__ int & order(  )
    {
        return den.n_gh[ 0 ];
    };    
    __host__ __device__ __forceinline__
    const    int & n_fld(  ) const
    {
        return den.n_int;
    };
    __host__ int & n_fld(  )
    {
        return den.n_int;
    };

    ////////// Host-side functions //////////
    __host__ std::string prefix(  ) const
    {
        return "cic_";
    };
    __host__ void setup
    ( const mesh::f_new_t & f_n, const mesh::geo_t & geo )
    {
        auto n_gh( type::idx_t::null(  ) );
        const auto order = this->order(  );
        for( int a = 0; a < n_gh.size(  ); ++ a )
            n_gh[ a ] = ( geo.n_ceff[ a ] > 1 ? order : 0 );
        den.init( f_n, geo.n_ceff, n_gh,     n_fld(  ) );
        acc.init( f_n, geo.n_ceff, n_gh, 3 * n_fld(  ) );
        return;
    };
    __host__ void free( const mesh::f_free_t & f_free )
    {
        acc       .free( f_free );
        return den.free( f_free );
    };
    __host__ void copy_to
    ( const mesh::f_cp_t & f_cp, cic_t & tgt ) const
    {
        acc       .cp_to( tgt.acc, f_cp );
        return den.cp_to( tgt.den, f_cp );
    };
    __host__ void read ( const mesh:: f_read_t & f_r )
    {
        acc       .read ( f_r, prefix(  ) + "acc_" );
        return den.read ( f_r, prefix(  ) + "den_" );
    };
    __host__ void write( const mesh::f_write_t & f_w )
    {
        acc       .write( f_w, prefix(  ) + "acc_" );
        return den.write( f_w, prefix(  ) + "den_" );
    };
};

struct block_data_t : mesh::block::base_data_t
{
    ////////// Tynpe and Data //////////
    cic_t cic;

    ////////// Memory operations //////////
    __host__ virtual void setup( const f_new_t  & f_n )
    {
        return cic.setup( f_n, geo );
    };
    __host__ virtual void free ( const f_free_t & f_f )
    {
        return cic.free( f_f );
    };

    ////////// Data transfer //////////
    __host__ virtual void copy_input
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return cic.copy_to
        ( f_cp, static_cast< block_data_t & >( tgt ). cic );
    };
    __host__ virtual void copy_output
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return copy_input( f_cp, tgt );
    };
    __host__ virtual void read ( const f_read_t  & f_r )
    {
        return cic.read( f_r );
    };
    __host__ virtual void write( const f_write_t & f_w )
    {
        return cic.write( f_w );
    };
};

};                        // namespace particle::cic
