#include "solver.h"
#include "../chemistry.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <utility>

namespace chemistry::solver
{
////////////////////////////////////////////////////////////
// Global workspace -- Plan B for the shared memory. 

void work_glob_t::setup
( const mesh::region_t & reg, const mesh::geo_t & geo,
  const int & s_sh )
{
    if( bmap.find( reg ) == bmap.end(  ) )
        bmap[ reg ].init( p_dev_w->f_malloc, geo.n_ceff, 0,
                          utils::align( s_sh, 8 ) );
    return;
}

void work_glob_t::free(  )
{
    for( auto & [ key, work ] : bmap )
         work.free( p_dev_w-> f_free );
    return;
}

////////////////////////////////////////////////////////////
// Initialization of the solver module

__host__ void base_t::init
( const input & args, ::chemistry::base_t & mod )
{
    dim        = mod.species.size (  ) + 1;
    iter_max   = args.get   < int >
               ( "chemistry", "iter_max",   255 );
    dbg_iter   = args.get   < int >
               ( "chemistry", "dbg_iter",     0 );
    atol       = args.get   < type ::   float_t >
               ( "chemistry", "atol",     1e-4f );
    rtol       = args.get   < type ::   float_t >
               ( "chemistry", "rtol",     1e-4f );
    ytol       = args.get   < type ::   float_t >
               ( "chemistry", "ytol",     1e-4f );
    etol_ratio = args.get   < type ::   float_t >
               ( "chemistry", "etol",      rtol ) / rtol;
    n_th       = args.get   < int >
               ( "chemistry", "n_th_base",   16 );
    n_th       = ( dim + n_th - 1 ) / n_th * n_th;
    cache_rate = args.get< bool >
               ( "chemistry", "cache_rate", true );

    p_work_glob = nullptr;
    if( args.get< bool >
        ( "chemistry", "work_glob", false ) )
    {
        p_work_glob  = new work_glob_t;
        p_work_glob->p_dev_w = mod.p_dev;
    }
    return;
}

////////////////////////////////////////////////////////////
// Global-memory workspaces ( work_glob )

__host__ void base_t::free( device::base_t & dev )
{
    if( p_work_glob != nullptr )
    {
        p_work_glob->free( );
        delete p_work_glob;
    }
    return;
};

////////////////////////////////////////////////////////////
// Launch configuration

__host__ std::tuple< dim3, dim3, int > base_t::resource
( const mesh::block::dual_t & d, const int & size_sh ) const
{
    const auto  & b( d.b_w.get(   ) );
    const auto n_ceff = b.geo.n_ceff;
    dim3  n_bl( n_ceff[ 0 ], n_ceff[ 1 ], n_ceff[ 2 ] );
    const int s_sh( size_sh + this->size_sh(  ) );
    if( p_work_glob != nullptr )
        p_work_glob->setup( b. reg, b. geo, s_sh );
    return std::make_tuple( n_bl, n_th, p_work_glob ==
                            nullptr ? s_sh : 0 );
}
};                           // namespace chemistry::solver
