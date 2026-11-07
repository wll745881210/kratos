#pragma once

// Batched-inference QUEUE for the DeepONet solver.
// Host-side lazy registry ( same lifecycle pattern as the
// work_glob workspace and the stiffness counters ):
// allocated on first integrate( ) per block dual, freed in
// solver free( ). Registration kernels append entries with
// atomicAdd on the count; every buffer is indexed by the
// QUEUE SLOT ( the value returned by the atomic ), so
// kernel B reads a contiguous block of rows and the
// cleanup pass visits only registered cells ( never a
// full grid scan, and no stale-status hazard ).
//
// Index-only design ( user ): the queue stores just the
// cell id, the pre-step temperature, and the status.
// Kernel B reads the cell STATE ( u, fluxes ) directly
// from the block's global memory, which is pristine for
// registered cells: the failed DI probe mutates only the
// shared-memory dat_t, and commit paths are the only
// writers of u / T_hyd / fluxes.

#include <map>

#include "../../../src/device/device.h"
#include "../../../src/mesh/block/dual_data.h"

namespace extension::deeponet
{
//////////////////////////////////////////////////////////
// Queue handle. POD-ish: the device pointers are passed
// to kernels by value. Layout, all SLOT-indexed ( slot s
// in [ 0, cand[ 0 ] ) unless noted ):
//   cand  : [ n_ct + 1 ]  cand[ 0 ] = count ( atomicAdd ),
//                            cand[ 1 + s ] = cell id
//   T_old : [ n_ct ]         pre-step temperature of slot s
//                            ( post-prep_dev d.T from the
//                            registering block; NOT re-read
//                            from T_hyd, which is garbage
//                            at t = 0 ).
//   ok    : [ n_ct ]         slot status: 0 pending,
//                            1 committed, 2 guard tripped
//                            ( cleanup pass -> DI ).
// Cell ids are BLOCK-LOCAL: each block dual owns its own
// queue ( registry key ), so no global id is needed.
//////////////////////////////////////////////////////////

struct bat_t
{
    int   * cand;     // count + cell ids
    float * T_old;    // pre-step temperatures ( by slot )
    int   * ok;       // commit status ( by slot )
    int     n_ct;     // queue capacity = cell count
    int     n_in;     // input row width ( diagnostics )

    __host__ bat_t( )
        : cand( nullptr ), T_old( nullptr ),
          ok( nullptr ), n_ct( 0 ), n_in( 0 )
        { return; };

    __host__ bool ready( ) const
    { return cand != nullptr; };
};

__host__ inline std::map< const void *, bat_t > &
bat_registry( )
{
    static std::map< const void *, bat_t > reg;
    return reg;
};

__host__ inline void bat_free
( device::base_t & dev, bat_t & b )
{
    if( b.T_old )
        dev.free_device( b.T_old );
    if( b.cand )
        dev.free_device( b.cand );
    if( b.ok )
        dev.free_device( b.ok );
    b.T_old = nullptr;  b.cand = nullptr;
    b.ok    = nullptr;  b.n_ct  = 0;  b.n_in = 0;
    return;
};

// Lazy allocation: one buffer set per block dual.
template< class mod_T >
__host__ inline bat_t & bat_lookup
( mod_T & mod, mesh::block::dual_t & d,
  const int & n_in )
{
    auto & reg = bat_registry( );
    auto itr = reg.find( & d );
    if( itr != reg.end( ) )
        return itr->second;
    const auto & b( d.b_w.get( ) );
    int n = 1;
    for( int a = 0; a < 3; ++ a )
        n *= b.geo.n_ceff[ a ];
    bat_t bat;
    bat.n_ct = n;
    bat.n_in = n_in;
    // Only cand[ 0 ] ( the count ) is reset per step; every
    // other field is written before it is read in the same
    // step, and stale slots beyond the count are never
    // visited. Zeroed once anyway, for determinism.
    bat.cand = mod.p_dev->template malloc_device< int >
                ( n + 1 );
    bat.T_old = mod.p_dev->template malloc_device< float >
                ( size_t( n ) );
    bat.ok = mod.p_dev->template malloc_device< int >
              ( n );
    mod.p_dev->f_mset( bat.cand, 0,
                       ( n + 1 ) * sizeof( int ), 0 );
    mod.p_dev->f_mset( bat.ok,  0,
                       n * sizeof( int ), 0 );
    mod.p_dev->f_mset( bat.T_old, 0,
                       n * sizeof( float ), 0 );
    reg.emplace( & d, std::move( bat ) );
    return reg.at( & d );
};
};
