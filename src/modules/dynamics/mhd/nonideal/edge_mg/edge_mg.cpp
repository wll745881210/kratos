#include "edge_mg_iface.h"
#include "edge_mg_driver.h"

#include <cstdio>

#include "../../../../../../src/device/device.h"
#include "../../../../../../src/io/args/input.h"

////////////////////////////////////////////////////////////
//  Edge-centred implicit tensor diffusion:  the module
//  interface.
//
//  This TU is what a pgen links against:  the knobs
//  ( par_t ),  the solver's own state ( solver_t ) and the
//  one-shot smoke diagnostic ( S0' ).  The implicit step
//  itself -- every host action it has,  and the state it
//  keeps between steps -- lives in edge_mg_driver.h/.cpp and
//  is reached through driver_t,  which solver_t owns.
//
//  Method:  docs/mhd/nonideal/algorithm-spec-edge-implicit-
//  diffusion.md.  Engineering constraints ( why this file
//  and not the pgen,  why no template on the block data,
//  why the eta tensor is data ):  docs/mhd/nonideal/
//  edge-mg-foundations.md sections 3.3 and 4.1.
//
//  S0' scope ( foundations section 9 ):  the leg exists,
//  allocates,  is switched by  [nonideal] ee_enable,  and
//  leaves the state BIT-IDENTICAL.  The face-flux
//  divergence printed here is a plumbing check,  NOT a
//  gate:  the gate form ( median,  per-cell eps,  the
//  polar rows excluded ) arrives with A1.

namespace mhd_ee::ni::edge_mg
{

__host__ par_t::par_t(  )
{
    enable     =    0;
    solve      =    0;      // S0':  the diagnostic only,  bit-identical
    subcell    =    0;      // one tensor a cell
    ee_coarsen =    3;      // r and theta,  never phi ( spec 5.6 )
    ee_verb    =    0;
    ee_iter    =  100;
    ee_nlevel  =    8;
    ee_n_smooth =   2;
    bc_edge    =   0;
    ee_writeback = 0;
    ee_dt      =   0.;
    ee_a_bridge =  0;
    ee_tol     = edge_mg_default_ee_tol;
    ee_coarse_iter = 2;
    ee_coarse_tol  = float_t( 1e-10 );
    return;
};

__host__ void par_t::read( const input & args )
{
    enable     = args.get< int >
               ( "nonideal", "ee_enable", 0 );
    solve      = args.get< int >
               ( "nonideal", "ee_solve", 0 );
    subcell    = args.get< int >
               ( "nonideal", "ee_subcell", 0 );
    ee_nlevel  = args.get< int >
               ( "nonideal", "ee_nlevel", 8 );
    ee_n_smooth = args.get< int >
               ( "nonideal", "ee_n_smooth", 2 );
    bc_edge    = args.get< int >
               ( "nonideal", "bc_edge", 0 );
    ee_writeback = args.get< int >
               ( "nonideal", "ee_writeback", 0 );
    ee_dt      = args.get< type::float_t >
               ( "nonideal", "ee_dt", 0. );
    ee_a_bridge = args.get< int >
               ( "nonideal", "ee_a_bridge", 0 );
    if( ee_dt < 0. )
    {
        throw std::runtime_error
        ( "mhd_ee::ni::edge_mg:  ee_dt must be >= 0"
          " ( 0 = the cycle dt )." );
    }
    if( bc_edge < 0 || bc_edge > 2 )
    {
        throw std::runtime_error
        ( "mhd_ee::ni::edge_mg:  bc_edge must be"
          " 0, 1 or 2" );
    }
    if( ee_writeback < 0 || ee_writeback > 1 )
    {
        throw std::runtime_error
        ( "mhd_ee::ni::edge_mg:  ee_writeback must be"
          " 0 ( strong curl ) or 1"
          " ( representation increment )" );
    }
    ee_coarsen = args.get< int >
               ( "nonideal", "ee_coarsen", 3 );
    ee_verb    = args.get< int >
               ( "nonideal", "ee_verb", 0 );
    ee_iter    = args.get< int >
               ( "nonideal", "ee_iter", 100 );
    ee_tol     = args.get< type::float_t >
               ( "nonideal", "ee_tol",
                 edge_mg_default_ee_tol );
    ee_coarse_iter = args.get< int >
               ( "nonideal", "ee_coarse_iter", 2 );
    ee_coarse_tol = args.get< type::float_t >
               ( "nonideal", "ee_coarse_tol", float_t( 1e-10 ) );
    if( ee_coarse_iter < 1 )
    {
        throw std::runtime_error
        ( "mhd_ee::ni::edge_mg:  ee_coarse_iter must be >= 1"
          " ( it caps the nested PCG of the coarsest level )." );
    }
    if( ! ( ee_coarse_tol > float_t( 0. ) ) )
    {
        throw std::runtime_error
        ( "mhd_ee::ni::edge_mg:  ee_coarse_tol must be > 0." );
    }
    if( subcell != 0 && subcell != 1 )
    {
        throw std::runtime_error
        ( "mhd_ee::ni::edge_mg:  ee_subcell must be 0 or 1." );
    }
    return;
};

////////////////////////////////////////////////////////////
//  Driver

__host__ solver_t::solver_t(  )
{
    ready     = false;
    p_dmax    = nullptr;
    div_dmax  = 0;
    last_it   = 0;
    last_nred = 0;
    last_ncomm = 0;
    last_res  = 0.;
    p_dr      = nullptr;
    return;
};

__host__ solver_t::~solver_t(  )
{
    //  The device side is released by finalize( ),  while the
    //  device is still alive;  this only reclaims the host
    //  object ( its own destructor guards the case where
    //  finalize was never called ).
    delete p_dr;
    p_dr = nullptr;
    return;
};

__host__ void solver_t::read( const input & args )
{
    par.read( args );
    return;
};

__host__ void solver_t::setup( device::base_t & dev )
{
    if( ready )
    {
        return;
    }
    p_dmax = dev.malloc< float_t >( 1, false );
    ready  = true;
    return;
};

__host__ void solver_t::step
( device::base_t & dev, const void * stream,
  view_t & v )
{
    if( ! par.enable )
    {
        return;
    }
    setup( dev );
    if( par.solve )
    {
        //  ---- THE IMPLICIT STEP.  The state and the whole
        //  host side of one step live in driver_t;  this
        //  function only owns the solver's slot and the
        //  scalars a gate or a log quotes.
        solve_stat_t ss;
        driver(  ) -> step( dev, stream, v, par, ss );
        last_it   = ss.it;
        last_nred = ss.n_red;
        last_ncomm = ss.n_comm;
        last_res  = ss.res;
        return;
    }
    dev.f_mset( p_dmax, 0, sizeof( float_t ), stream );
    const idx_t n = v.geo.n_ceff;
    dev.launch( ee_div_k,
                std::make_tuple
                ( dim3( n[ 1 ], n[ 2 ], 1 ),
                  dim3( 64 ), 0 ),
                stream, v, p_dmax );
    dev.cp( & div_dmax, p_dmax, 1 );
    if( par.ee_verb > 0 )
    {
        std::printf
        ( "EE0 div|max = %.6e  ( n = %d %d %d )\n",
          float_t( div_dmax ), int( n[ 0 ] ),
          int( n[ 1 ] ), int( n[ 2 ] ) );
    }
    return;
};

__host__ driver_t * solver_t::driver( void )
{
    if( p_dr == nullptr )
    {
        p_dr = new driver_t(  );
    }
    return p_dr;
};

__host__ void solver_t::finalize( device::base_t & dev )
{
    if( p_dmax )
    {
        dev.free_device( p_dmax );
        p_dmax = nullptr;
    }
    if( p_dr != nullptr )
    {
        p_dr->free(  );
        delete p_dr;
        p_dr = nullptr;
    }
    ready = false;
    return;
};

__host__ void solver_t::traffic_read
( int & n_read, int & n_write, long & b_read,
  long & b_write ) const
{
    if( p_dr == nullptr )
    {
        n_read  = 0;
        n_write = 0;
        b_read  = 0;
        b_write = 0;
        return;
    }
    p_dr->traffic( n_read, n_write, b_read, b_write );
    return;
};

__host__ bool solver_t::traffic_on( void ) const
{
    return p_dr == nullptr ? false : p_dr->traffic_on(  );
};

};    // namespace mhd_ee::ni::edge_mg