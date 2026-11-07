#pragma once

#include "../../hydro/boundary/keeper.h"
#include "comm_functions.h"
#include "phys_functors.h"
#include "comm.h"

namespace dynamics::mhd::boundary
{
////////////////////////////////////////////////////////////
// One-shot ghost-shell fill: writes val into every slot of
// arr whose cell lies outside the interior ( any axis
// storage index < n_gh or >= n_cell - n_gh ), for all
// field components and all steps. Interior is init_cond's
// and is never touched. See keeper_t::update for use.

template< class d3d_T > __global__ void
ker_ghost_fill( const d3d_T arr,
                const typename d3d_T::f_t val )
{
    const long long n_ctot = arr.n_ctot;
    const long long id = blockIdx.x
                       * ( long long ) blockDim.x
                       + threadIdx.x;
    if( id >= n_ctot )
        return;
    const long long cell = id / arr.n_int;
    const long long n01  = ( long long ) arr.n_cell[ 0 ]
                         * arr.n_cell[ 1 ];
    const int s2 = ( int ) ( cell / n01 );
    const int rem= ( int ) ( cell % n01 );
    const int s1 = rem / arr.n_cell[ 0 ];
    const int s0 = rem % arr.n_cell[ 0 ];
    bool shell = false;
    for( int a = 0; a < 3; ++ a )
    {
        const int s ( a == 0 ? s0 : ( a == 1 ? s1 : s2 ) );
        shell |= ( s < arr.n_gh[ a ] )
              || ( s >= arr.n_cell[ a ] - arr.n_gh[ a ] );
    }
    if( ! shell )
        return;
    for( int s = 0; s < arr.n_step; ++ s )
        arr.dat[ s * n_ctot + id ] = val;
    return;
};

////////////////////////////////////////////////////////////
// Boundary condition keeper

//  Thin - axis face sync  ( see integrator.h ).   Lives
//  here so it runs AFTER the physical - boundary ghost
//  fill and BEFORE the EMF sweep:  a periodic axis with a
//  single cell has no neighbour,  so its wrap slots must
//  BE the slot - 0 face --  left at the zeroed ghost the
//  corner EMFs read a phantom face and the CT stops
//  telescoping  ( the  n_phi = 1  south - pole leak,
//  2026-10-21 ).
template< class bdt_T >
__global__ void thin_axis_sync_k( const bdt_T b_d )
{
    const auto & geo( b_d.geo );
    const int t( blockIdx.x * blockDim.x + threadIdx.x );
    for( int a = 0; a < 3; ++ a )
    {
        if( geo.n_ceff[ a ] != 1 )
            continue;
        if(    geo.phybnd_flag[ a ][ 0 ]
            || geo.phybnd_flag[ a ][ 1 ] )
            continue;
        const int a1( utils::mod3( a + 1 ) );
        const int a2( utils::mod3( a + 2 ) );
        const int g ( geo.n_gh );
        const int N1( geo.n_ceff[ a1 ] + 2 * g );
        const int N2( geo.n_ceff[ a2 ] + 2 * g );
        if( t >= N1 * N2 )
            continue;
        //  Interior AND the ghost band of the other two
        //  axes  ( the corner EMFs read the pole - adjacent
        //  ghost rows ).
        type::idx_t i;
        i[ a  ] = 0;
        i[ a1 ] =   t % N1 - g;
        i[ a2 ] =   t / N1 - g;
        auto * d0( b_d.fld.bf.at( i ) );
        //  Both the upper and the lower wrap slots.
        for( int k = -g; k <= g; ++ k )
        {
            if( k == 0 )
                continue;
            type::idx_t j( i );
            j[ a ] = k;
            auto * d1( b_d.fld.bf.at( j ) );
            d1[ 0 ] = d0[ 0 ];
            d1[ 1 ] = d0[ 1 ];
            d1[ 2 ] = d0[ 2 ];
        }
    }
    return;
};

class keeper_t : public hydro::boundary::keeper_t
{
    ////////// Initialization //////////
public:                         // Functions
    keeper_t(  )
    {
        enroll_phys< reflect_t<  > > ( "ref" );
        enroll_phys< outflow_t<  > > ( "out" );
        enroll_phys<    free_t<  > > ( "fre" );
        enroll_phys<    pole_t<  > > ( "pol" );
        enroll_phys<    axis_t<  > > ( "axs" );
        p_fixd = enroll_phys< fixed_t<  > > ( "fixd" );
        return;
    };
    //  ghost_init: adjustable fill value for the w / bf
    //  ghost shells at setup ( default 0 = deterministic ).
    //  An insane value ( e.g. 1e30 at PREC=1 ) turns
    //  never-computed ghost slots into a falsifiable test:
    //  if the physics is invariant under the fill value,
    //  no uncomputed slot is ever read; any read trips
    //  except_ferr when enabled.
    virtual void init( const input & args, mesh_t & mesh )
    {
        ghost_init = args.get< double >
                     ( "boundary", "ghost_init", 0.0 );
        //  fixd: Dirichlet ghost state = the ambient
        //  medium.  The boundary functors operate on the
        //  PRIMITIVE w,  so the fix is simply
        //  ( rho0, pre0, vel0, b0 ).
        {
            const auto rho0 = args.get< type::float2_t >
                            ( "init", "rho0", 0 );
            const auto pre0 = args.get< type::float2_t >
                            ( "init", "pre0", 0 );
            const auto vel0 = args.get< type::coord2_t >
                            ( "init", "vel0", 0 );
            const auto   b0 = args.get< type::coord2_t >
                            ( "init",  "b0", 0 );
            p_fixd->w_fix[ idx::rho ] = rho0;
            p_fixd->w_fix[ idx::ene ] = pre0;
            for( int n = 0; n < 3; ++ n )
            {
                p_fixd->w_fix[ idx::mom + n ] = vel0[ n ];
                p_fixd->b_fix[ n ]           =   b0[ n ];
            }
        }
        hydro::boundary::keeper_t::init( args, mesh );
        return;
    };

    ////////// Actions //////////
protected:                      // Data
    //  One-shot flag: ee/ec zero-init done.
    bool   f_zeroed   { false };
    //  Fill value for the w / bf ghost shells.
    double ghost_init {   0.0 };
    //  Dirichlet ( "fixd" ) boundary functor; its ambient
    //  state is filled from [init] in init.
    std::shared_ptr< fixed_t<  > > p_fixd;
protected:                      // Functions
    //  The out/fre side mask into the block geometry:  the
    //  enrolled physical functors decide  ( dynamic_cast ),
    //  assign_phys runs per block before any step.   This
    //  drives the boundary - plane advective edge EMF  (
    //  calc_field;  the r_out field - anchoring fix ).
    virtual void assign_phys( block_t & b, mesh_t & mesh )
    {
        hydro::boundary::keeper_t::assign_phys( b, mesh );
        auto & mod( * p_mod );
        auto & geo
            ( mod.p_data( b.reg )->d(  )
                   .derive< block_data_t >(  ).geo );
        for( int a = 0; a < 3; ++ a )
            for( int s = 0; s < 2; ++ s )
                geo.bnd_out[ a ][ s ] = false;
        for( auto & h : bnd_phys[ b.reg ] )
        {
            const int a( h.side / 2 );
            const int s( h.side % 2 );
            const bool adv
            (    dynamic_cast
                 < const outflow_t<  > * >( h.p.get(  ) )
             || dynamic_cast
                 < const free_t<  > * >( h.p.get(  ) ) );
            geo.bnd_out[ a ][ s ] = adv;
        }
        return;
    };
    virtual void update( mesh_t & mesh )
    {
        hydro::boundary::keeper_t::update( mesh );
        //  ee / ec edge-corner E and the flux arrays are
        //  read by calc_field before the first exchange on
        //  refined interfaces ( and a slot may have no
        //  writer at all ); stale heap debris there drove
        //  the SMR annihilation.  Zero them once at setup.
        //  w is NOT zeroed: cons2prim owns it, and the
        //  chemistry solver may read species slots from w
        //  before the first cons2prim pass.
        if( ! f_zeroed && bnd_comm.size(  ) > 0 )
        {
            f_zeroed = true;
            for( const auto & [ reg, h_map ] : bnd_comm )
                for( const auto & [ nb, h ] : h_map )
                {
                    auto & bd ( h.p_d->d(  )
                        .derive< block_data_t >(  ) );
                    auto & fld ( bd.fld );
                    p_dev->f_mset
                        ( fld.ee.dat, 0, fld.ee.n_size,
                          nullptr );
                    p_dev->f_mset
                        ( fld.ec.dat, 0, fld.ec.n_size,
                          nullptr );
                    for( int a = 0; a < 3; ++ a )
                        p_dev->f_mset( bd.hyd.f[ a ].dat, 0,
                                       bd.hyd.f[ a ].n_size,
                                       nullptr );
                    //  Non-ideal arrays ( e_dif / j_edge /
                    //  heat ),  when allocated:  same
                    //  reasoning as ee / ec above --  the
                    //  diffusion kernels read edge and
                    //  corner slots that may have no writer
                    //  before the first exchange on refined
                    //  interfaces.
                    if( fld_t::need_ni_ee )
                    {
                        //  The edge 1-form and the eta tensor
                        //  of the implicit tensor-diffusion
                        //  leg:  zero once at setup,  for the
                        //  same reason as ee / e_dif above.
                        p_dev->f_mset( fld.a_edge.dat, 0,
                                       fld.a_edge.n_size,
                                       nullptr );
                        p_dev->f_mset( fld.eta_cc.dat, 0,
                                       fld.eta_cc.n_size,
                                       nullptr );
                    }
                    if( fld_t::need_ni )
                    {
                        p_dev->f_mset( fld.e_dif .dat, 0,
                                       fld.e_dif .n_size,
                                       nullptr );
                        p_dev->f_mset( fld.j_edge.dat, 0,
                                       fld.j_edge.n_size,
                                       nullptr );
                        p_dev->f_mset( fld.heat  .dat, 0,
                                       fld.heat  .n_size,
                                       nullptr );
                    }
                    //  w / bf ghost-shell fill: no correct
                    //  run reads them before the exchange
                    //  writes them, so the physics must be
                    //  invariant under this value.
                    const auto fill_ghost =
                    [ & ] ( const auto & arr )
                    {
                        if( ! arr )
                            return;
                        const unsigned n_bl
                        ( ( arr.n_ctot + 255 ) / 256 );
                        p_dev->launch
                        ( ker_ghost_fill< std::decay_t
                          < decltype( arr ) > >,
                          dim3( n_bl ), dim3( 256 ), 0,
                          nullptr, arr,
                          ( typename std::decay_t
                            < decltype( arr ) > :: f_t )
                          ghost_init );
                        return;
                    };
                    fill_ghost( bd.hyd.w   );
                    fill_ghost( bd.fld.bf  );
                }
            p_dev->sync_all_streams(  );
        }
        return;
    };
    virtual bool desired_mode( const neighbor_t & nb )
    {
        //  MHD's CT / E machinery reads edge and corner
        //  ghost cells, so mode 1 and mode 2 holders must
        //  be enrolled ( hydro enrolls faces only ); the
        //  offset_c-aware band math keeps mode 2 safe.
        return ( nb.mode < 3 );
    };
public:                         // Functions
    virtual void act_bnd( mesh_t & mesh, const int & step )
    {
        hydro::boundary::keeper_t::act_bnd( mesh, step );
        for( auto & [ reg , v_pb ] : bnd_phys )
            for( auto & h : v_pb )
                h.launch( h,  -1 ) ;
        //  Then re - assert the thin - axis wrap slots  (
        //  the ghost fill above zeroes them ).
        for( auto & b : mesh )
        {
            auto & bd( p_mod->p_data( b.reg )->d(  )
                       .derive< block_data_t >(  ) );
            int n_t( 0 );
            for( int a = 0; a < 3; ++ a )
            {
                if( bd.geo.n_ceff[ a ] != 1 )
                    continue;
                if(    bd.geo.phybnd_flag[ a ][ 0 ]
                    || bd.geo.phybnd_flag[ a ][ 1 ] )
                    continue;
                const int g( bd.geo.n_gh );
                n_t = std::max
                    ( n_t, ( int( bd.geo.n_ceff
                                  [ ( a + 1 ) % 3 ] ) + 2 * g )
                         * ( int( bd.geo.n_ceff
                                  [ ( a + 2 ) % 3 ] ) + 2 * g ) );
            }
            if( n_t > 0 )
            p_dev->launch
            ( thin_axis_sync_k< std::remove_const_t
                < std::remove_reference_t
                  < decltype( bd ) > > >,
              std::make_tuple
              ( dim3( ( n_t + 127 ) / 128 ), dim3( 128 ), 0 ),
              mesh.p_cyc->stream, bd );
        }
        p_dev->sync_all_streams(  );
        return;
    }
};
};                     // namespace dynamics::mhd::boundary
