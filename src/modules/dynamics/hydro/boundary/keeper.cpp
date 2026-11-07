#include "phys_functors.h"
#include "comm_functions.h"
#include "keeper.h"
#include <thread>

namespace dynamics::hydro::boundary
{
////////////////////////////////////////////////////////////
// Initializer

keeper_t::keeper_t(  ) : mesh::boundary::keeper_t (  )
{
    //  Boundary kinds:
    //    ref  : reflect, flips the normal momentum;
    //    out  : outflow = zero-gradient + inward normal
    //           momentum clamp ( suppresses inflow by
    //           design -- do not use where inflow occurs );
    //    fre  : free zero-gradient extrapolation ( inflow
    //           passes through unchanged );
    //    fixd : Dirichlet, ghost fixed to the ambient
    //           [init] medium ( prescribed inflow );
    //    pol  : ( sph, theta sides ) through-pole wedge
    //           continuation, MHD keeper only;
    //    axs  : ( cyl, r sides ) through-axis wedge
    //           continuation, MHD keeper only.
    enroll_phys< reflect_t<  > > ( "ref"  );
    enroll_phys< outflow_t<  > > ( "out"  );
    enroll_phys<    free_t<  > > ( "fre"  );
    p_fixd = enroll_phys< fixed_t<  > > ( "fixd" );
    return;
}

void keeper_t::init( const input & args, mesh_t & mesh )
{
    n_th_lim = args.get< int >
           ( "dynamics", "bnd_thread_limit", 64 );

    pool_phys.resize( 6 );
    std::vector< std::string > bnd_kinds  ;
    args( "boundary", "kinds", bnd_kinds );
    for( int s = 0; s < pool_phys.size(  ); ++ s )
    {
        const auto & kind( bnd_kinds.at( s ) );
        if( kind.find( "per" ) != std::string::npos )
            continue; // Periodic is not an actual boundary
        if( kind.find( "pol" ) != std::string::npos &&
            s / 2 != 1 )
            throw std::runtime_error
                ( "pol bnd only exists on theta sides" );
        if( kind.find( "axs" ) != std::string::npos &&
            s / 2 != 0 )
            throw std::runtime_error
                ( "axs bnd only exists on radial sides" );

        auto it  = holder_phys. find( kind );
        if ( it == holder_phys.  end(    ) )
            throw std::runtime_error( "Undef bnd " + kind );
        pool_phys.at( s ) = std::ref( it->second ) ;
    }
    //  fixd: Dirichlet ghost state = the ambient medium.
    //  The boundary functors operate on the PRIMITIVE w,
    //  so the fix is simply ( rho0, pre0, vel0 ).
    {
        const auto rho0 = args.get< type::float2_t >
                        ( "init", "rho0", 0 );
        const auto pre0 = args.get< type::float2_t >
                        ( "init", "pre0", 0 );
        const auto vel0 = args.get< type::coord2_t >
                        ( "init", "vel0", 0 );
        p_fixd->w_fix[ idx::rho ] = rho0;
        p_fixd->w_fix[ idx::ene ] = pre0;
        for( int n = 0; n < 3; ++ n )
            p_fixd->w_fix[ idx::mom + n ] = vel0[ n ];
    }
    return mesh::boundary::keeper_t::init( args, mesh );
}

////////////////////////////////////////////////////////////
// Action

bool keeper_t::desired_mode( const neighbor_t & nb )
{
    return ( nb.mode < 1 );
}

void keeper_t::update( mesh_t & mesh )
{
    mesh::boundary::keeper_t::update ( mesh );
    for( auto & [ b, m_nb ] : bnd_comm )
        for( auto & [ nb, h ] :   m_nb )
        {
             if( h.f_pack )
             {
                 if  ( h.same_rank(  )   && h.d_lvl ==  0 )
                     q_same.     push_back( std::ref( h ) );
                 else if( h.d_lvl <= 0 )
                     q_ghst.pack.push_back( std::ref( h ) );
                 else
                     q_plgt.pack.push_back( std::ref( h ) );
             }
             if( h.f_unpk )
             {
                 if( h.d_lvl >= 0 )
                     q_ghst.unpk.push_back( std::ref( h ) );
                 else
                     q_plgt.unpk.push_back( std::ref( h ) );
             }
             if( h.f_send )
             {
                 if( h.d_lvl <= 0 )
                     q_ghst.send.push_back( std::ref( h ) );
                 else
                     q_plgt.send.push_back( std::ref( h ) );
             }
             if( h.f_recv )
             {
                 if( h.d_lvl >= 0 )
                     q_ghst.recv.push_back( std::ref( h ) );
                 else
                     q_plgt.recv.push_back( std::ref( h ) );
             }
             if( h.f_fpck )
             {
                 if( h.d_lvl <  0 )
                     q_flux.pack.push_back( std::ref( h ) );
                 else if( ! h.same_rank(  ) )
                     q_flux.unpk.push_back( std::ref( h ) );
             }
             if( h.f_fcom )
             {
                 if( h.d_lvl <  0 )
                     q_flux.send.push_back( std::ref( h ) );
                 else
                     q_flux.recv.push_back( std::ref( h ) );
             }
        }
    return;
}

void keeper_t::exec_queue
( queue_t & q, const bool & comm, const int & step,
  const std::function< void(  ) > & func )
{
    if( q.pack.size(  ) + q.unpk.size(  ) < 1 )
    {
        func           (     );
        return;
    }
    for( auto & h_w : q.pack )
         comm ? h_w.get(     ).f_fpck( step ):
                h_w.get(     ).f_pack( step );
    for( auto & h_w : q.send )
         comm ? h_w.get(     ).f_fcom(      ):
                h_w.get(     ).f_send(      );
    for( auto & h_w : q.recv )
         comm ? h_w.get(     ).f_fcom(      ):
                h_w.get(     ).f_recv(      );
    func               (     );
    if( q.send.size(  ) + q.recv.size(  ) > 0 )
        p_com-> wait_all_d(  );
    for( auto & h_w : q.unpk )
         comm ? h_w.get(     ).f_fpck( step ):
                h_w.get(     ).f_unpk( step );
    return;
}

void keeper_t::act_bnd( mesh_t & mesh, const int & step )
{
    exec_queue( q_ghst, false, step, [ & ] (  )
    {
        for( auto & [ reg , v_pb ] : bnd_phys )
            for( auto & h : v_pb )
                h.launch  ( h,    step );
        for( auto & h_w : q_same       )
            h_w.get(  ) . f_pack( step );
    }   );
    exec_queue( q_plgt, false, step, [ & ] (  ) {  } );
    return;
}

void keeper_t::act_flx( mesh_t & mesh, const int & step )
{
    exec_queue( q_flux,  true, step, [ & ] (  ) {  } );
    return;
}

};  // namespace dynamics::hydro::boundary
