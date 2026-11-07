#pragma once

#include "../../src/mesh/mesh.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  cycle with a dt - collapse fuse ( [cycle] dt_collapse,
//  default 0 = off ):  when the freshly reduced dt for the
//  next cycle drops below the threshold --  or goes NaN --
//  dump the state and stop cleanly.   The saved state is
//  the one that PRODUCED the collapsed dt:  the crash -
//  forensics hook.   The dt is the MPI - reduced global
//  one,  so all ranks take the branch together.

struct disk_cycle_t : mesh::cycle::base_t
{
    type::float2_t dt_collapse;
    //  [cycle] dt_collapse_n:  how many CONSECUTIVE steps may sit below
    //  dt_collapse before the fuse fires ( default 1 = the old
    //  one-strike behaviour, bit-identical ).  Rationale:  the cycle
    //  seeds the next step's dt as  prev * dt_expand  ( default 2 ), so
    //  a single transient dip -- one stiff implicit step, one chemistry
    //  spike, one bad cell's demand -- RECOVERS within a few steps and
    //  is not a collapse.  Only a dt that STAYS below the threshold is.
    //  A non-finite or non-positive dt always fires at once:  that is
    //  not a transient.
    int dt_collapse_n;
    int n_collapse;

    __host__ disk_cycle_t(  ) : base_t(  ), dt_collapse( 0 ),
        dt_collapse_n( 1 ), n_collapse( 0 )
    {  };

    __host__ virtual void init
    ( const input & args, mesh::mesh_t & mesh ) override
    {
        base_t::init( args, mesh );
        dt_collapse = args.get< type::float2_t >
            ( "cycle", "dt_collapse", 0 );
        dt_collapse_n = args.get< int >
            ( "cycle", "dt_collapse_n", 1 );
        if( dt_collapse_n < 1 )
            dt_collapse_n = 1;
        n_collapse = 0;
        return;
    };

protected:
    __host__ virtual bool step( mesh::mesh_t & mesh )
        override
    {
        if( ! base_t::step( mesh ) )
            return false;
        if( dt_collapse > 0 )
        {
            //  The t_lim TOP - UP step is clamped to  t_lim - t  by the
            //  cycle  ( cycle.cpp : 89:  p_dt_h[0] = min( p_dt_h[0],
            //  t_lim - t ) ),  so ITS dt is small BY CONSTRUCTION.
            //  Reaching t_lim is the NORMAL end of the run --  and it is
            //  base_t::step() that ends it ( its  `t >= t_lim`  guard
            //  fires on the next call ),  so fusing HERE is a false
            //  alarm:  it prints "DT - COLLAPSE fuse" for a healthy run
            //  and dumps a checkpoint whose dt is the clamped one,  i.e.
            //  a checkpoint that cannot be restarted ( the restart seeds
            //  the device dt from it --  dead - ends M14 / M15 ).   Skip
            //  the fuse once we have arrived;  a NON - t_lim small dt is
            //  still caught exactly as before.
            const bool at_tlim( ( t_lim - t ) <= 0 );
            if( at_tlim )
                return true;
            const bool bad( ! ( dt(  ) > 0 ) );       // NaN or <= 0
            const bool sml( dt(  ) < dt_collapse );
            if( bad || sml )
            {
                n_collapse = bad ? dt_collapse_n : n_collapse + 1;
                if( n_collapse < dt_collapse_n )
                {
                    if( mesh.p_com->is_root(  ) )
                        std::cout << "DT - COLLAPSE warning:  dt = "
                                  << dt(  ) << " < " << dt_collapse
                                  << " at cycle " << i_cycle
                                  << ", t = " << t << "  ( "
                                  << n_collapse << " / "
                                  << dt_collapse_n
                                  << " consecutive ) -- continuing."
                                  << std::endl;
                    return true;
                }
                if( mesh.p_com->is_root(  ) )
                    std::cout << "DT - COLLAPSE fuse:  dt = "
                              << dt(  ) << " at cycle "
                              << i_cycle << ", t = " << t
                              << "  ( below the threshold for "
                              << n_collapse << " consecutive step(s) )"
                              << " --  dumping and stopping."
                              << std::endl;
                mesh.save( file_name( mesh ) );
                return false;
            }
            n_collapse = 0;
        }
        return true;
    };
};

};                              // namespace interm
