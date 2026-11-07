#pragma once

#include "../../../src/modules/dynamics/hydro/hydro.h"
#include "../../../src/utilities/phys/constants.h"
#include "../../../src/utilities/phys/unit.h"
#include "../../../src/types.h"

#include "intg.h"
#include "boundary.h"

namespace friction
{
////////////////////////////////////////////////////////////
// Hydrodynamic under cosmology

class hydro_t : public dynamics::hydro_t
{
protected:                      // Types
    using float2_t = type::float2_t;
    using coord2_t = type::coord2_t;

    using bdk_t = dynamics::hydro::boundary::keeper_t;
    using rk2_t = dynamics::hydro::integrator::rk2_t<  >;
    using prx_t = dynamics::hydro::proxy_t;
    using stp_t = friction::stp_t<  >;
public:                         // Functions
    virtual void read( const input & args ) override
    {
        rho0 = args.get< float2_t >( "prob", "rho0", 1 );
        pre0 = args.get< float2_t >( "prob", "pre0", 1 );
        vel0 = args.get< coord2_t >( "prob", "vel0", 0 );

        enroll_bdk   < bdk_t >(      );
        enroll< prx_t, stp_t >(      );        
        dynamics::base_t::read( args );
        p_bdk->enroll_phys< inflow_t<  > >( "inflow" );
        return;
    };

    virtual void init
    ( const input & args, mesh::mesh_t & mesh ) override
    {
        dynamics::hydro_t::init( args, mesh );

        phys::unit_t<   > unit  ;
        unit.init( args );
        p_eos->d_floor = args.get< float2_t >
            ( "prob", "rho_floor", 1e-8 );
        p_eos->t_floor = args.get< float2_t >
            ( "prob", "tmp_floor", 1e-1 );

        auto & star
             = dynamic_cast < stp_t & >( * p_itg ).star;
        star.rho_floor = p_eos->d_floor;
        star.tmp_floor = p_eos->t_floor;
        return;
    };
};                              // class hydro_t
};                              // namespace friction
