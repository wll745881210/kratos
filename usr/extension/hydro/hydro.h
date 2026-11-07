#pragma once

#include "../algo/interp.h"
#include "proxy.h"
#include "intg.h"

namespace extension
{
////////////////////////////////////////////////////////////
// (Pseudo) hydrodynamics initialization

using intp_t = interp_t< type::float_t, 3 >;

struct hydro_init_t
{
    ////////// Data //////////
    intp_t                          rho;
    intp_t                          pre;
    std::array < intp_t,      3 >   vel;
    std::map   < int,    intp_t > x_spe;
    std::vector< type::float2_t > x_sp0;

    ////////// Functions //////////
    void read( const input & args )
    {
        const auto file
            = args.get< std::string > ( "init", "file" );

        binary_io::default_t bio;
        bio.open    ( file,   "r" );
        bio.load    (             );
        rho.read_bin( bio, "rho_" );
        pre.read_bin( bio, "pre_" );
        rho.set_nearest(  );
        pre.set_nearest(  );

        const auto vel0 = args.get
            < type::coord2_t >( "init", "vel0_cgs" );
        for( int a = 0; a < 3; ++ a )
            if( ! vel[ a ].read_bin
                ( bio, "vel_" + std::to_string( a ) ) )
                vel[ a ].set_fill( vel0[ a ] );

        type::float2_t norm( 0 );
        std::vector < std  ::string > sps  ;
        args( "chemistry", "species", sps );
        for( const auto & s : sps )
        {
            x_sp0.push_back ( args.get< type::float2_t >
                            ( "species_init", s, 1e-20 ) ) ;
            norm += x_sp0.back(  );
        }
        for( auto & x_s : x_sp0 )
             x_s /= norm;

        for( int a = 0; a < sps.size(  ); ++ a )
        {
            if( x_spe[ a ].read_bin( bio, sps[ a ] + "_" ) )
                x_spe[ a ].set_nearest(  );                
            else
                x_spe.erase( a );
        }
        return  bio.close(  );
    };

    template < class vec_T, class x_T, class sps_T >
    void get_spe( vec_T & x_spe, 
                  const x_T & x, const sps_T & sps )
    {
        type::float2_t norm( 0 );
        for( auto & [ i_spe, x_sps ] : this->x_spe )
        {
            x_spe [ i_spe ] = x_sps( x );
            norm += x_spe [ i_spe ];
        }
        if( norm == 0 )
            for( auto  n = 0; n < x_sp0.size(  ); ++ n )
                x_spe[ n ] = x_sp0[ n ];
        else
            for( auto & x_s : x_spe )
                x_s /= norm ;
        return;
    };
};

class hydro_chem_t : public chemistry::hydro::base_t
{
protected:                      // Data
    bool    is_hydro_off;
    type::float_t dt_tst;
    hydro_init_t   f_hyd;
protected:                      // Function
    template< class bdt_T > void
    set_cell( const bdt_T & b_h, const type::idx_t & i );
public:                         // Function
    virtual void read( const  input & args ) override
    {
        enroll_default(  );

        is_hydro_off = args.get< bool >
            ( "chemistry", "hydro_off",  false ) ;
        if( is_hydro_off )
            dt_tst = args.get< type::float_t >
                   ( "chemistry", "dt_tst", 1e-2 );

        using eos_t = chemistry::hydro::eos_t<  >;
        if( ! integrate )        
            enroll_chem_hyd < hydro::prx_t<  >, eos_t,
                              hydro::stp_t<  > > (  );
        return chemistry::hydro::base_t::read( args );
    };

    virtual void init
    ( const input & args, mesh::mesh_t & mod ) override
    {
        f_hyd.read( args );
        return chemistry::hydro::base_t::init( args, mod );
    };  // Do NOT change the sequence of init here

    virtual void step( mesh::mesh_t & mesh ) override
    {
        if( is_hydro_off )
        {
            mesh.p_cyc->p_dt_h[ 0 ] = dt_tst;
            mesh.p_cyc->p_dt_h[ 1 ] = dt_tst;
            for( auto & d : ( * this ) )
            {
                d.h(  ).dt_0 = dt_tst;
                d.d(  ).dt_0 = dt_tst;
            }
            p_dev->sync_all_streams(  );
        }
        else
            chemistry::hydro::base_t::step( mesh );
        return;
    };  // Do nothing for testing chemistry only

    virtual void init_cond
    ( mesh::block::dual_t & d ) override
    {
        if( q_che.expired(  ) || ! q_che.lock(  )->p_eos )
            throw std::runtime_error( "q_che unbound\n" );
        auto & b_h  = dynamic_cast
          < ::dynamics::hydro::block_data_t & > ( d.h(  ) );
        const  auto & geo  ( d.b_w.get(  ).geo ) ;
        for( auto i : utils::loop( geo. n_ceff ) )
            set_cell( b_h, i );
        return;
    };
};

////////////////////////////////////////////////////////////
//

template< class bdt_T >
void hydro_chem_t::set_cell
( const bdt_T & b_h, const type::idx_t & i )
{
    const auto & geo = b_h.geo;
    const auto     x = b_h.geo.x_cc( i );

    auto *       u = b_h.hyd.u.at( i );
    const auto rho = f_hyd.rho( x ) / p_eos->unit.rho0;
    u [ idx::rho ] =   rho;

    auto  x_spe = this-> x_spe;    
    const auto & species( q_che.lock(  )->species );    
    f_hyd.get_spe( x_spe, x, species );
    for( int n = 0; n <  x_spe.size(  ); ++ n )
        u[ n + p_eos->n_hvar(  ) ] = u [ idx::rho ]
             * phys::cgs::ma / mu_cgs_ave * x_spe [ n ];

    const auto pre = f_hyd. pre( x ) / p_eos->unit.ene0;
    const auto   T = pre * p_eos->unit.ene0
                 / ( rho * p_eos->unit.rho0 )
                 * mu_cgs_ave / phys::cgs::kb;
    u[ idx::ene ] = pre * p_eos->cvkb_xn_T_h
                  ( x_spe.data(  ), T );
    for( int n = 0; n < 3; ++ n )
    {
        const auto vel = f_hyd.vel[ n ]( x )
                       / p_eos->unit.vel0;
        u[ idx::ene ] += pow( vel, 2 ) * rho / 2;
        u[ idx::mom + n  ]  = vel      * rho ;
    }
    return;
}

};                              // namespace extension
