#pragma once

namespace friction
{
////////////////////////////////////////////////////////////
// Shearing source type

template < class prx_T = dynamics::hydro::proxy_t >
struct inflow_t : dynamics::hydro::boundary::phys_base_t
                < prx_T, inflow_t< prx_T > >
{
    ////////// Types and Data //////////
    using float_t = type::float_t;
    using coord_t = type::coord_t;
    
    float_t rho;
    float_t pre;
    coord_t vel;

    ////////// Functions //////////
    __host__ virtual void init
    ( const  input & args, mesh::mod_base_t & mod ) override
    {
        rho  = args.get< float_t > ( "prob", "rho0", 1 );
        pre  = args.get< float_t > ( "prob", "pre0", 1 );
        vel  = args.get< coord_t > ( "prob", "vel0", 0 );
        args( "prob", "vel", vel );        
        return;
    };

    template< class d1d_T, class bdt_T >
    __forceinline__ __device__  void hydro
    ( d1d_T & d, int i_3d[  ], bdt_T & b_d ) const
    {
        for( int i = -d.n_spa / 2; i < 0; ++ i )
        {
            d( i, idx::rho ) = rho;
            d( i, idx::ene ) = pre;
            for( int a = 0; a < 3; ++ a )
                d( i, idx::mom + a ) = vel[ a ];
        }
        return;
    };
};

};                              // namespace friction
