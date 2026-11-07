#include "../../../utilities/mapping/loop.h"
#include "types/block_data.h"
#include "mhd.h"

namespace dynamics
{
////////////////////////////////////////////////////////////
// MHD

void mhd_t::read( const input & args )
{
    b0 = args.get< type::coord2_t >( "init", "b0", 0 );
    return hydro_t::read( args );
}

void mhd_t::enroll_default(  )
{
    if( ! f_bdt_yield )
        enroll_dat< mhd::      block_data_t > (  );
    if( ! p_bdk )
        enroll_bdk< mhd::boundary::keeper_t > (  );
    if( ! integrate  )
        enroll<  >(  );
    return;
}

void mhd_t::init_cond( mesh::block::dual_t & d )
{
    hydro_t ::init_cond( d );
    auto & b_h  = dynamic_cast
                < mhd::block_data_t & > ( d.h(  ) );
    const  auto & geo ( d.b_w.get(  ).geo );
    auto n_ceff = geo.n_ceff;
    for( auto & n_s : n_ceff.x )
        n_s += ( n_s > 1 );

    for( auto i : utils::loop( n_ceff ) )
    {
        auto * bf = b_h.fld.bf.at( i );
        for( int a = 0;  a < 3; ++ a )
             bf[ a ] = b0[ a ];

        bool is_cc( true );
        for( int a = 0; a < 3; ++ a )
            is_cc &= ( i[ a ] < b_h.geo.n_ceff[ a ] );
        if( is_cc )
        {
            auto * u = b_h. hyd.u.at( i );
            for( int a = 0; a < 3; ++ a )
                u[ idx::ene ] += pow( b0[ a ], 2 ) / 2;
        }
    }
    return;
}
};                              // namespace dynamics
