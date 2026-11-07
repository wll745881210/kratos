#include "stoichiometry.h"
#include "../chemistry.h"

namespace chemistry
{
////////////////////////////////////////////////////////////
// Stoichiometric projection

__host__ void sto_t::init( device:: base_t &   f_dev ,
                           const mat_int_t & sp_elem )
{
    n  = sp_elem.n ;
    auto [ u, v, s ]  = utils::math::svd ( sp_elem   );
    std::vector< size_t > idx( sp_elem.n - sp_elem.m );
    std::iota( idx.begin(  ), idx.end(  ), sp_elem.m );
    v.col_rearrange  ( idx );
    const auto res = v.mul ( v.tran(  ) );
    std::vector < f_t >  a ( n * n  );
    for( size_t i = 0; i < a.size(  ); ++ i )
        a[ i ] = res.dat[ i ];

    const size_t s_p( a.size(  ) * sizeof( f_t ) );
    dat  = ( f_t * ) f_dev.f_malloc_const  ( s_p );
    f_dev.           f_cc ( dat, a.data(  ), s_p );
    return;
}


__host__ void sto_t::init( device::base_t &   f_dev ,
                           const sp_vec_t & species )
{
    std::map< int, int > e_map;
    for( const auto & s : species )
        for( const auto & [ z, n_e ] : s.elem )
            e_map [ z ] = 0 ;
    int i_e_count ( 0 ) ;
    for( auto & [ e, i_e ] : e_map )
        i_e = ( i_e_count ++ );

    mat_int_t sp_elem( e_map.size(  ), species.size(   ) );
    for( size_t i = 0; i < species.size(  ); ++ i )
        for( const auto & [ z, n_e ] : species[ i ].elem )
            sp_elem( e_map[ z ], i ) = n_e;
    return init( f_dev, sp_elem );
}

};
