#include "../../../mesh/boundary/boundary.h"
#include "dual_data.h"

namespace multigrid::data
{
////////////////////////////////////////////////////////////
// Initialization and finalization

void dual_t::free(  )
{
    for( auto &  p : p_d_arr )  
        p->free  ( p_dev->f_free      );
    if( p_h )
        p_h->free( p_dev->f_free_host );
    return; // Not using super_t::free to avoid double free
}

void dual_t::setup(  )
{
    lvl_0 = b_w.get(  ) . reg.level ;
    mesh::block::dual_t ::setup(   );
    const auto & geo_h( b_w.get(   ). geo );
    //  Only THINNABLE axes count.  An axis with <= 2 cells is
    //  a thin / axisymmetric direction:  halving 2 -> 1 costs
    //  the whole multigrid ( n_lvl = 1 + log2(2.5) = 2 leaves
    //  lvl_lim = 0 ),  and halving 1 -> 1 is a no-op whose
    //  n_min = 1 makes lvl_lim = -1.  A grid with every axis
    //  <= 2 falls back to the plain min.
    int  n_min( 1 << 30 );
    for( int a = 0; a < geo_h.n_dim; ++ a )
        if( geo_h.n_ceff[ a ] > 2 )
            n_min = std::min( n_min, geo_h.n_ceff[ a ] );
    if( n_min == ( 1 << 30 ) )
    {
        n_min = geo_h.n_ceff[ 0 ];
        for( int a = 1; a < geo_h.n_dim; ++ a )
            n_min = std::min( n_min, geo_h.n_ceff[ a ] );
    }
    const int n_lvl = 1 + log2( n_min + 0.5 );

    p_d_arr.push_back( std::dynamic_pointer_cast
                     < block_data_t > ( p_d ) );
    // p_d must be some derivation of block_data_t,
    // otherwise it is eligible to throw an exception    
    for( int d_l = 1; d_l < n_lvl; ++ d_l )
    {
        p_d_arr.push_back( p_d_arr.front(  )->clone(  ) );
        p_d_arr.back(  )->set_lvl( * this, d_l );
    }
    auto & b_d = dynamic_cast< block_data_t & >( d(  ) );
    auto & b_h = dynamic_cast< block_data_t & >( h(  ) );
    b_d.mgd.null( [ & ] ( void * p, const size_t & s )
    {  if( p != nullptr ) p_dev->f_mset( p, 0, s, 0 ); } );
    b_h.mgd.null( [ & ] ( void * p, const size_t & s )
    {  if( p != nullptr ) memset( p, 0, s );           } );
    return;
}

////////////////////////////////////////////////////////////
// Block data access by level

int dual_t:: n_lvl(  ) const
{
    return p_d_arr.size(  ) ;
}

block_data_t & dual_t::d_mg( const int & lvl ) const
{
    return ( * p_d_arr.at( -lvl ) );
}
};                          // namespace multigrid::block
