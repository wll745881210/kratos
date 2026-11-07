#include "edge_mg_cplx.h"

#include <algorithm>
#include <tuple>

namespace mhd_ee::ni::edge_mg
{

////////////////////////////////////////////////////////////
//  The host half of the cell complex.
//
//  `setup` builds the offset/stride tables,  `d0_row` and
//  `d1_row` give the incidence rows the gates compare
//  against ( d1 d0 = 0 is an integer identity,  so the
//  gates may look at them ),  and the two `make_keep_*`
//  build the live-dof masks.
//
//  Nothing here is device code:  the device reaches the
//  complex through the tables only,  and every device
//  function that needs an index is `__host__ __device__`
//  and lives in the header.

__host__ void cplx_t::setup
( const idx_t & n_cell_eff, const bool ( & p )[ 3 ] )
{
    setup( n_cell_eff, p, false, false );
    return;
};

__host__ void cplx_t::setup
( const idx_t & n_cell_eff, const bool ( & p )[ 3 ],
  const bool & pole_lo, const bool & pole_hi )
{
    pole[ 0 ] = pole_lo ? 1 : 0;
    pole[ 1 ] = pole_hi ? 1 : 0;
    n = n_cell_eff;
    for( int a = 0; a < 3; ++ a )
    {
        per[ a ] = p[ a ] ? 1 : 0;
        NN[ a ]  = per[ a ] ? n[ a ] : ( n[ a ] + 1 );
    }
    n_node = int( NN[ 0 ] ) * int( NN[ 1 ] ) * int( NN[ 2 ] );
    n_cell = int( n[ 0 ] ) * int( n[ 1 ] ) * int( n[ 2 ] );
    n_edge = 0;
    n_face = 0;
    for( int a = 0; a < 3; ++ a )
    {
        off_e[ a ] = n_edge;
        off_f[ a ] = n_face;
        for( int q = 0; q < 3; ++ q )
        {
            dim_e[ a ][ q ] = ( q == a )
                            ? int( n[ a ] )
                            : int( NN[ q ] );
            dim_f[ a ][ q ] = ( q != a )
                            ? int( n  [ q ] )
                            : ( per[ a ] ? int( n[ a ] )
                                         : int( n[ a ] ) + 1 );
        }
        int se( 1 ), sf( 1 );
        for( int q = 0; q < 3; ++ q )
        {
            st_e[ a ][ q ] = se;
            st_f[ a ][ q ] = sf;
            se *= dim_e[ a ][ q ];
            sf *= dim_f[ a ][ q ];
        }
        n_edge += se;
        n_face += sf;
    }
    return;
};

__host__ void cplx_t::d0_row
( const int & e, int ( & nd )[ 4 ], int ( & sg )[ 4 ],
  int & nz ) const
{
    int a( 0 );
    int a_off( e );
    for( a = 2; a >= 0; -- a )
    {
        if( a_off >= off_e[ a ] )
        {
            break;
        }
    }
    a_off -= off_e[ a ];
    idx_t nd_( 0 );
    for( int q = 0; q < 3; ++ q )
    {
        nd_[ q ] = ( a_off / st_e[ a ][ q ] ) % dim_e[ a ][ q ];
    }
    if( int( NN[ a ] ) == 1 )
    {
        nz = 0;
        return;
    }
    idx_t tail( nd_ );
    idx_t head( nd_ );
    head[ a ] = ( head[ a ] + 1 ) % int( NN[ a ] );
    nz = 2;
    nd[ 0 ] = node( tail );
    sg[ 0 ] = -1;
    nd[ 1 ] = node( head );
    sg[ 1 ] = +1;
    return;
};

__host__ void cplx_t::d1_row
( const int & f, int ( & ed )[ 4 ], int ( & sg )[ 4 ],
  int & nz ) const
{
    int b_( 0 );
    int f_off( f );
    for( b_ = 2; b_ >= 0; -- b_ )
    {
        if( f_off >= off_f[ b_ ] )
        {
            break;
        }
    }
    f_off -= off_f[ b_ ];
    idx_t nd_( 0 );
    for( int q = 0; q < 3; ++ q )
    {
        nd_[ q ] = ( f_off / st_f[ b_ ][ q ] ) % dim_f[ b_ ][ q ];
    }
    const int a = b_;
    const int b = ( a + 1 ) % 3;
    const int l = ( a + 2 ) % 3;
    idx_t cn[ 4 ];
    cn[ 0 ] = nd_;
    cn[ 1 ] = nd_;
    cn[ 1 ][ b ] += 1;
    cn[ 2 ] = cn[ 1 ];
    cn[ 2 ][ l ] += 1;
    cn[ 3 ] = nd_;
    cn[ 3 ][ l ] += 1;
    nz = 4;
    for( int t = 0; t < 4; ++ t )
    {
        const idx_t & c0 = cn[ t ];
        const idx_t & c1 = cn[ ( t + 1 ) % 4 ];
        int ax( -1 );
        for( int q = 0; q < 3; ++ q )
        {
            if( c1[ q ] != c0[ q ] )
            {
                ax = q;
            }
        }
        idx_t lo( c0 );
        lo[ ax ] = c0[ ax ] < c1[ ax ] ? c0[ ax ] : c1[ ax ];
        ed[ t ] = edge( ax, lo );
        sg[ t ] = ( c1[ ax ] > c0[ ax ] ) ? +1 : -1;
    }
    return;
};

__host__ void make_keep
( const cplx_t & c, std::vector< int > & keep,
  std::vector< int > & comp, int & n_compact,
  const bool & site_major, const int & bc_edge )
{
    keep.assign( c.n_edge, 1 );
    comp.assign( c.n_edge, -1 );
    n_compact = 0;
    if( site_major )
    {
        std::vector< std::tuple< int, int, int, int, int > > ord;
        ord.reserve( c.n_edge );
        for( int e = 0; e < c.n_edge; ++ e )
        {
            if( c.degenerate_e( e )
             || ( bc_edge == 2 && wall_tangent_e( c, e ) ) )
            {
                keep[ e ] = 0;
                continue;
            }
            int   a( 0 );
            idx_t nd( 0 );
            c.edge_key( e, a, nd );
            ord.push_back( std::make_tuple( int( nd[ 0 ] ), int( nd[ 1 ] ),
                                            int( nd[ 2 ] ), a, e ) );
        }
        std::sort( ord.begin(  ), ord.end(  ) );
        for( const auto & t : ord )
        {
            comp[ std::get< 4 >( t ) ] = n_compact ++;
        }
        return;
    }
    for( int e = 0; e < c.n_edge; ++ e )
    {
        if( c.degenerate_e( e ) )
        {
            keep[ e ] = 0;
            continue;
        }
        comp[ e ] = n_compact ++;
    }
    return;
};

__host__ void make_keep_face
( const cplx_t & c, std::vector< int > & keep,
  std::vector< int > & comp, int & n_compact )
{
    keep.assign( c.n_face, 1 );
    comp.assign( c.n_face, -1 );
    n_compact = 0;
    for( int f = 0; f < c.n_face; ++ f )
    {
        if( c.degenerate_f( f ) )
        {
            keep[ f ] = 0;
            continue;
        }
        comp[ f ] = n_compact ++;
    }
    return;
};

};    // namespace mhd_ee::ni::edge_mg