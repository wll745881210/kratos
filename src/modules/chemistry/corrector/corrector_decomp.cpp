#include "corrector.h"

namespace chemistry::solver::corrector
{
////////////////////////////////////////////////////////////
// Decomposition type

__host__ decomp_t:: decomp_t(  ) : size( 0 ), i_map( 0 )
{
    mem .capacity = 4;
    idx = new int [ mem.capacity ];
    num = new f_t [ mem.capacity ];
    return;
}

__host__ void decomp_t::relocate_shifted( void * p0 )
{
    char * p = ( ( char * ) p0 ) + mem.shift;
    return relocate( p, p + sizeof( * num ) * size );
}

__host__ size_t decomp_t::set_shift( const size_t & shift )
{
    mem.shift = shift;
    return shift + size * ( sizeof( int ) + sizeof( f_t ) );
}

__host__ void decomp_t::relocate( void * n_new ,
                                  void * i_new )
{
    memcpy (  i_new, idx, size * sizeof( * idx ) );
    memcpy (  n_new, num, size * sizeof( * num ) );
    delete [  ] idx;
    delete [  ] num;
    idx  = ( int * ) i_new;
    num  = ( f_t * ) n_new;
    return ;
}

__host__ void decomp_t::add( const int & i, const f_t & n )
{
    if( size == mem.capacity )
    {
        mem.capacity *= 2;
        relocate( new f_t[ mem.capacity ] ,
                  new int[ mem.capacity ] );
    }
    idx[ size ] = i;
    num[ size ] = n;
    ++   size ;
    return;
}

__host__ void decomp_t::setup( void * p0 )
{
    num = decltype( num )( ( ( char * ) p0 ) + mem.shift );
    idx = decltype( idx )( num + size ) ;
    return;
}

////////////////////////////////////////////////////////////
// Decomposition list for the corrector

__host__ void decomp_ls_t::init
( const int & n_sp, const int & n_el )
{
    this-> n_sp = n_sp;
    size = n_sp + n_el;
    return super_t::init( size );
}

__host__ decomp_t & decomp_ls_t::s( const int & i )
{
    return ( * this )[ i ];
}

__host__ decomp_t & decomp_ls_t::e( const int & i )
{
    return ( * this )[ i + n_sp ];
}

};                 // namespace chemistry::solver::corrector
