#pragma once

#include <map>
#include <mpi.h>

namespace comm
{
////////////////////////////////////////////////////////////
// Device-host buffer pair

using  tag_t = int;
using rank_t = int;

struct buff_t
{
    static       int n_active;
    int              i_active;
    void *                  d;  // User  -side ptr
    void *                  h;  // Shared-window ptr
    size_t                  n;  // size
    size_t                  c;  // capacity
    bool               active;
    bool              is_host;  // true if d a host pointer
    device::stream_t        s;  // Device stream
    buff_t(   )
        : n( 0 ), i_active( 0 ), c( 0 ),
          active( false ), is_host( false ),
          d( nullptr ), h( nullptr ) {  };
    
    void  set_active(  )
    {
        i_active = n_active;
    };
    bool test_active(  )
    {
        active = ( ( i_active -- ) == n_active );
        return active;
    };
    void free( device::base_t & f_dev )
    {
        if( h != nullptr )
            f_dev.free_host( h );
    };
};

struct remote_t
{
    ////////// Data //////////
    std::map< tag_t, buff_t > dat;

    ////////// Functions //////////
    buff_t &  operator(  )
    ( void *  p_d, const size_t  & size, const tag_t & tag,
      device::base_t & f_dev, const device::stream_t & str,
      const float & safety = 0.5f )
    {
        auto & b( dat[ tag ] );
        b.d      =   p_d;
        b.s      =   str;
        b.n      =  size;
        b.set_active(  );
        if( size > b.c )
        {
            f_dev.sync_stream        ( b.s );
            b.c = b.n + ( safety   *   b.n );
            b.free                 ( f_dev );
            b.h = f_dev.f_malloc_host( b.c );
            f_dev.sync_stream        ( b.s );
        }
        return b;
    };

    void sync( device::base_t & f_dev )
    {
        for( auto it = dat.begin(  ); it != dat.end(  ); )
        {
            auto  & b( it->second );
            if( b.test_active(  ) )
                f_dev.a_cp( b.d, b.h, b.n, b.s );
            if( b.i_active > 0 )
                ++ it;
            else
            {
                b  .   free( f_dev );
                it = dat.erase( it );
            }
        }
        return;
    };

    void free( device::base_t & f_dev )
    {
        for( auto & [ tag, b ] : dat )
            f_dev.free_host( b.h );
        return;
    };
};

struct local_t
{ 
    ////////// Data //////////
    int                     lrank;
    bool                     send;
    bool                   update;
    void *                   dat0; // Head
    size_t               capacity;
    std::map< tag_t, buff_t > dat;

    ////////// Functions //////////
    local_t( ) : lrank( 0 ), send( false ), update( false ),
                 dat0( nullptr ), capacity( 0 ) {  };
    
    void clear(  )
    {
        dat0 = nullptr;
        capacity = 0;
        return;
    };    

    void operator (  )
    ( void  * p_d , const size_t & size,
      const tag_t & tag, device::base_t & f_dev,
      const device::stream_t & strm,
      const bool   & on_host  = false )
    {
        auto &  b =   dat   [ tag ];
        update   |= ( size != b.n );
        b.d       =  p_d;
        b.n       = size;
        b.s       = strm;
        b.is_host = on_host;
        b.set_active(  );
        return;
    };

    void assign( void * p )
    {
        dat0 = p;
        size_t shift( 0 );
        for( auto & [ tag, lbuf ] : dat )
        {
            lbuf.h = dat0 == nullptr ? nullptr :
                ( ( char * ) dat0 ) + shift;
            shift += lbuf.n;
        }
        return;
    };

    void regularize( device::base_t & f_dev,
                     const float & safety = 0.5f )
    {
        size_t shift( 0 );
        for( auto it = dat.begin(  ); it != dat.end(  );  )
        {
            auto & b = it->second;
            if( b.test_active(  ) )
            {
                shift += b.n;
                ++ it ;
            }
            else if( b.i_active > 0 )
            {
                shift += b.n;
                ++ it ;
            }
            else
            {
                update = true;
                it = dat.erase( it );
            }
        }
        if( shift >  capacity )
        {
            capacity = shift + size_t( shift * safety );
            update = true;
        }
        if( update )
            assign( dat0 );
        update = false;
        return;
    };
};

};                                 //    namespace comm
