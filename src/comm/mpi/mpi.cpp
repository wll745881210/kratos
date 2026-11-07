#ifdef  __MPI__                  // Skip if not MPI-enabled

#include "mpi.h"
#include <thread>

namespace comm
{
////////////////////////////////////////////////////////////
// Type

int buff_t::n_active( 100 );

////////////////////////////////////////////////////////////
// Constructor

int mpi_t::count( 0 );          // Total MPI instances

mpi_t:: mpi_t(  )
{
    int ini( 0 ),    mt( 0 );
    MPI_Initialized( & ini );
    if( ini == 0 )
    {
        MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, & mt );
        if( mt < MPI_THREAD_MULTIPLE )
            throw std::runtime_error( "Not multi-th MPI" );
    }
    root = 0;
    ++ count;
    finalized = false;
    MPI_Comm_dup ( MPI_COMM_WORLD, & comm );
    MPI_Comm_rank( comm, &   rank_ );
    MPI_Comm_size( comm, & n_rank_ );
    local_optimize           =  true;
    local_init   (                 );
    wait_background          = false;
    return;
}

mpi_t::~mpi_t(  )
{
    this->finalize(  );
    if( ( -- count ) == 0 )
    { 
        int flag      ( 0      );
        MPI_Finalized ( & flag );
        if( flag == 0 )
            MPI_Finalize(  );
    }
    return;    
}

void mpi_t::finalize(  )
{
    if( finalized )
        return;
    if( p_dev )
        p_dev->sync_all_streams(  );    
    local_finalize (  );
    remote_finalize(  );
    if( comm != MPI_COMM_WORLD )
        MPI_Comm_free( &  comm );
    finalized = true;
    return;
}

void mpi_t::init( const input & args )
{
    local_optimize   = args.get < bool >
        ( "comm", "local_optimize",  1 ) && nl_rank > 1;
    wait_background  = args.get < bool >
        ( "comm", "wait_background", 0 );
    buff_t::n_active = args.get <  int >
        ( "comm", "buff_n_active", 100 );
    return;
}

std::shared_ptr< base_t > mpi_t::split(  ) const
{
    auto  res = std::make_shared  < mpi_t > (  );
    res-> local_optimize = this-> local_optimize;
    res->wait_background = this->wait_background;
    return res;
}

////////////////////////////////////////////////////////////
// Direct memory access for node-local processes

template< class fun_T >
void mpi_t::map_local( const fun_T & f )
{
    for( int m = 0; m < nl_rank; ++ m )
        if( m != l_rank_ )
            f( m, m < l_rank_ ? local_send : local_recv ,
                  m < l_rank_ ? local_recv : local_send );
    return;
}

int mpi_t::lrank_of( const int & rank )
{
    auto it = lrank_of_rank.find( rank );
    if( it == lrank_of_rank.end(  ) )
        return -1;
    return it->second;
}

bool mpi_t::is_local( const int & rank )
{
    return local_optimize && lrank_of( rank ) >= 0;
}

void mpi_t::local_init(  )
{
    MPI_Comm_split_type( comm , MPI_COMM_TYPE_SHARED,
                         rank_, MPI_INFO_NULL, & l_comm );
    MPI_Comm_rank ( l_comm, &    l_rank_ );
    MPI_Comm_size ( l_comm, &   nl_rank  );

    local_optimize     &= nl_rank > 1;
    local_window_locked = false;
    local_window_pinned = false;
    local_dat0          = nullptr;
    local_capacity      = 0;
    local_send.resize( nl_rank );
    local_recv.resize( nl_rank );
    local_capacity_all.resize( nl_rank * nl_rank, 0 );

    std::vector< int > ranks( nl_rank );
    MPI_Allgather( & rank_, 1, MPI_INT,
                   ranks.data(  ), 1, MPI_INT, l_comm );
    lrank_of_rank.clear(  );
    for( int m = 0; m < nl_rank; ++ m )
    {
        lrank_of_rank[ ranks[ m ] ] = m;
        local_send[ m ].lrank = m;
        local_recv[ m ].lrank = m;
        local_send[ m ].send  = true;
        local_recv[ m ].send  = false;
    }
    return;
}

void mpi_t::local_finalize(  )
{
    if( l_comm == MPI_COMM_NULL )
        return;
    MPI_Barrier( l_comm );
    if( local_window_locked )
    {
        MPI_Win_unlock_all( local_window );
        local_window_locked = false;
        if( local_window_pinned )
        {
            p_dev->unpin( local_dat0 );
            local_window_pinned = false;
        }
        MPI_Win_free( & local_window );
        local_dat0 = nullptr;
    }
    for( auto & local : local_send )
        local.clear(  );
    for( auto & local : local_recv )
        local.clear(  );
    MPI_Comm_free( & l_comm );
    l_comm = MPI_COMM_NULL;
    return;
}

void mpi_t::local_synchronize(  )
{
    if( ! local_optimize )
        return this->barrier(  );

    for( int m = 0; m < nl_rank; ++ m )
        if( m != l_rank_ )
        {
            local_send[ m ].regularize( * p_dev );
            local_recv[ m ].regularize( * p_dev );
        }

    std::vector< size_t > local_cap( nl_rank, 0 );
    size_t new_capacity( 0 );
    for( int m = 0; m < nl_rank; ++ m )
    {
        local_cap[ m ] = local_send[ m ].capacity;
        new_capacity += local_cap[ m ];
    }
    MPI_Allgather
        ( local_cap.data(  ), nl_rank, MPI_UNSIGNED_LONG,
          local_capacity_all.data(  ), nl_rank,
          MPI_UNSIGNED_LONG, l_comm );

    int has_window( new_capacity > 0 );
    MPI_Allreduce( MPI_IN_PLACE, & has_window, 1,
                   MPI_INT, MPI_MAX, l_comm );
    int reset_window
        ( ( new_capacity != local_capacity )
          || ( has_window && ( ! local_window_locked ) )
          || ( ( ! has_window ) && local_window_locked ) );
    MPI_Allreduce( MPI_IN_PLACE, & reset_window, 1,
                   MPI_INT, MPI_MAX, l_comm );
    if( reset_window )
    {
        if( local_window_locked )
        {
            MPI_Win_unlock_all( local_window );
            local_window_locked = false;
            if( local_window_pinned )
            {
                p_dev->unpin( local_dat0 );
                local_window_pinned = false;
            }
            MPI_Win_free( & local_window );
            local_dat0 = nullptr;
        }
        local_capacity = new_capacity;
        if( has_window )
        {
            MPI_Win_allocate_shared
                ( local_capacity, 1, MPI_INFO_NULL, l_comm,
                  & local_dat0, & local_window );
            MPI_Win_lock_all( 0, local_window );
            local_window_locked = true;
            if( local_capacity > 0 )
            {
                p_dev->pin( local_dat0, local_capacity );
                local_window_pinned = true;
            }
        }
    }

    size_t shift( 0 );
    for( int m = 0; m < nl_rank; ++ m )
    {
        local_send[ m ].assign
            ( local_capacity > 0 ?
              ( ( char * ) local_dat0 ) + shift : nullptr );
        shift += local_send[ m ].capacity;
    }
    if( has_window )
        for( int m = 0; m < nl_rank; ++ m )
        {
            if( m == l_rank_ )
                continue;
            MPI_Aint c( 0 );
            int      d( 0 );
            void *   p( nullptr );
            MPI_Win_shared_query
                ( local_window, m, & c, & d, & p );
            size_t off( 0 );
            for( int n = 0; n < l_rank_; ++ n )
                off += local_capacity_all
                      [ m * nl_rank + n ];
            if( local_recv[ m ].capacity >
                local_capacity_all[ m * nl_rank + l_rank_ ])
                throw std::runtime_error
                    ( "MPI_Win inconsistent size.\n" );
            local_recv[ m ].assign
                ( local_recv[ m ].capacity > 0 ?
                  ( ( char * ) p ) + off : nullptr );
        }
    for( auto & local : local_send )
        for( auto & [ rag,  b ] : local. dat )
        {
             if( ! b.active )
                 continue;
             if( b.is_host )
                 memcpy ( b.h, b.d, b.n );
             else
             {
                 p_dev->a_cp( b.h, b.d, b.n, b.s );
                 p_dev->sync_stream        ( b.s );
             }
        }
    if( local_window_locked )
        MPI_Win_sync( local_window );
    MPI_Barrier( l_comm );
    if( local_window_locked )
        MPI_Win_sync( local_window );
    for( auto & local : local_recv )
        for( auto & [ rag,  b ] : local. dat )
        {
             if( ! b.active )
                 continue;
             if( b.is_host )
                 memcpy ( b.d, b.h, b.n );
             else
             {
                 p_dev->a_cp( b.d, b.h, b.n, b.s );
                 p_dev->sync_stream        ( b.s );
             }
        }
    MPI_Barrier( l_comm );
    return;
}

////////////////////////////////////////////////////////////
// Conventional send and recv

void mpi_t::remote_finalize(  )
{
    for( auto * p_b : { & buf_send, & buf_recv } )
        for( auto & [ tag, b ] : ( * p_b ) )
            b.free( *  p_dev ) ;
    return;
}

void mpi_t::remote_synchronize(  )
{
    if( buf_send.size(  ) < 1 && buf_recv.size(  ) < 1 )
        return;
    for( auto & [ r,  buf ] : buf_send )
        buf.sync( * p_dev ) ;        
    reqs_d.wait(  );
    for( auto & [ r , buf ] : buf_recv )
        buf.sync( * p_dev ) ;
    return;
}

////////////////////////////////////////////////////////////
// Deal with requests

void mpi_t::wait_all_d(  )
{
    if( wait_background )
    {
        std::thread th( [ & ] (  )
        {
            remote_synchronize(  );
        }   );
        local_synchronize     (  );
        return th.join        (  );
    }
    remote_synchronize        (  );
    return local_synchronize  (  );
}

void mpi_t::wait_all_h(  )
{
    reqs_h.wait  (  );
    if( local_optimize )
        local_synchronize(  );
    this->barrier(  );    
}

////////////////////////////////////////////////////////////
// Device-side aysnc comm

void mpi_t::isend_d
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag ,
  const device:: stream_t      & stream )
{
    const int l_rank = lrank_of( rank );
    if( local_optimize && l_rank >= 0 )            
        local_send[ l_rank ]
            ( data, size, tag, * p_dev, stream );
    else
    {
        auto * pb  = & buf_send[ rank ]
                     ( data, size, tag, * p_dev, stream );
        p_dev ->a_cp ( pb->h, pb->d, pb->n, stream );
        p_dev->launch_host ( stream, [ = ] (  )           
        {
            auto * pr = this->reqs_d.get(  );
            MPI_Isend( pb->h, pb->n, MPI_CHAR, rank,
                       tag, comm, pr );
            if( ( * pr ) == 0 )
                throw std::runtime_error( "" );            
        }   );
    }
    return;
}

void mpi_t::irecv_d
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag ,
  const device:: stream_t      & stream )
{
    const int l_rank = lrank_of( rank );
    if( local_optimize && l_rank >= 0 )
        local_recv[ l_rank ]
            ( data, size, tag, * p_dev, stream );
    else
    {
        auto & b = buf_recv[ rank ]
                     ( data, size, tag, * p_dev, stream );
        auto * pr = reqs_d.get(  ) ;
        
        MPI_Irecv( b.h, b.n, MPI_CHAR, rank,
                   tag, comm, pr );
        if( ( * pr ) == 0 )
            throw std::runtime_error( "" );
    }
    return;
}

////////////////////////////////////////////////////////////
// Host-side async comm

void mpi_t::isend_h
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag )
{
    const int l_rank = lrank_of( rank );
    if( local_optimize && l_rank >= 0 )
    {
        local_send[ l_rank ]
            ( data, size, tag, * p_dev, 0, true );
        return;
    }
    auto * pr = reqs_h.get(  ) ;    
    MPI_Isend( data, size, MPI_CHAR, rank, tag, comm, pr );
    if( ( * pr ) == 0 )
        throw std::runtime_error( "" );    
    return;
}

void mpi_t::irecv_h
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag )
{
    const int l_rank = lrank_of( rank );
    if( local_optimize && l_rank >= 0 )
    {
        local_recv[ l_rank ]
            ( data, size, tag, * p_dev, 0, true );
        return;
    }
    auto * pr = reqs_h.get(  ) ;        
    MPI_Irecv( data, size, MPI_CHAR, rank, tag, comm, pr );
    if( ( * pr ) == 0 )
        throw std::runtime_error( "" );        
    return;
}

////////////////////////////////////////////////////////////
// Host-side broadcast

void mpi_t::bcast( void * p, const size_t & size )
{
    if( n_rank(  ) > 1 )
        MPI_Bcast( p, size, MPI_CHAR, root, comm );
    return;
}

void mpi_t::barrier(  )
{
    MPI_Barrier( comm );
    return;
}

////////////////////////////////////////////////////////////
// Host-side reduction

void mpi_t::reduce_all_ker
( void  * p,  const std::type_info & t ,
  const operation_t & o, const int & n,
  const bool & async )
{
    if( n_rank(  ) <= 1 )
        return;
    MPI_Op       op( MPI_OP_NULL       );
    MPI_Datatype tp( MPI_DATATYPE_NULL );

    switch( o )
    {
    case min:
        op = MPI_MIN;
        break;
    case max:
        op = MPI_MAX;
        break;
    case sum:
        op = MPI_SUM;
        break;
    default:
        break;
    };
    if     ( t == typeid(    int ) )
        tp = MPI_INT;
    else if( t == typeid(  float ) )
        tp = MPI_FLOAT;
    else if( t == typeid( double ) )
        tp = MPI_DOUBLE;
    else if( t == typeid( size_t ) )
        tp = MPI_UNSIGNED_LONG;
    else
        throw std::runtime_error( "Undefined reduce type" );
    
    if( async )
        MPI_Iallreduce( MPI_IN_PLACE, p, n, tp, op, comm,
                        & reduce_req );
    else
        MPI_Allreduce( MPI_IN_PLACE, p, n, tp, op, comm );
    return;
}

void mpi_t::reduce_all_finish(  )
{
    MPI_Wait( & reduce_req, MPI_STATUSES_IGNORE );
    return;
}

};                              // namespace communicate
#endif // __MPI__
