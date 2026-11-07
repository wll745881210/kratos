#pragma once

#include "../../src/types.h"
#include "../../src/mesh/mesh.h"
#include "../../src/modules/dynamics/indices.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  The non - finite guard.  A cell gone NaN / Inf must dump
//  the state and stop,  not sail on:  in the failure that
//  motivated it dt doubled every cycle while the whole field
//  vacuumed to 1e-21,  so the dt_collapse fuse never fired.
//  Enrolled as the LAST module of the step order,  parasitic
//  on the evolution module:  the sweep reads the very state
//  the next dt is taken from.  On a hit every rank prints its
//  first bad cell,  then all ranks throw together --  cycle.cpp's
//  catch then dumps ( final_output ) and exits,  so the .bin
//  is the crash forensics.

template< class T, class = void >
struct guard_has_fld : std::false_type { };
template< class T >
struct guard_has_fld< T, std::void_t
    < decltype( std::declval< T & >(  ).fld.bf ) > >
    : std::true_type { };

////////////////////////////////////////////////////////////
//  Device helpers.

//  v - v == 0  fails for NaN ( a - a is NaN )  and for
//  +/- Inf ( a - a is NaN there too ),  and holds for
//  every finite value incl. denormals.
__device__ __forceinline__ bool guard_bad
( const type::float2_t & v )
{
    return ! ( v - v == 0 );
};

//  acc layout:  [ 0 ] = bad - cell count;  [ 1..3 ] =
//  i + 1 of a first bad cell ( 0 = none );  [ 4 ] =
//  field code ( n >= 0:  hyd.u slot n;  -1 - a:  fld.bf
//  axis a );  [ 5 ] = the offending value.   The  1..5
//  writes race - gate on acc[ 1 ] --  the winner is AN
//  arbitrary bad cell,  which is all the forensics
//  needs.
__device__ __forceinline__ void guard_report
( type::float_t * p_acc, const type::idx_t & i,
  const int & code, const type::float2_t & v )
{
    atomicAdd ( p_acc, type::float_t( 1 ) );
    if( p_acc[ 1 ] != 0 )
        return;
    atomicExch( p_acc + 1, type::float_t( i[ 0 ] + 1 ) );
    atomicExch( p_acc + 2, type::float_t( i[ 1 ] + 1 ) );
    atomicExch( p_acc + 3, type::float_t( i[ 2 ] + 1 ) );
    atomicExch( p_acc + 4, type::float_t( code ) );
    atomicExch( p_acc + 5, type::float_t( v ) );
    return;
};

template< class bdt_T >
__global__ void disk_guard_k
( const bdt_T b_d, type::float_t * p_acc )
{
    const auto & geo( b_d.geo );
    const int n0( geo.n_ceff[ 0 ] );
    const int n1( geo.n_ceff[ 1 ] );
    const int n_u( b_d.hyd.u.n_int );
    const int t ( blockIdx.x * blockDim.x + threadIdx.x );
    if( t >= n0 * n1 * geo.n_ceff[ 2 ] )
        return;
    type::idx_t i;
    i[ 0 ] =   t % n0;
    i[ 1 ] = ( t / n0 ) % n1;
    i[ 2 ] =   t / ( n0 * n1 );
    //  The CT face field is written before the conserved
    //  rows catch up ( funnel_bath's finding ),  so both
    //  lattices are swept.
    const auto * u( b_d.hyd.u.at( i ) );
    for( int n = 0; n < n_u; ++ n )
        if( guard_bad( u[ n ] ) )
            return guard_report( p_acc, i, n, u[ n ] );
    if constexpr( guard_has_fld< bdt_T >::value )
    {
        const auto * bf( b_d.fld.bf.at( i, 0 ) );
        for( int a = 0; a < 3; ++ a )
            if( guard_bad( bf[ a ] ) )
                return guard_report( p_acc, i, -1 - a,
                                     bf[ a ] );
    }
    return;
};

template< class bdt_T > __host__
auto disk_guard_rsc( const bdt_T & b_d )
{
    const int n_c( b_d.geo.n_ceff.prod(  ) );
    const int n_t( std::min( n_c, 256 ) );
    const dim3  n_b( ( n_c + n_t - 1 ) / n_t );
    return std::make_tuple( n_b, dim3( n_t ), 0 );
};

////////////////////////////////////////////////////////////
//  The module.

template< class prx_T >
struct disk_guard_t : mesh::mod_base_t
{
    using bdt_t = typename prx_T::bdt_t;

    int               on;
    //  check = 0:  the diagnostic A/B mode --  the sweep
    //  kernels are skipped,  only the cycle - end
    //  sync_all_streams remains ( the stream - race
    //  discriminator:  a run healed by  check = 0  is a
    //  missing - barrier problem,  not bad data ).
    int            check;
    type::float_t * d_acc;

    __host__ disk_guard_t(  )
        : mod_base_t(  ), on( 1 ), check( 1 ),
          d_acc( nullptr )
    {  };

    __host__ virtual void read
    ( const input & args ) override
    {
        on    = args.get< int >( "guard", "on",    1 );
        check = args.get< int >( "guard", "check", 1 );
        this->template enroll_bdk
            < mesh::boundary::keeper_t > (  );
        return;
    };

    __host__ virtual void init
    ( const input & args, mesh::mesh_t & mesh ) override
    {
        mod_base_t::init( args, mesh );
        if( on )
            d_acc = this->p_dev->template
                malloc< type::float_t >( 6 );
        return;
    };

    __host__ virtual void finalize
    ( mesh::mesh_t & mesh ) override
    {
        if( d_acc != nullptr )
            this->p_dev->free( d_acc );
        return mod_base_t::finalize( mesh );
    };

    __host__ virtual void step
    ( mesh::mesh_t & mesh ) override
    {
        if( ! on )
            return;
        if( ! check )
        {
            //  Legacy - default - stream memset:  an
            //  implicit device - wide barrier for the
            //  blocking streams ( the diagnostic knob
            //  for the stream - race hunt ).
            auto & dev( * this->p_dev );
            dev.f_mset( d_acc, 0,
                        6 * sizeof( type::float_t ), 0 );
            dev.sync_all_streams(  );
            return;
        }
        if( d_acc == nullptr )
            return;
        auto & dev( * this->p_dev );
        dev.f_mset( d_acc, 0,
                    6 * sizeof( type::float_t ), 0 );
        for( auto & d : ( * this ) )
        {
            const auto b_d( prx_T::d( d.d(  ) ) );
            dev.launch( disk_guard_k< decltype( b_d ) >,
                        disk_guard_rsc( b_d ),
                        d.stream, b_d, d_acc );
        }
        dev.sync_all_streams(  );
        type::float_t h[ 6 ];
        dev.cp( h, d_acc, 6 );
        if( h[ 0 ] > 0 )
        {
            auto & cyc( * mesh.p_cyc );
            std::cout << "GUARD  rank "
                      << mesh.p_com->rank(  )
                      << ":  " << h[ 0 ]
                      << " bad cells,  first at ( "
                      << h[ 1 ] - 1 << ", "
                      << h[ 2 ] - 1 << ", "
                      << h[ 3 ] - 1 << " )  "
                      << slot_name( h[ 4 ] )
                      << " = " << h[ 5 ]
                      << ",  cycle = " << cyc.i_cycle
                      << ",  t = " << cyc.t
                      << std::endl;
        }
        mesh.p_com->reduce_all( h[ 0 ], comm::sum );
        if( h[ 0 ] > 0 )
            throw std::runtime_error
                ( "GUARD:  non - finite cells present;"
                  "  dumping ( final_output ) and"
                  "  stopping.\n" );
        return;
    };

private:
    __host__ std::string slot_name
    ( const type::float_t & code_f ) const
    {
        const int code = int( code_f );
        if( code < 0 )
        {
            const char * ax[ 3 ] = { "r", "th", "phi" };
            return std::string( "fld.bf " )
                 + ax[ -1 - code ];
        }
        //  MHD rows:  0 rho,  1 ene,  2..4 mom,  5..7
        //  bcc,  then the species;  hydro rows drop bcc.
        constexpr int n_hvar
            ( guard_has_fld< bdt_t >::value ? 8 : 5 );
        if( code >= n_hvar )
            return "species "
                 + std::to_string( code - n_hvar );
        if( code == idx::rho )
            return "rho";
        if( code == idx::ene )
            return "ene";
        if( code >= idx::mom && code < idx::mom + 3 )
        {
            const char * ax[ 3 ] = { "r", "th", "phi" };
            return std::string( "mom " )
                 + ax[ code - idx::mom ];
        }
        const char * ax[ 3 ] = { "r", "th", "phi" };
        return std::string( "bcc " )
             + ax[ code - idx::bcc ];
    };
};

};                              // namespace interm
