#pragma once

#include "../../src/utilities/phys/constants.h"
#include "../../src/utilities/phys/unit.h"
#include "../../src/modules/dynamics/indices.h"
#include "../../src/types.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  The polar - cone toroidal - field bath.
//  The cone  sin( th ) < s_th  fills with gas
//  and winds up a nearly PURE toroidal field
//  ( B_phi / |B| ~ 0.99,  v_A -> 200 ),  which sets the
//  global ideal CFL and stalls the run.  The bath caps
//  v_A there by damping  B_phi  on the phi - faces:  the
//  CT divergence reads only the r / theta faces,  so this
//  is  div - B - exact  ( no projection needed ),  and the
//  removed magnetic energy turns into heat by itself  (
//  the EOS subtracts  |bf|^2 / 2  from the total energy ).
//
//  A non - finite cell is repaired from the nearest
//  finite theta neighbour,  so one bad cell can no longer
//  poison the implicit multigrid  ( its 1e30 sentinel ).

struct funnel_bath_t
{
    ////////// Types //////////
    using float_t  = type::  float_t;
    using float2_t = type::float2_t;

    ////////// Data //////////
    int      on;
    float2_t s_th;      // sin( th ) of the cone edge
    int      w_cell;    // the edge ramp, in theta cells
    float2_t va_max;    // the target Alfven speed
    //  mode:  0 = damp the toroidal face field  (div - B
    //  exact,  cannot bound a POLOIDAL pile - up);
    //  1 = MASS LOADING  --  raise rho (with the same
    //  specific momentum and energy) until
    //  v_A = |B| / sqrt( rho ) <= va_max.  Touches no face
    //  field at all,  so  div B  is untouched by
    //  construction.  The GRMHD practice  ( inject mass to
    //  keep the magnetisation below a ceiling;
    //  arXiv:2404.01471 ).
    int      mode;
    float2_t tau_b;     // the damping time, in dt
    int   repair;       // the L3 scrubber
    int   print_every;  // the budget report interval, steps

    ////////// Functions //////////
    __host__ funnel_bath_t(  )
    {
        on = 0;
        s_th = 0.5;
        w_cell = 6;
        va_max = 30;
        mode = 0;
        tau_b = 1;
        repair = 1;
        print_every = 200;
        return;
    };

    __host__ void read( const input & args )
    {
        on      = args.get< int >
                  ( "funnelbath", "on",          0 );
        s_th    = args.get< float2_t >
                  ( "funnelbath", "s_th",      0.5 );
        w_cell  = args.get< int >
                  ( "funnelbath", "w_cell",      6 );
        va_max  = args.get< float2_t >
                  ( "funnelbath", "va_max",     30 );
        mode    = args.get< int >
                  ( "funnelbath", "mode",        0 );
        tau_b   = args.get< float2_t >
                  ( "funnelbath", "tau_b",       1 );
        repair  = args.get< int >
                  ( "funnelbath", "repair",      1 );
        print_every = args.get< int >
                  ( "funnelbath", "print_every", 200 );
        return;
    };

    template< class geo_T, class idx_T >
    __device__ __forceinline__ float2_t wgt
    ( const geo_T & geo, const idx_T & i ) const
    {
        const auto sn( __abs1( __sin1
                      ( geo.x_cc( 1, i[ 1 ] ) ) ) );
        if( sn <= s_th )
            return 1;
        const auto s1
        ( s_th + w_cell * geo.dx_f( 1, i ) );
        if( sn >= s1 )
            return 0;
        const float2_t t( ( s1 - sn ) / ( s1 - s_th ) );
        return t * t * ( 3 - 2 * t );
    };
};

////////////////////////////////////////////////////////////
//  The per - cell kernel and its launch resource.

template< class bdt_T >
__global__ void funnel_bath_k
( const bdt_T b_d, const funnel_bath_t bath,
  type::float_t * p_acc )
{
    const auto & geo( b_d.geo );
    const int  n0( geo.n_ceff[ 0 ] );
    const int  n1( geo.n_ceff[ 1 ] );
    const int  n_u( b_d.hyd.u.n_int );
    const int  t ( blockIdx.x * blockDim.x + threadIdx.x );
    if( t >= n0 * n1 * geo.n_ceff[ 2 ] )
        return;
    type::idx_t i;
    i[ 0 ] =   t % n0;
    i[ 1 ] = ( t / n0 ) % n1;
    i[ 2 ] =   t / ( n0 * n1 );
    const auto w( bath.wgt( geo, i ) );
    if( w <= 0 )
        return;
    auto * u ( b_d.hyd.u.at( i ) );
    auto * bf( b_d.fld.bf.at( i, 0 ) );   // the post - step slice
    bool bad( false );
    for( int n = 0; n < n_u; ++ n )
        if( ! ( u[ n ] == u[ n ] ) )
            bad = true;
    //  The face field is written by the CT update directly,
    //  so it can go non - finite while  u  is still clean;
    //  c2p then poisons the next step.
    for( int a = 0; a < 3; ++ a )
        if( ! ( bf[ a ] == bf[ a ] ) )
            bad = true;
    if( bad && bath.repair )
    {
        //  Repair from the nearest finite theta neighbour,
        //  walking towards the midplane.
        const int dir( i[ 1 ] < n1 / 2 ? 1 : -1 );
        for( int k = 1; k <= 8; ++ k )
        {
            type::idx_t j( i );
            j[ 1 ] = i[ 1 ] + dir * k;
            if( j[ 1 ] < 0 || j[ 1 ] >= n1 )
                break;
            const auto * uj( b_d.hyd.u.at( j ) );
            bool ok( true );
            for( int n = 0; n < n_u; ++ n )
                if( ! ( uj[ n ] == uj[ n ] ) )
                    ok = false;
            if( ! ok )
                continue;
            for( int n = 0; n < n_u; ++ n )
                u[ n ] = uj[ n ];
            const auto * bj( b_d.fld.bf.at( j, 0 ) );
            for( int a = 0; a < 3; ++ a )
                bf[ a ] = bj[ a ];
            atomicAdd( p_acc + 2, type::float_t( 1 ) );
            return;
        }
    }
    const auto rho( u[ idx::rho ] );
    const auto b_2( bf[ 0 ] * bf[ 0 ] + bf[ 1 ] * bf[ 1 ]
                  + bf[ 2 ] * bf[ 2 ] );
    const auto va ( sqrt( b_2 / __max1
                        ( rho, type::float2_t( 1e-30 ) ) ) );
    if( ! ( va > bath.va_max ) )       // also skips a NaN
        return;
    //  tau_b counts  dt's  ( 1 = remove the excess every
    //  step ).  Scaling by  dt / tau_b  instead would make
    //  the damping  dt - proportional  --  1.7e-4 per step
    //  here,  which leaves the field free to keep growing.
    if( bath.mode == 1 )
    {
        //  Mass loading:  scale the whole conserved row by
        //  s  so that  rho -> |B|^2 / va_max^2.  Every slot
        //  scales together,  so  v,  T  and the composition
        //  are unchanged and only  v_A  drops.
        const auto s( b_2 / ( bath.va_max * bath.va_max
                              * rho ) );
        const auto f( type::float2_t( 1 )
                    + w * ( s - type::float2_t( 1 ) ) );
        if( f > 1 )
        {
            for( int n = 0; n < n_u; ++ n )
                u[ n ] = type::float_t( u[ n ] * f );
            atomicAdd( p_acc + 4, type::float_t
                ( geo.volume( i ) * rho * ( f - 1 ) ) );
            atomicAdd( p_acc + 1, type::float_t( 1 ) );
        }
        return;
    }
    const auto a( w * __min1( 1 / bath.tau_b,
                              type::float2_t( 1 ) ) );
    const auto f( bath.va_max / va );
    const auto b_old( bf[ 2 ] );
    bf[ 2 ] = type::float_t
        ( b_old * ( 1 - a + a * f ) );
    //  The EOS's dt reads  max( b_low, b_high )  on every
    //  axis,  and the  i2 = n_phi  slot is a SEPARATE
    //  periodic image of slot 0:  damp it too,  else the dt
    //  never sees the cap.
    if( i[ 2 ] == 0 )
    {
        type::idx_t j( i );
        j[ 2 ] = geo.n_ceff[ 2 ];
        b_d.fld.bf.at( j, 0 )[ 2 ] = bf[ 2 ];
    }
    //  The bath is a SINK:  the removed magnetic energy
    //  leaves the total energy too.  Left in  ene  it would
    //  heat the cone,  raising  c_s  and defeating the very
    //  CFL the bath is there to protect.
    const auto d_b2( type::float2_t
        ( b_old * b_old - bf[ 2 ] * bf[ 2 ] ) );
    u[ idx::ene ] -= type::float_t( 0.5 * d_b2 );
    atomicAdd( p_acc, type::float_t
        ( geo.volume( i ) * 0.5 * d_b2 ) );
    atomicAdd( p_acc + 1, type::float_t( 1 ) );
    //  How often the cap is ineffective  ( a poloidal -
    //  dominated cell:  damping  B_phi  cannot bound v_A ).
    const auto b_p( bf[ 0 ] * bf[ 0 ] + bf[ 1 ] * bf[ 1 ] );
    if( b_p > bath.va_max * bath.va_max * rho )
        atomicAdd( p_acc + 3, type::float_t( 1 ) );
    return;
};

template< class bdt_T > __host__
auto funnel_bath_rsc( const bdt_T & b_d )
{
    const int n_c( b_d.geo.n_ceff.prod(  ) );
    const int n_t( std::min( n_c, 256 ) );
    const dim3  n_b( ( n_c + n_t - 1 ) / n_t );
    return std::make_tuple( n_b, dim3( n_t ), 0 );
};

////////////////////////////////////////////////////////////
//  The budget accumulator  ( L3 ):  magnetic energy removed,
//  phi - faces damped,  cells repaired.

struct funnel_bath_acc_t
{
    using float_t = type::float_t;

    float_t * d_acc;
    double    cum[ 5 ];
    double    pre[ 5 ];
    int       n_step;

    __host__ funnel_bath_acc_t(  ) : d_acc( nullptr )
    {
        for( int k = 0; k < 5; ++ k )
            cum[ k ] = pre[ k ] = 0;
        n_step = 0;
        return;
    };

    __host__ void reset( device::base_t & dev )
    {
        if( d_acc == nullptr )
            d_acc = dev.template malloc< float_t >( 5 );
        dev.f_mset( d_acc, 0, 5 * sizeof( float_t ), 0 );
        for( int k = 0; k < 5; ++ k )
            cum[ k ] = pre[ k ] = 0;
        n_step = 0;
        return;
    };

    __host__ void poll( device::base_t & dev, const int every )
    {
        if( d_acc == nullptr )
            return;
        ++ n_step;
        if( every <= 0 || n_step % every != 0 )
            return;
        float_t h[ 5 ];
        dev.cp( h, d_acc, 5 );
        for( int k = 0; k < 5; ++ k )
            cum[ k ] = h[ k ];
        std::cout << "FUNNELBATH  dE_mag = " << cum[ 0 ]
                  << "  ( +" << cum[ 0 ] - pre[ 0 ] << " )"
                  << "  damped = " << cum[ 1 ]
                  << "  ( +" << cum[ 1 ] - pre[ 1 ] << " )"
                  << "  repaired = " << cum[ 2 ]
                  << "  pol_in_cone = " << cum[ 3 ]
                  << "  mass_in = " << cum[ 4 ]
                  << std::endl;
        for( int k = 0; k < 5; ++ k )
            pre[ k ] = cum[ k ];
        return;
    };
};

};                              // namespace interm
