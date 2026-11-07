#pragma once

#include "../../../src/device/device.h"
#include "../../../src/modules/chemistry/types.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace extension::deeponet
{
//////////////////////////////////////////////////////////
// DeepONet model "model.dnn" v2 ( see python/export.py ):
//
//   magic  "DON1"                          ( 4 bytes  )
//   u32    version(=2) n_spe n_hv n_in n_out
//          n_basis n_lay_br n_lay_tr act(=1) flags
//   f32    T_lo T_hi dt_min dt_max lam
//   f32[ n_in ]   mu_in   sd_in
//   f32[ n_out ]  mu_out  sd_out
//   f32    mu_dt  sd_dt
//   n_spe x { u8 len + name }, block padded to 4
//   f32[ n_hv ]   hv_eV
//   n_lay_br x { u32 nin u32 nout
//                f32[ nin*nout ] W  ( [ nin x nout ] row-major )
//                f32[ nout ]    b }
//   n_lay_tr x { same }
//   f32[ n_out ]  bias_out
//   magic  "DON1"
//   u32    checksum: additive sum of ALL preceding words
//          ( including both magics ) & 0xFFFFFFFF
//
// Inference convention ( must match python/train.py ), v2:
//   inputs  z-scored: eps -> log10 ( specific internal
//           energy, cgs erg/g, kinetic-subtracted ),
//           rho -> log10( code units ),
//           X -> Box-Cox( lam ), F -> log10( code units )
//   trunk   z-scored log10( dt ) in CODE units
//   labels  z-scored: X' -> Box-Cox( lam ), eps' -> log10;
//           increment mode ( flags bit1 ): dBC( X ),
//           d( lg eps )
//   The solver commits u[ ene ] from eps' EXACTLY ( no
//   cv*mu-constant T-ratio approximation ) and derives
//   T_hyd via the eos.temp fixed-point ( nn_eos_T ).
//   combine y[ z ] = bias_out[ z ]
//                  + sum_k B[ z*n_basis + k ] * Tk[ k ]
//////////////////////////////////////////////////////////

constexpr int nn_max_lay = 16;

//////////////////////////////////////////////////////////
// Layer descriptor ( POD, device-visible )

struct nn_lay_t
{
    int n_in;
    int n_out;
    int off_w;  // float offset of W in w_dev
    int off_b;  // float offset of b in w_dev
};

//////////////////////////////////////////////////////////
// Model handle ( POD, embedded in the solver and copied
// into kernel arguments ). All statistics, the gate, and
// the weights live in one flat device buffer `w_dev',
// described by the offsets below.

struct nn_t
{
    // Dimensions
    bool   ready;
    int    n_spe;
    int    n_hv;
    int    n_in;
    int    n_out;
    int    n_basis;
    int    n_lay_br;
    int    n_lay_tr;
    int    n_hdn;    // max hidden width ( shared-mem sizing )
    int    n_scr_f;  // device shared floats needed ( ex. flags )
    bool   inc;      // flags bit1: increment targets ( v2 )

    // Envelope ( gate ): T in [ K ], dt in [ s ]
    float  T_lo;
    float  T_hi;
    float  dt_min;
    float  dt_max;

    // Float offsets into w_dev
    int    off_mu_in;
    int    off_sd_in;
    int    off_mu_out;
    int    off_sd_out;
    int    off_mu_dt;
    int    off_sd_dt;
    int    off_lam;
    int    off_bias;

    // Device buffers
    nn_lay_t * br_lay;
    nn_lay_t * tr_lay;
    float    * w_dev;

    // Host-side default
    nn_t( )
        : ready ( false ), n_spe ( 0 ), n_hv ( 0 ),
          n_in ( 0 ), n_out ( 0 ), n_basis ( 0 ),
          n_lay_br ( 0 ), n_lay_tr ( 0 ),
          n_hdn ( 0 ), n_scr_f ( 0 ),
          T_lo ( 0 ), T_hi ( 0 ),
          dt_min ( 0 ), dt_max ( 0 ),
          off_mu_in ( 0 ), off_sd_in ( 0 ),
          off_mu_out ( 0 ), off_sd_out ( 0 ),
          off_mu_dt ( 0 ), off_sd_dt ( 0 ),
          off_lam ( 0 ), off_bias ( 0 ),
          br_lay ( nullptr ), tr_lay ( nullptr ),
          w_dev ( nullptr )
    { return; };
};

//////////////////////////////////////////////////////////
// Host-side loader. `species' is the chemistry module's
// species list ( order defines the u[] slot order and must
// match the model ). Throws std::runtime_error on any
// mismatch. The radiation band count ( n_hv ) is validated
// later, in solver_t::integrate, where it is known.

__host__ inline void nn_load
( const std::string & path, device::base_t & dev,
  const ::chemistry::sp_vec_t & species, nn_t & nn )
{
    std::ifstream ifs( path, std::ios::binary );
    if( ! ifs )
        throw std::runtime_error
            ( "[deeponet] cannot open model file: " + path );
    std::vector< char > raw
        ( ( std::istreambuf_iterator< char >( ifs ) ),
          ( std::istreambuf_iterator< char >( ) ) );
    if( raw.size( ) < 64 || raw.size( ) % 4 != 0 )
        throw std::runtime_error
            ( "[deeponet] malformed model file: " + path );

    const char magic[ 4 ] = { 'D', 'O', 'N', '1' };
    if( std::memcmp( raw.data( ), magic, 4 ) != 0
     || std::memcmp( raw.data( ) + raw.size( ) - 8, magic, 4 ) != 0 )
        throw std::runtime_error
            ( "[deeponet] bad magic in model file: " + path );

    // Checksum over all words but the last
    {
        uint32_t sum = 0;
        const size_t n_w = raw.size( ) / 4 - 1;
        for( size_t i = 0; i < n_w; ++ i )
        {
            uint32_t w;
            std::memcpy( & w, raw.data( ) + i * 4, 4 );
            sum += w;
        }
        uint32_t chk;
        std::memcpy( & chk, raw.data( ) + raw.size( ) - 4, 4 );
        if( sum != chk )
            throw std::runtime_error
                ( "[deeponet] checksum mismatch in: " + path );
    }

    // Byte cursor
    size_t o = 4;
    auto u32 = [ & ] ( )
    {
        uint32_t x;
        std::memcpy( & x, raw.data( ) + o, 4 );
        o += 4;
        return int( x );
    };
    auto f32 = [ & ] ( )
    {
        float x;
        std::memcpy( & x, raw.data( ) + o, 4 );
        o += 4;
        return x;
    };

    const int version = u32( );
    nn.n_spe     = u32( );
    nn.n_hv      = u32( );
    nn.n_in      = u32( );
    nn.n_out     = u32( );
    nn.n_basis   = u32( );
    nn.n_lay_br  = u32( );
    nn.n_lay_tr  = u32( );
    const int act    = u32( );
    const int flags  = u32( );
    nn.T_lo  = f32( ); nn.T_hi  = f32( );
    nn.dt_min = f32( ); nn.dt_max = f32( );
    const float lam = f32( );

    if( version != 2 || act != 1 || ( flags & ~2 ) != 0 )
        throw std::runtime_error
            ( "[deeponet] unsupported model variant ( want "
              "version 2, eps-semantics ) in: " + path );
    nn.inc = ( flags & 2 ) != 0;
    if( nn.n_lay_br < 1 || nn.n_lay_br > nn_max_lay
     || nn.n_lay_tr < 1 || nn.n_lay_tr > nn_max_lay )
        throw std::runtime_error
            ( "[deeponet] layer count out of range in: " + path );
    if( nn.n_spe != int( species.size( ) ) )
        throw std::runtime_error
            ( "[deeponet] species count mismatch: model "
              + std::to_string( nn.n_spe ) + " vs chemistry "
              + std::to_string( species.size( ) ) );

    // Stats block ( file order, staged at the buffer head ):
    // [ gate 4 ][ lam ][ mu/sd_in ][ mu/sd_out ][ mu_dt sd_dt ]
    std::vector< float > buf;
    buf.reserve( 40 + nn.n_in * 2 + nn.n_out * 4 + 8 );
    auto push = [ & ] ( const float & v )
    { buf.push_back( v ); return; };

    push( nn.T_lo );  push( nn.T_hi );
    push( nn.dt_min ); push( nn.dt_max );
    push( lam );
    nn.off_lam = 4;
    nn.off_mu_in  = int( buf.size( ) );
    for( int i = 0; i < nn.n_in;  ++ i ) push( f32( ) );
    nn.off_sd_in  = int( buf.size( ) );
    for( int i = 0; i < nn.n_in;  ++ i ) push( f32( ) );
    nn.off_mu_out = int( buf.size( ) );
    for( int i = 0; i < nn.n_out; ++ i ) push( f32( ) );
    nn.off_sd_out = int( buf.size( ) );
    for( int i = 0; i < nn.n_out; ++ i ) push( f32( ) );
    nn.off_mu_dt = int( buf.size( ) ); push( f32( ) );
    nn.off_sd_dt = int( buf.size( ) ); push( f32( ) );

    // Species names ( order must match the u[] slots )
    std::vector< std::string > names;
    for( int i = 0; i < nn.n_spe; ++ i )
    {
        const int n = int( uint8_t( raw[ o ] ) );  ++ o;
        names.emplace_back( raw.data( ) + o, n );
        o += n;
    }
    o += ( 4 - o % 4 ) % 4;    // name-block padding
    for( int i = 0; i < nn.n_spe; ++ i )
        if( names[ i ] != species.at( i ).name )
            throw std::runtime_error
                ( "[deeponet] species order mismatch at slot "
                  + std::to_string( i ) + ": model '" + names[ i ]
                  + "' vs chemistry '" + species.at( i ).name + "'" );

    o += nn.n_hv * 4;    // hv_eV ( unused on device )

    std::vector< nn_lay_t > br( nn.n_lay_br );
    std::vector< nn_lay_t > tr( nn.n_lay_tr );
    nn.n_hdn = 0;

    const auto rd_layers = [ & ] ( std::vector< nn_lay_t > & lay,
                                    const int & n_lay )
    {
        for( int l = 0; l < n_lay; ++ l )
        {
            lay[ l ].n_in  = u32( );
            lay[ l ].n_out = u32( );
            if( lay[ l ].n_in < 1 || lay[ l ].n_out < 1
             || lay[ l ].n_in * lay[ l ].n_out > ( 1 << 24 ) )
                throw std::runtime_error
                    ( "[deeponet] bad layer dims in: " + path );
            lay[ l ].off_w = int( buf.size( ) );
            for( int i = 0; i < lay[ l ].n_in
                             * lay[ l ].n_out; ++ i )
                push( f32( ) );
            lay[ l ].off_b = int( buf.size( ) );
            for( int i = 0; i < lay[ l ].n_out; ++ i )
                push( f32( ) );
            if( l + 1 < n_lay )
                nn.n_hdn = std::max( nn.n_hdn, lay[ l ].n_out );
        }
        return;
    };
    rd_layers( br, nn.n_lay_br );
    rd_layers( tr, nn.n_lay_tr );

    // Combine bias ( after the layers in the file )
    nn.off_bias = int( buf.size( ) );
    for( int i = 0; i < nn.n_out; ++ i ) push( f32( ) );

    // Structural checks
    if( br.front( ).n_in  != nn.n_in
     || br.back( ).n_out  != nn.n_out * nn.n_basis
     || tr.front( ).n_in  != 1
     || tr.back( ).n_out  != nn.n_basis )
        throw std::runtime_error
            ( "[deeponet] layer topology mismatch in: " + path );
    if( nn.n_in != 2 + nn.n_spe + nn.n_hv
     || nn.n_out != nn.n_spe + 1 )
        throw std::runtime_error
            ( "[deeponet] dimension mismatch in: " + path );
    for( const auto & v : buf )
        if( ! std::isfinite( v ) )
            throw std::runtime_error
                ( "[deeponet] non-finite parameter in: " + path );

    // Shared scratch ( floats, excl. trailing flag ints ):
    // x_in, X, act1, act2, B, Tk, zout, zdt
    nn.n_scr_f = nn.n_in + nn.n_spe + 2 * nn.n_hdn
               + nn.n_out * nn.n_basis + nn.n_basis
               + nn.n_out + 1;

    // Upload
    nn.w_dev = dev.malloc_device< float >( buf.size( ) );
    dev.cp( nn.w_dev, buf.data( ), buf.size( ) );
    nn.br_lay = dev.malloc_device< nn_lay_t >( nn.n_lay_br );
    dev.cp( nn.br_lay, br.data( ), nn.n_lay_br );
    nn.tr_lay = dev.malloc_device< nn_lay_t >( nn.n_lay_tr );
    dev.cp( nn.tr_lay, tr.data( ), nn.n_lay_tr );
    nn.ready = true;

    std::printf
        ( "[deeponet] loaded %s: n_spe=%d n_hv=%d n_in=%d "
          "n_out=%d p=%d br=[%dx%d] tr=%d lay, n_hdn=%d, "
          "T=[%.3g,%.3g] K dt=[%.3g,%.3g] s lam=%.3g\n",
          path.c_str( ), nn.n_spe, nn.n_hv, nn.n_in, nn.n_out,
          nn.n_basis, nn.n_lay_br, br.front( ).n_out, nn.n_lay_tr,
          nn.n_hdn, nn.T_lo, nn.T_hi, nn.dt_min, nn.dt_max, lam );
    return;
};

//////////////////////////////////////////////////////////
// Host-side release

__host__ inline void nn_free
( device::base_t & dev, nn_t & nn )
{
    if( nn.w_dev  != nullptr ) dev.free_device( nn.w_dev  );
    if( nn.br_lay != nullptr ) dev.free_device( nn.br_lay );
    if( nn.tr_lay != nullptr ) dev.free_device( nn.tr_lay );
    nn.w_dev  = nullptr;
    nn.br_lay = nullptr;
    nn.tr_lay = nullptr;
    nn.ready  = false;
    return;
};
};                       // namespace extension::deeponet
