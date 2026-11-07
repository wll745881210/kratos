#pragma once

#include "../../../src/modules/dynamics/hydro/reconstruct/plm.h"

namespace extension::immersed::recon
{
////////////////////////////////////////////////////////////
// PLM limiter variants, derived from the base (van Leer
// harmonic) reconstruction.  plm_t dispatches uniform() /
// general() through the CRTP self(), so a derived class only
// replaces the limiter; the stencil, geometry loading and the
// kernel interface are all inherited.
//
// Conventions (see plm_t): face i lies between cells i-1 and i;
// q_ll..q_rr hold q(i-2..i+1).  The left state extrapolates from
// cell i-1 over dxfr (face minus cell centre); the right state
// from cell i over dxfl.  On uniform meshes load_geo() is
// skipped, so uniform() works on raw differences with the fixed
// half-cell offset.
//
// Both limiters keep the interface values within the
// neighbouring cell averages (TVD in 1-D); minmod is the most
// dissipative of the family, the monotonized-central (MC) one is
// the least dissipative that is still TVD.

namespace plm = dynamics::hydro::reconstruct;

struct plm_minmod_t : public plm::plm_t< plm_minmod_t >
{
    using float_t = type::float_t;

    template< class f_T > __device__ __forceinline__
    static f_T minmod( const f_T & a, const f_T & b )
    {
        if( a * b <= 0 )
            return f_T( 0 );
        return __abs1( a ) < __abs1( b ) ? a : b;
    };

    template< class dat_T, class bdt_T > __device__
    void uniform( dat_T & d, const int & i, const int & n,
                  const bdt_T & bdt ) const
    {
        auto * ql = d.ql.at( i );
        auto * qr = d.qr.at( i );
        ql[ n ] = q_l + float_t( 0.5 )
                * minmod( q_l - q_ll, q_r - q_l );
        qr[ n ] = q_r - float_t( 0.5 )
                * minmod( q_r - q_l, q_rr - q_r );
        return;
    };

    template< class dat_T, class bdt_T > __device__
    void general( dat_T & d, const int & i, const int & n,
                  const bdt_T & bdt ) const
    {
        auto * ql = d.ql.at( i );
        auto * qr = d.qr.at( i );
        ql[ n ] = q_l + dxfr * minmod( ( q_l - q_ll ) * dxv_l_i,
                                       ( q_r - q_l  ) * dxv_c_i );
        qr[ n ] = q_r - dxfl * minmod( ( q_r - q_l  ) * dxv_c_i,
                                       ( q_rr - q_r ) * dxv_r_i );
        return;
    };
};

////////////////////////////////////////////////////////////

struct plm_mc_t : public plm::plm_t< plm_mc_t >
{
    using float_t = type::float_t;

    template< class f_T > __device__ __forceinline__
    static f_T minmod3( const f_T & a, const f_T & b,
                        const f_T & c )
    {
        if( a * b <= 0 || b * c <= 0 || a * c <= 0 )
            return f_T( 0 );
        const auto sa = __abs1( a );
        const auto sb = __abs1( b );
        const auto sc = __abs1( c );
        const auto m  = sa < sb ? ( sa < sc ? sa : sc )
                                : ( sb < sc ? sb : sc );
        return a > 0 ? m : -m;
    };

    template< class dat_T, class bdt_T > __device__
    void uniform( dat_T & d, const int & i, const int & n,
                  const bdt_T & bdt ) const
    {
        auto * ql = d.ql.at( i );
        auto * qr = d.qr.at( i );
        ql[ n ] = q_l + float_t( 0.5 )
                * minmod3( ( q_r - q_ll ) * float_t( 0.5 ),
                           ( q_l - q_ll ) * 2, ( q_r - q_l ) * 2 );
        qr[ n ] = q_r - float_t( 0.5 )
                * minmod3( ( q_rr - q_l ) * float_t( 0.5 ),
                           ( q_r - q_l ) * 2, ( q_rr - q_r ) * 2 );
        return;
    };

    template< class dat_T, class bdt_T > __device__
    void general( dat_T & d, const int & i, const int & n,
                  const bdt_T & bdt ) const
    {
        const auto * xv = d.xv.at( i );
        auto * ql = d.ql.at( i );
        auto * qr = d.qr.at( i );
        // cell i-1 (left state): backward/forward/central slopes
        const auto s_bk = ( q_l - q_ll ) * dxv_l_i;
        const auto s_fw = ( q_r - q_l  ) * dxv_c_i;
        const auto s_ct = ( q_r - q_ll ) / ( xv[ 0 ] - xv[ -2 ] );
        ql[ n ] = q_l + dxfr * minmod3( s_ct, 2 * s_bk, 2 * s_fw );
        // cell i (right state)
        const auto t_fw = ( q_rr - q_r ) * dxv_r_i;
        const auto t_ct = ( q_rr - q_l ) / ( xv[ 1 ] - xv[ -1 ] );
        qr[ n ] = q_r - dxfl * minmod3( t_ct, 2 * s_fw, 2 * t_fw );
        return;
    };
};

};                 // namespace extension::immersed::recon
