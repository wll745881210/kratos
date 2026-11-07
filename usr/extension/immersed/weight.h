#pragma once

#include "../../../src/modules/dynamics/indices.h"
#include "../../../src/types.h"

namespace extension::immersed
{
////////////////////////////////////////////////////////////
// Interpolation and reflection weights for one image cell

using type::float_t;
using type::coord_t;

struct weight_t
{
    ////////// Types //////////
    struct ws_t
    {
        int       i[ 3 ];
        float_t   w_s;
    };

    ////////// Data //////////
    int     n_wght;
    ws_t      w[ 8 ];
    coord_t   norm;
    coord_t   v_wall;
    int       bc;             // 0 = slip (inviscid reflect)
                              // 1 = no-slip (Dirichlet wall velocity)
    float_t   T_wall;         // > 0 : isothermal wall (p = rho T)
    float_t   phi0    = 0;    // SDF at the band cell (negative inside)
    float_t   nu_lam  = 0;    // > 0 : log-law wall model, kinematic
                              // viscosity [m^2/s]

    ////////// Application //////////
    // src / tgt hold PRIMITIVE variables ( rho, p, v ) -- see
    // eos::adiabatic_t::c2p -- so idx::mom + a is a VELOCITY.
    // Reconstruct the primitives at cell i0 from its stencil (read
    // from the PRE-conditioning snapshot src) and write the result
    // to tgt, then enforce the wall condition:
    //   slip   : reflect the normal velocity about the wall velocity;
    //   no-slip: set the full velocity to v_wall (Dirichlet).
    // An optional isothermal wall overrides the pressure to rho*T.
    // The two-buffer form makes the pass Jacobi (order-free), which
    // preserves the mirror symmetry of symmetric bodies; an
    // in-place sweep would be Gauss-Seidel order-dependent.
    template< class i_T, class dat_T > __device__ __forceinline__
    void operator(  ) ( const i_T & i0, const dat_T & src,
                        const dat_T & tgt ) const
    {
        auto * q_tgt = tgt.at( i0 );
        for( int n = 0; n < src.n_int; ++ n )
            q_tgt[ n ] = 0;
        for( int j = 0; j < n_wght; ++ j )
        {
            const auto & w_s = w[ j ].w_s;
            if( w_s < 0 )
                continue;
            const auto * q_src = src.at( w[ j ].i );
            for( int n = 0; n < src.n_int; ++ n )
                q_tgt[ n ] += q_src[ n ] * w_s;
        }
        const auto & rho = q_tgt[ idx::rho ];
        if( bc == 1 )
        {
            for( int a = 0; a < 3; ++ a )
                q_tgt[ idx::mom + a ] = v_wall[ a ];
        }
        else
        {
            float_t vn( 0 ), un( 0 );
            for( int a = 0; a < 3; ++ a )
            {
                vn += q_tgt[ idx::mom + a ] * norm[ a ];
                un += v_wall[ a ] * norm[ a ];
            }
            if( nu_lam > 0 )
            {
                // Log-law wall model: retard the ghost's tangential
                // velocity so that the reconstruction sees the
                // log-law gradient at the wall.  u_tau solves
                //     u_e = u_tau * ( ln( y_e u_tau / nu ) / k + B )
                // with the interpolated (image-point) state as the
                // exchange value at height y_e = |phi0|.  The
                // tangential part is then scaled by
                //     fac = 1 - 2 u_tau / ( kappa u_e ),
                // so fac = 1 recovers free slip (u_tau -> 0) and
                // fac = -1 the no-slip reversal.
                const float_t kappa( 0.41 ), B( 5.2 );
                float_t ue2( 0 );
                for( int a = 0; a < 3; ++ a )
                {
                    const float_t vt = q_tgt[ idx::mom + a ]
                                     - vn * norm[ a ];
                    ue2 += vt * vt;
                }
                const float_t ue = __sqrt1( ue2 );
                const float_t ye = fabsf( phi0 );
                if( ue > 0 && ye > 0 )
                {
                    float_t ut = ue * 0.06;
                    for( int it = 0; it < 6; ++ it )
                    {
                        float_t arg = ye * ut / nu_lam;
                        if( arg < 1.05 )
                            arg = 1.05;      // viscous sublayer guard
                        const float_t la = logf( arg );
                        const float_t f  = ut * ( la / kappa + B ) - ue;
                        const float_t df = la / kappa + B + 1 / kappa;
                        ut -= f / df;
                        const float_t u_min = nu_lam / ye;
                        if( ut < u_min )
                            ut = u_min;
                    }
                    float_t fac = 1 - 2 * ut / ( kappa * ue );
                    if( fac < -1 )
                        fac = -1;
                    for( int a = 0; a < 3; ++ a )
                    {
                        const float_t vt = q_tgt[ idx::mom + a ]
                                         - vn * norm[ a ];
                        q_tgt[ idx::mom + a ] = vt * fac
                                              + vn * norm[ a ];
                    }
                }
            }
            for( int a = 0; a < 3; ++ a )
                q_tgt[ idx::mom + a ]
                    -= 2 * ( vn - un ) * norm[ a ];
        }
        // Isothermal wall: R = 1 in code units, so p = rho T.
        if( T_wall > 0 )
            q_tgt[ idx::ene ] = rho * T_wall;
        return;
    };
};

};                              // namespace extension::immersed
