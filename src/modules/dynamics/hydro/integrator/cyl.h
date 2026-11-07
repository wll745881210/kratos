#pragma once

#include "rk2.h"
#include "../../../../utilities/types/my_type_traits.h"

namespace dynamics::hydro::integrator
{
////////////////////////////////////////////////////////////
// Cylindrical

template< class derived_T = crtp::dummy_t >
struct cyl_t : crtp::helper_t< cyl_t, derived_T, rk2_t >
{
    ////////// CRTP functions //////////
    __crtp_def_self__( cyl_t, derived_T );
    using super_t
        = crtp::helper_t< cyl_t, derived_T, rk2_t >;

    ////////// Data //////////
    type::float_t omega_0;

    ////////// Functions //////////
    __host__ cyl_t(  ) : super_t(  ), omega_0( 0 ) {  };

    __host__ virtual void init
    ( const input & args, const::dynamics::base_t & mod )
    {
        omega_0 = args.get < type::float_t >
                ( "dynamics", "omega_0", 0 );
        return super_t::init( args, mod );
    };

    template< class  f2_T, class bdt_T,
              class fif_T, class idx_T, int I = 1 >
    __device__ __forceinline__ f2_T  flux_to_du
    ( const fif_T & f, const bdt_T & b_d, const idx_T & i ,
      const int &  ax, const int   & n )  const
    {
        static_assert( I * I < 0,
                       " Do NOT use default flux_to_du with"
                       " cylindrical coordinates !!!" );
    };

    __device__ __host__ const super_t & get_super(  ) const
    {
        return static_cast< const super_t & > ( * this );
    };

    //  Flux divergence in cylindrical coordinates.  The
    //  azimuthal momentum is transported as the LAB-frame
    //  angular momentum:  its flux is augmented by the
    //  frame advection omega * lever_f * F( rho ) and
    //  rescaled by lever_f / lever_c ( the lever arm r is
    //  constant along phi- and z-faces, so only r-faces
    //  need the rescale ), making r * rho * v_phi^lab
    //  telescope exactly;  the frame part is then removed
    //  again below from the mass divergence.  The energy
    //  row picks up the corresponding rotational
    //  kinetic-energy flux and frame work.  With
    //  omega_0 = 0 all augmentations are exact +0 and the
    //  scheme is bit-neutral.  This replaces the earlier
    //  never-wired "mod_flux" hook design.
    template< class  f2_T, class bdt_T , class x_T,
              class fif_T, class idx_T >
    __device__ __forceinline__ void flux_to_du_cyl
    ( f2_T * u, const fif_T & f, const bdt_T & b_d,
      const idx_T & i , const x_T & x, const int & ax,
      const int   & n ) const
    {
        const auto & geo( b_d.geo );
        const auto r  (           x[ 0 ] );
        const auto vf0( omega_0 * r      );
        f2_T res ( 0 );
        for( int j = 0; j < 2 ; ++ j )
        {
            if( f.f_a[ j ] <= 0 )
                continue;
            const auto f_side( 1 - 2 * j );
            if( n == idx::mom + 1 )
            {
                auto vff( omega_0 );
                if( ax == 0 )
                    vff *= geo.x_fc( 0, i[ 0 ] + j );
                else
                    vff *= r;
                auto f_mphi = f.flux[ j ][ n ]
                            + f.flux[ j ][ idx::rho ] * vff;
                if( ax == 0 )
                    f_mphi *= geo.x_fc( 0, i[ 0 ] + j ) / r;
                res += f.f_a[ j ] * f_mphi * f_side;
            }
            else if( n == idx::ene )
            {
                auto vff( omega_0 );
                if( ax == 0 )
                    vff *= geo.x_fc( 0, i[ 0 ] + j );
                else
                    vff *= r;
                const auto f_ene = f.flux[ j ][ n ]
                    + f.flux[ j ][ idx::rho ] * vff * vff / 2
                    + f.flux[ j ][ idx::mom + 1 ] * vff;
                res += f.f_a[ j ] * f_ene * f_side;
            }
            else
                res += f.f_a[ j ] * f.flux[ j ][ n ]
                    *  f_side;
        }
        if( omega_0 != 0 )
        {
            if( n == idx::rho )
            {
                u[ idx::mom + 1 ] -= res * vf0;
                u[ idx::ene     ] += res * ( vf0 * vf0 / 2 );
            }
            else if( n == idx::mom + 1 )
                u[ idx::ene ] -= res * vf0;
        }
        u[ n ] += res;
        return;
    };

    template< class dat_T, class bdt_T, class eos_T >
    __device__ void add_flux( dat_T & d, const bdt_T & b_d,
                              const eos_T & eos ) const
    {
        const auto &  geo = b_d.geo;
        const auto & self = get_self(  );

        int i[ 3 ], i_1d_c, i_1d_p;
        self.dispatch  ( i, i_1d_c, i_1d_p, d, b_d );

        using  f_t = typename decltype( b_d.hyd )::   f_t;
        dynamics::hydro::integrator::flux_info_t< f_t > f;

        auto * u     = d.u.at( i_1d_c );
        auto * u_tgt = self.load( u, b_d.hyd.u, i, d.step );
        const auto & n_var = b_d.hyd.u.n_int;

        coord_t   x;
        geo.x_cc( x, i );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            self.load_flux_sfc( f, d, b_d, i, i_1d_p, a );
            for( int n = 0; n < n_var; ++ n )
                self.flux_to_du_cyl
                    ( u, f, b_d, i, x, a, n );
        }
        self.src_geo( u, d.q.at( i_1d_p ), b_d, x, i );
        for( int n = 0 ; n < n_var; ++ n )
            u_tgt[ n ] = u [ n ];
        return;
    };

    template < class bdt_T, class eos_T >
    __device__ void  add_flux
    ( const bdt_T &  b_d , const eos_T &  eos,
      const   int & axis , const bool  & side,
      const   int & step ) const
    {
        idx_t i;
        i[ axis ] = side * b_d.geo.n_ceff [ axis ];
        i[ utils::mod3( axis + 2 ) ] =  blockIdx.x;
        i[ utils::mod3( axis + 1 ) ] = threadIdx.x
                        + blockIdx.z *  blockDim.x;
        using  f_t = typename decltype( b_d.hyd )::f_t;
        hydro::integrator::flux_info_t< f_t > f;
        i[ axis ]  -= side;
        f.vol =  b_d.geo.volume( i );
        const auto s( step == b_d.hyd.u.n_step - 1
                      ? 0 : step  +  1 );
        auto  * u = b_d.hyd.u.at( i, s );
        coord_t x;
        b_d.geo.x_cc( x, i );

        i[ axis ]  += side;
        f.flux[   side ] = b_d.hyd.f[ axis ].at ( i );
        f.f_a [   side ] = b_d.geo.surface( i, axis )
                         * b_d.dt / f.vol ;
        f.flux[ ! side ] = nullptr;
        f.f_a [ ! side ] = -1;

        for( int n = 0; n < b_d.hyd.n_fld (  ); ++ n )
            flux_to_du_cyl( u, f, b_d, i, x, axis, n );
        return;
    };

    //  Geometric source term:  only the radial momentum
    //  equation has one in cylindrical coordinates,
    //  ( p + rho * v_phi^2 ) / r.  The cell radius
    //  ( r_l + r_r ) / 2 is the EXACT volume-averaged
    //  < 1 / r > moment of the cell, on any radial
    //  spacing.
    template< class  f2_T, class   f_T ,
              class bdt_T, class idx_T >
    __device__ __forceinline__ void  src_geo
    ( f2_T * u, const f_T * q, const bdt_T &  b_d,
      const coord_t & x,  const  idx_T & i ) const
    {
        const auto & geo = b_d.geo;
        const auto r_c = ( geo.x_fc( 0, i[ 0 ] + 1 )
                         + geo.x_fc( 0, i[ 0 ]     ) ) / 2;
        //  Centripetal term uses the LAB-frame azimuthal
        //  velocity:  ( v_phi + omega * r )^2 expands to
        //  the curvature + Coriolis + centrifugal forces.
        const auto vf  = q[ idx::mom + 1 ] + omega_0 * r_c;
        const auto vf2 = vf * vf;
        u[ idx::mom ] += b_d.dt / r_c
            * ( q[ idx::ene ] + q[ idx::rho ] * vf2 );
        return;
    };
};

}; // namespace dynamics::hydro::integrator
