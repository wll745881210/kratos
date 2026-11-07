#pragma once

#include "rk2.h"
#include "../../../../utilities/types/my_type_traits.h"

namespace dynamics::hydro::integrator
{
////////////////////////////////////////////////////////////
// Spherical

template< class derived_T = crtp::dummy_t >
struct sph_t : crtp::helper_t< sph_t, derived_T, rk2_t >
{
    ////////// CRTP functions //////////
    __crtp_def_self__( sph_t, derived_T );
    using super_t
        = crtp::helper_t< sph_t, derived_T, rk2_t >;

    ////////// Data //////////
    type::float_t omega_0;

    ////////// Functions //////////
    __host__ sph_t(  ) : super_t(  ), omega_0( 0 ) {  };
    
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
                       " spherical polar coordinates !!!" );
    };

    __device__ __host__ const super_t & get_super(  ) const
    {
        return static_cast< const super_t & > ( * this );
    };
    
    template< class  f2_T, class bdt_T , class x_T,
              class fif_T, class idx_T >
    __device__ __forceinline__ void flux_to_du_sph
    ( f2_T * u, const fif_T & f, const bdt_T & b_d,
      const idx_T & i , const x_T & x, const int & ax,
      const int   & n ) const
    {
        const auto & geo( b_d.geo );
        const auto r   (           x[ 0 ]   ) ;
        const auto s_th(   __sin1( x[ 1 ] ) ) ;
        const auto vf0 ( omega_0 * r * s_th ) ;
        f2_T res ( 0 );
        for( int j = 0; j < 2 ; ++ j )
        {
            if( f.f_a[ j ] <= 0 )
                continue;
            const auto f_side( 1 - 2 * j );
            if( ax == 0 && n == idx::mom + 1 )
                res += f.f_a[ j ] * f.flux[ j ] [ n ] / r
                    *  geo.x_fc( 0, i[ 0 ] + j ) * f_side;
            else if( n == idx::mom + 2 ) //  && ax <= 1 )
            {
                auto vf0 = omega_0;
                if( ax == 0 )
                    vf0 *= geo.x_fc( 0, i[ 0 ] + j ) * s_th;
                else if( ax == 1 )
                    vf0 *= r * __sin1
                         ( geo.x_fc( 1, i[ 1 ] + j ) );
                else
                    vf0 *= r * s_th;
                
                auto f_mphi = f.flux[ j ][ n ]
                            + f.flux[ j ][ idx::rho ] * vf0;
                if( ax == 0 )
                    f_mphi *= geo.x_fc( 0, i[ 0 ] + j ) / r;
                else if( ax == 1 )
                    f_mphi *= 1 / s_th 
                      * __sin1( geo.x_fc( 1, i[ 1 ] + j ) );
                res  += f.f_a[ j ] * f_mphi * f_side;
            }
            else if( n == idx::ene )
            { 
                auto vf( omega_0 );
                auto f_ene = f.flux[ j ][ n ];
                if( ax == 0 )
                    vf *= geo.x_fc( 0, i[ 0 ] + j ) * s_th;
                else if( ax == 1 )
                    vf *= r * __sin1
                         ( geo.x_fc( 1, i[ 1 ]  + j ) );
                else
                    vf *= r * s_th;
                f_ene 
                    += f.flux[ j ][ idx::rho ] * vf * vf / 2
                    +  f.flux[ j ][ idx::mom + 2 ]  * vf ;
                res += f.f_a [ j ] * f_ene * f_side;
            }
            else
                res = get_super (  ).template
                    flux_to_du< f2_T >( f, b_d, i, ax, n );
        }
        if( omega_0 != 0 )
        {
            if( n == idx::rho )
            { 
                u[ idx::mom + 2 ] -= res * vf0 ;
                u[ idx::ene ] += res * ( vf0 * vf0 / 2 );
            }   // from -res * vf0**2 / 2 but with dp_phi...
            else if( n == idx::mom + 2 )
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
                self.flux_to_du_sph
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
            flux_to_du_sph( u, f, b_d, i, x, axis, n );
        return;
    };
    
    template< class  f2_T, class   f_T ,
              class bdt_T, class idx_T >
    __device__ __forceinline__ void  src_geo
    ( f2_T * u, const f_T * q, const bdt_T &  b_d,
      const coord_t & x,  const  idx_T & i ) const
    {
        const auto & geo = b_d.geo;
        auto   r_l = geo.x_fc( 0, i[ 0 ]     );
        auto   r_r = geo.x_fc( 0, i[ 0 ] + 1 );
        auto  th_l = geo.x_fc( 1, i[ 1 ]     );
        auto  th_r = geo.x_fc( 1, i[ 1 ] + 1 );

        static const f_T pi( 3.1415926535897932384626 );
        th_l = ( th_l < 0  ? 0  : th_l );
        th_r = ( th_r > pi ? pi : th_r );
        const auto s_th( __sin1( x[ 1 ] ) );
        const auto c_th( __cos1( x[ 1 ] ) );        
        
        const auto * vel( q + idx::mom );
        const auto vf0 = s_th * x[ 0 ] * omega_0;
        const auto v12 = vel[ 1 ] * vel[ 1 ];
        const auto vf  = vel[ 2 ] + vf0;
        const auto v22 = vf * vf;
        
        const auto &  pre( q[ idx::ene ] );
        const auto &  rho( q[ idx::rho ] );
        const auto dt_rho( b_d.dt *  rho );
        const auto rl2 = r_l * r_l;
        const auto rr2 = r_r * r_r;
        const auto cm0 = 3 * ( rr2 - rl2 ) / 2.f
                           / ( rr2 * r_r - rl2 * r_l );

        //////////////////////////////////////////////////
        // Geometric pressure terms and force terms.
        // Note: geometric pressure terms have no
        // contribution to the energy!
        coord_t d_mom;
        d_mom[ 0 ] = b_d.dt * cm0 * 2 * pre;
        d_mom[ 1 ] = pre * b_d.dt * cm0
                   * ( __sin1( th_r ) - __sin1( th_l ) ) 
                   / ( __cos1( th_l ) - __cos1( th_r ) );
        d_mom[ 0 ] += dt_rho * ( v12 + v22 ) / x[ 0 ];
        d_mom[ 1 ] += dt_rho / x[ 0 ]
                    * ( v22 * c_th / s_th );
        // The -rho*v_r*v_th/r part is already carried by
        // the rescaled r-flux for mom_theta; it must NOT
        // appear here. "Original" form: * ( v22 * c_th /
        // s_th - vel[ 0 ] * vel[ 1 ] );
        d_mom[ 2 ] = 0 ;
        for( int a = 0; a < 3; ++ a )
            u[ idx::mom + a ] +=  d_mom[ a ];
        return;
    };
};
    
}; // namespace dynamics::hydro::integrator
