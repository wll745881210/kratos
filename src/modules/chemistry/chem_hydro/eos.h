#pragma once

#include "../../../utilities/phys/constants.h"
#include "../../dynamics/hydro/eos/adiabatic.h"
#include "base_gen.h"
#include "cv_poly.h"

namespace chemistry::hydro
{
////////////////////////////////////////////////////////////
// EoS with chemistry

template< typename T >
using eos_ad_t = dynamics::hydro::eos::adiabatic_t< T >;

template< class      cv_T = cv_poly_list_t ,
          class derived_T = crtp:: dummy_t >
struct eos_t : base_gen_t ,
    eos_ad_t < crtp::helper< eos_t, cv_T, derived_T > >
{
    ////////// Type //////////
    __crtp_def_self__( eos_t, derived_T );
    using super_t
    = eos_ad_t < crtp::helper< eos_t, cv_T, derived_T > >;

    ////////// Data //////////
    using base_gen_t  ::n_hva;
    using base_gen_t  ::n_sps;   // e_gas not included!
    using    super_t  :: unit;
    type  :: float_t     Tmax;
    type  :: float_t     Tmin;
    type  :: float_t     Ttol;
    type  :: float_t * mcgs_h;
    type  :: float_t * mcgs_d;
    type  :: float_t * gm1i_h;
    type  :: float_t * gm1i_d;
    cv_T               cv_dat;    
    int                n_iter;

    ////////// Host-side functions //////////
    __host__ eos_t(  ) : Tmax  ( 1e12 ), Tmin( 1 ),
                         mcgs_h( nullptr )   {   };

    __host__ virtual void read( const input & args )
    {
        Tmax   = args.get< type::float_t >
               ( "chemistry", "Tmax",    1e7 );
        Tmin   = args.get< type::float_t >
               ( "chemistry", "Tmin",    2.7 );
        Ttol   = args.get< type::float_t >
               ( "chemistry", "T_rtol", 1e-4 );        
        n_iter = args.get< int >
               ( "chemistry", "eos_iter", 20 );
        return super_t::read( args );
    };

    __host__ virtual void set_chem
    ( const input & args, const sp_vec_t & species,
      device::base_t & f_dev )
    {
        n_sps  = species.size(  );
        n_hva  = this->n_hvar(  );
        mcgs_h = new type::float_t[ n_sps ];
        gm1i_h = new type::float_t[ n_sps ];        
        for( int n = 0; n < species.size(  ); ++ n )
        {
            const  auto &   sp = species[ n ];
            mcgs_h[ n ] =   sp.mass_cgs;
            gm1i_h[ n ] = 0.5 * 
                  ( sp.n_atom  <= 1 ? 3 :
                  ( sp.n_atom  >= 3 ? 6 : 5 ) );
        }
        mcgs_d = f_dev.malloc_const
               < type::float_t >  ( n_sps );
        gm1i_d = f_dev.malloc_const
               < type::float_t >  ( n_sps );
        f_dev.f_cc( mcgs_d, mcgs_h,
                    n_sps * sizeof( type::float_t ) );
        f_dev.f_cc( gm1i_d, gm1i_h,
                    n_sps * sizeof( type::float_t ) );
        return cv_dat.setup( args, species, f_dev );
    };

    template< class v_T, class f_T > __device__
    auto mu_cgs_xn( const v_T & x, const f_T & dat ) const
    {   // From number fraction
        type::float_t tot( 0 ), norm( 1e-32f );
        for( int n = 0; n < n_sps; ++ n )
        {
            const auto x_n( __max1( x[ n ], 0 ) );
            tot  += x_n * mcgs( n );
            norm += x_n ;
        }
        return tot / norm;
    };

    template< class f_T, class g_T > __host__ __device__
    f_T basic_cv( const f_T * x, const g_T * gm1i ) const
    {
        f_T cv_kb( 0 ) ;
        f_T norm ( 1e-32f );
        for( int n = 0; n < n_sps; ++ n )
        {
            const auto x_n( __max1( x[ n ], 0 ) );
            cv_kb += x_n * gm1i[ n ];
            norm  += x_n ;
        }
        cv_kb /= norm;
        if( cv_kb <= 0 || __isnan( cv_kb ) ||
                          __isinf( cv_kb ) )
            cv_kb = 1 / ( this->gamma - 1 );
        return cv_kb;
    };

    __host__ virtual type::float_t cvkb_xn_T_h
    ( type::float2_t * x, const type::float2_t & T ) const
    {
        if( x == nullptr )
            return 1. / ( this->gamma - 1 );
        else if( ! cv_dat )
            return basic_cv( x, gm1i_h );
        return cv_dat.xn_h( x, T );
    };
    
    __host__ virtual void free( device::base_t & f_dev )
    {
        if( cv_dat )
            cv_dat.free( f_dev );
        if( mcgs_h !=  nullptr )
        {
            delete [  ]  mcgs_h;
            delete [  ]  gm1i_h;            
            mcgs_h    = nullptr;
            gm1i_h    = nullptr;
        }
        return;
    };

    ////////// Device-side function //////////
    template< class v_T, class f_T > __device__
    void w_erho_d( f_T & cv_kb, f_T & gamma, const v_T * w,
                   const f_T & e_g, const f_T & rho ) const
    {   // input: primative-number frac ( wn ), e_g, rho;
        if( ! cv_dat )
            return w_prho_d( cv_kb, gamma, w, e_g, rho );

        const auto    * x = w + n_hva;
        auto ee = e_g * unit.ene0 / ( rho * unit.rho0 )
             * mu_cgs_xn( x, cv_dat.dat_d ) / phys::cgs::kb;
        auto T  = ee / cv_dat.xn_d_simple( x );
        for( int i = 0; i < n_iter; ++ i )
        {
            const auto Told = T;
            T = ee / cv_dat.xn_d( x, T );
            if( __abs1( T - Told ) / ( T + 1e-32f ) < Ttol )
                break;            
        }
        cv_kb = ee / T;
        gamma = 1 / cv_dat.dT_xn_d( x, T ) + 1;
        if( T <= 0 || __isnan( T ) || __isinf( T ) )
        {
            cv_kb = basic_cv( x, gm1i_d );
            gamma = 1 / cv_kb + 1;
        }
        return;
    };

    template< class v_T, class f_T > __device__
    void w_prho_d( f_T & cv_kb, f_T & gamma, const v_T * w,
                    const f_T & pre, const f_T & rho ) const
    {   // input: primative-number frac ( wn ), pre, rho;
        const auto * x = w + n_hva;
        if( ! cv_dat )
        {
            cv_kb = basic_cv( x, gm1i_d );
            gamma = 1 / cv_kb + 1;
            return;
        }
        const auto mu_cgs = mu_cgs_xn ( x, cv_dat.dat_d );
        const auto T = pre * unit.ene0 * mu_cgs
                   / ( rho * unit.rho0 * phys:: cgs::kb );
        cv_kb = cv_dat.xn_d( x, T );
        gamma = 1 / cv_dat.dT_xn_d( x, T ) + 1;
        // gamma = __max1( gamma, 1 / cv_kb + 1 );        
        return;
    };

    template< class f_T, class f2_T > __forceinline__
    __device__  f_T c2p( f_T *  prim, f2_T *  cons ,
                         f_T * p_src, f2_T * c_src ) const
    {
        const  auto & n_var( n_hva  +  n_sps );
        for( int  n = n_hva; n < n_var; ++ n )
            prim[ n ] = c_src[ n ];

        const auto     cs2
            = super_t::c2p   ( prim,  cons, p_src, c_src );
        if( fabs( cons[ idx::rho ] / prim[ idx::rho ] - 1 )
            > 1e-2f )
        {
            const f_T & rho_ratio( prim[ idx::rho ] / f_T
                                 ( cons[ idx::rho ] ) );
            for( int n = 0; n < n_sps; ++ n )
            {
                prim [ n_hva + n ] *= rho_ratio;
                c_src[ n_hva + n ] *= rho_ratio;
            }
        }
        f_T norm ( 0 );
        for( int n = 0; n < n_sps; ++ n )
        {
            auto  & u_n( prim[ n_hva + n ] );
            u_n   = utils::max( u_n, 0 );
            norm += u_n * mcgs_d[ n ];
        }
        norm = prim[ idx::rho ] * phys::cgs::ma / norm;
        for( int n = 0; n < n_sps; ++ n )
        {
            prim [ n_hva + n ] *= norm;
            c_src[ n_hva + n ]  = prim[ n_hva + n ];
        }
        for( int n = n_hva; n < n_var; ++ n )
            p_src[ n ] = prim [ n ];
        return cs2;
    };
    
    template < class f2_T > __device__ __forceinline__
    type::float_t e_g_cgs( f2_T * cons ) const
    {
        using  f_t = type::float_t;
        f_t e( 0 ), p2( 0 );
        for( int a = 0; a < 3; ++ a )
        {
            const f_t p( cons[ idx::mom + a ] );
            p2 += p * p;
        }
        f_t rho( cons[ idx::rho ] );
        e = ( f_t )( cons[ idx::ene ] ) - p2 / ( 2 * rho );
        const auto emin = this->t_floor * rho;
        if( __isnan( e ) || __isinf( e ) || e < emin )
        {
            e = emin;
            cons[ idx::ene ] = e + p2 / ( 2 * rho );
        }
        return e * unit.ene0;
    };

    template < class dat_T > __forceinline__
    __device__  void temp( dat_T & d ) const
    {   // v-- Here cv is the specific heat capacity.
       __syncthreads(  );
        auto ee = d.y[ d.i_en ] * mu_cgs_xn
            ( d.y, cv_dat.dat_d ) / phys::cgs::kb;
        type::float_t T( d.T );
        if( ! cv_dat )
            T = ee / basic_cv( d.y, gm1i_d );
        else
        {
            if( T < 0 )
                T = ee / cv_dat.xn_d_simple( d.y );
            for( int i = 0; i < n_iter; ++ i )
            {
                const auto Told = T;
                T   = ee / cv_dat.xn_d( d.y, T );
                if( __abs1( T - Told ) / ( T + 1e-32f )
                    < Ttol )
                    break;
            }
            if( T <= 0 || __isnan( T ) || __isinf( T ) )
                T = ee / basic_cv( d.y, gm1i_d );
        }
        __syncthreads(  );
        if( T <= 0 || __isnan( T ) || __isinf( T ) )
            T = ee * ( this->gamma - 1 );
        d.T = __max1( __min1( T, Tmax ), Tmin );
        if( threadIdx.x == 0 && T != d.T )
            d.y[ d.i_en ] = __abs1
               ( d.y[ d.i_en ] * ( d.T / T ) );
        __syncthreads(  );
        return;
    };
    
    template < class dat_T > __forceinline__
    __device__ type::float_t dlnT_dlne( dat_T & d ) const
    {   // ln( Specific_Energy ) / ln( T )
        return 1;
    };  // 1 for ideal gas, but may be different otherwise
    
    template < class dat_T > __forceinline__ __device__
    type::float_t deps_dn( dat_T & d, const int & i ) const
    {
        return ( cv_dat ? cv_dat.dat_d[ i ]( d.T )
                 : gm1i_d[ i ] ) * phys::cgs::kb * d.T;
    };  // d( energy_density ) / d( number_density )

    __device__ __forceinline__
    type::float_t mcgs( const int & l ) const
    {
        return mcgs_d[ l ];
    };
};
};                         // namespace chemistry::hydro
