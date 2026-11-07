#pragma once

#include "../../../utilities/phys/constants.h"
#include "../../dynamics/hydro/eos/adiabatic.h"
#include "../chem_hydro/base_gen.h"

namespace chemistry::nuclear
{
//////////////////////////////////////////////////////////
// EoS with chemistry

// Adiabatic base is a template parameter so that both the
// hydro ( n_hyd = 5 ) and the MHD ( n_hyd = 8 ) adiabatic
// cores can host nuclear chemistry. The base resolves to
// ad_T< this_t > when used stand-alone, and to ad_T<
// derived_T > when derived ( mirroring crtp::helper,
// which cannot carry a template-template argument ).
template< template< class > class ad_T
              = dynamics::hydro::eos::adiabatic_t,
          class derived_T = crtp::dummy_t >
struct eos_t : ::chemistry::hydro::base_gen_t ,
    ad_T < std::conditional_t
         < std::is_same_v< derived_T, crtp::dummy_t >,
           eos_t < ad_T, crtp::dummy_t >,
           derived_T > >
{
    ////////// Type //////////
    __crtp_def_self__( eos_t, derived_T, ad_T );
    using super_t = ad_T < std::conditional_t
         < std::is_same_v< derived_T, crtp::dummy_t >,
           eos_t < ad_T, crtp::dummy_t >,
           derived_T > >;

    ////////// Data //////////
    int              n_hva;
    int              n_sps;
    type::float_t     Tmax;
    type::float_t     Tmin;
    type::float_t * mcgs_h;
    type::float_t * mcgs_d;
    using super_t::   unit;

    ////////// Functions //////////

    __host__ virtual void set_chem
    ( const input & args, const sp_vec_t & species,
      device::base_t & f_dev )
    {
        Tmax = args.get
             < float > ( "chemistry", "Tmax", 1e10 );
        Tmin = args.get
             < float > ( "chemistry", "Tmin", 1e3  );
        n_sps  = species.size(  );
        n_hva  = this->n_hvar(  );
        mcgs_h = new type::float_t[ n_sps ];
        for( int n = 0; n < species.size(  ); ++ n )
            mcgs_h[ n ] = species[ n ].mass_cgs;
        mcgs_d = f_dev.malloc_const
               < type::float_t >  ( n_sps );
        f_dev.f_cc( mcgs_d, mcgs_h,
                    n_sps * sizeof( type::float_t ) );
        return;
    };

    __host__ virtual void free( device::base_t & f_dev )
    {
        if( mcgs_h !=  nullptr )
        {
            delete [  ] mcgs_h ;
            mcgs_h   = nullptr ;
        }
        return;
    };

    template< class f_T, class f2_T > __forceinline__
    __device__  f_T c2p( f_T *  prim, f2_T *  cons ,
                         f_T * p_src, f2_T * c_src ) const
    {
        const auto & n_var( n_hva  +  n_sps );
        for( int n = n_hva; n < n_var; ++ n )
            prim[ n ] = c_src[ n ];
        const auto cs2
            = super_t::c2p   ( prim,  cons, p_src, c_src );
        if( cons[ idx::rho ] < p_src[ idx::rho ] * 0.99f )
        {
            const  f_T & rho_ratio( prim[ idx::rho ] / f_T
                                  ( cons[ idx::rho ] ) );
            for( int n = n_hva; n < n_var; ++ n )
                prim [ n ] *= rho_ratio;
        }
        for( int n = n_hva; n < n_var; ++ n )
            p_src[ n ] = prim[ n ];
        return cs2;
    };

    // Gas energy per volume in cgs. For an MHD base the
    // conserved energy also carries |B|^2 / 2 ( and KE ),
    // both of which are removed here. n_hva is configured
    // by set_chem before any device-side use.
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
        f_t b2( 0 );
        if( n_hva > idx::bcc )     // MHD base
            for( int a = 0; a < 3; ++ a )
            {
                const f_t b( cons[ idx::bcc + a ] );
                b2 += b * b;
            }
        f_t rho( cons[ idx::rho ] );
        if( __isnan( rho ) || __isinf( rho ) ||
            rho < this->d_floor )
        {
            rho = this->d_floor ;
            cons[ idx::rho ] = rho;
        }
        e = ( f_t )( cons[ idx::ene ] )
          - p2 / ( 2 * rho ) - b2 / 2;
        const auto emin = this->t_floor * rho;
        if( __isnan( e ) || __isinf( e ) || e < emin )
        {
            e = emin;
            cons[ idx::ene ] = e
                  + p2 / ( 2 * rho ) + b2 / 2;
        }
        return e * unit.ene0;
    };

    __device__ __forceinline__
    type::float_t mcgs( const int & l ) const
    {
        return mcgs_d[ l ];
    };

    template < class dat_T > __forceinline__ __device__
    type::float_t deps_dn( dat_T & d, const int & i ) const
    {
        return phys::cgs::kb * d.T / ( this->gamma - 1 );
    };  // d( energy_density ) / d( number_density )

    template < class dat_T > __forceinline__
    __device__  void temp( dat_T & d ) const
    {   // v-- Here cv is the specific heat capacity.
        type::float_t cv ( 0 );
        __syncthreads(  );
        for( int  i = 0; i < n_sps; ++ i )
            if( i != d.i_en )
                cv += d.y[ i ] ;
        if( cv <= 0 || __isnan( cv ) )
            cv  = 1 ;
        cv *= phys::cgs::R / ( this->gamma - 1 );

        //  Rates are evaluated at the true temperature:
        //  no [Tmin, Tmax] clamp ( those limits encode
        //  database fit ranges, not physics ) and no
        //  energy rescale.  The guard only keeps the
        //  rate evaluation ( pow( T9, 1/3 ), 1/T9 )
        //  finite; at 1e-20 K every rate is ~ e^-80 = 0
        //  anyway, so nothing physical is restricted.
        d.T = __max1( d.y[ d.i_en ] / cv, 1e-20f );
        __syncthreads(  );
        return;
    };
    template < class dat_T > __forceinline__
    __device__ type::float_t dlnT_dlne( dat_T & d ) const
    {   // ln( Specific_Energy ) / ln( T )
        return 1;
    };  // 1 for ideal gas, but may be different otherwise
};
};                         // namespace chemistry::nuclear
