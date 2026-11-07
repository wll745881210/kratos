#pragma once

#include "../../src/modules/chemistry/reaction/reaction.h"
#include "../../src/utilities/phys/constants.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  Analytic diffuse ionization ( WBG19 App. A.1;  usr_ext /
//  ppd / design.md sec. 5;  ledger cont. 45 ):  xi_X ( the
//  two - component Igea & Glassgold 99 / Bai & Goodman 09
//  fit )  + xi_CR ( Umebayashi & Nakano 81 ) + xi_SLR,  with
//  the GAUSSIAN vertical - column estimate and the direct -
//  X - ray veto.
//
//  disk_ionize_t is a PROVIDER in the disk_dust_t sense:
//  eval_xi( d, b_d, eos, flx_conv ) is THE upgrade point --
//  a future runtime particle / Monte - Carlo module will
//  supply the ionization rates directly,  and only this
//  eval ( or a deposited - field read inside it ) changes.
//  The three branch reactions ( H2 -> H + H+ + e- at 1.0,
//  H -> H+ + e- at 0.5,  He -> He+ + e- at 0.84,  WBG19
//  mhd_wind.cpp:776-830 ) stay.

struct disk_ionize_t
{
    ////////// Types //////////
    using float2_t = type::float2_t;

    ////////// Data //////////
    int       on;        // master switch
    float2_t  Lx_29;     // L_X / 1e29 erg/s
    float2_t  xi_CR;     // cosmic - ray rate     [ 1/s ]
    float2_t  xi_SLR;    // short - lived radioisotopes
    float2_t  h_ratio;   // h = h_ratio * r  ( WBG19: 0.06 )
    float2_t  nt_floor;  // n_H floor for the columns
    float2_t  sig_veto;  // the direct - X veto coefficient
    int       i_H, i_Hp, i_H2;  // species slots ( -1 = off )
    int       i_xray;    // first band with hv > 100 eV

    ////////// Functions //////////
    __host__ disk_ionize_t(  )
    {
        on       =      1;
        Lx_29    =     10;      // L_X = 1e30 erg/s default
        xi_CR    =  1e-17;
        xi_SLR   =  6e-19;
        h_ratio  =   0.06;
        nt_floor =   1e8;
        sig_veto = 5e-21 * 3e-4;
        i_H  = -1;   i_Hp = -1;   i_H2 = -1;
        i_xray = -1;
        return;
    };

    __host__ void read( const input & args )
    {
        on       = args.get< int >
                   ( "ionize", "on",        1 );
        xi_CR    = args.get< float2_t >
                   ( "ionize", "xi_CR",  1e-17 );
        xi_SLR   = args.get< float2_t >
                   ( "ionize", "xi_SLR", 6e-19 );
        h_ratio  = args.get< float2_t >
                   ( "ionize", "h_ratio", 0.06 );
        nt_floor = args.get< float2_t >
                   ( "ionize", "nt_floor_cgs", 1e8 );
        //  L_X is par - driven ( the generator consumes
        //  lum_ph_cgs and the module does not retain it ).
        Lx_29    = args.get< float2_t >
                   ( "ionize", "L_X_cgs", 1e30 ) / 1e29;
        return;
    };

    //  Species slots and the X - ray band index ( first
    //  band above 100 eV );  called from ppd_list_t::init.
    template< class chem_T >
    __host__ void setup( const chem_T & chem )
    {
        const auto & sps( chem.species );
        i_H  = sps.exist( "H"  ) ? sps.at( "H"  ) : -1;
        i_Hp = sps.exist( "H+" ) ? sps.at( "H+" ) : -1;
        i_H2 = sps.exist( "H2" ) ? sps.at( "H2" ) : -1;
        const auto & chem_ext
            = dynamic_cast< const extension::chem_t & >
              ( chem );
        if( chem_ext.p_rad )
        {
            const auto & hv( chem_ext.p_rad->hv_eV );
            for( size_t n = 0; n < hv.size(  ); ++ n )
                if( hv[ n ] > 100 )
                {
                    i_xray = ( int ) n;
                    break;
                }
        }
        return;
    };

    //  xi_eff per cell [ 1 / s,  CGS ].   d.n_tot is the
    //  TOTAL particle density;  WBG19's nt is the HYDROGEN -
    //  NUCLEI density n( H ) + n( H+ ) + 2 n( H2 ).
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ type::float_t eval_xi
    ( const dat_T & d, const bdt_T & b_d, const eos_T & eos,
      const type::float_t & flx_conv ) const
    {
        using f_t = type::float_t;
        const auto r_cgs = b_d.geo.x_cc( 0, d.idx[ 0 ] )
                         * eos.unit.l0;
        const auto th    = b_d.geo.x_cc( 1, d.idx[ 1 ] );
        const auto z     = fabs( r_cgs * __cos1( th ) );
        const auto R_au  = r_cgs * __sin1( th )
                         / 1.495978707e13;
        f_t nt( 0 );
        if( i_H  >= 0 ) nt +=     d.y[ i_H  ];
        if( i_Hp >= 0 ) nt +=     d.y[ i_Hp ];
        if( i_H2 >= 0 ) nt += 2 * d.y[ i_H2 ];
        nt *= d.rho_cgs / phys::cgs::ma;
        nt  = utils::max( nt, ( f_t ) nt_floor );
        const auto h   = h_ratio * r_cgs;
        const auto NH1 = nt * h;
        const auto NH2 = 2 * nt * h
            * __exp1( 0.5f * ( z / h ) * ( z / h ) );

        static const f_t xi1 = 4e-12,  xi2 = 2e-15;
        static const f_t al  = 0.5,    be  = 0.65;
        static const f_t N1  = 1.5e21, N2  = 7.0e23;
        static const f_t Ncr = 5.7e25;
        const auto pw = [] ( const f_t & x, const f_t & p )
        {
            return x < 1e-32f ? ( f_t ) 0
                 : __exp1( p * logf( x ) );
        };
        const f_t f1 = __exp1( - pw( NH1 / N1, al ) )
                     + __exp1( - pw( NH2 / N1, al ) );
        const f_t f2 = __exp1( - pw( NH1 / N2, be ) )
                     + __exp1( - pw( NH2 / N2, be ) );
        f_t xi = Lx_29 * pw( R_au, ( f_t ) -2.2 )
               * ( f1 * xi1 + f2 * xi2 )
               + xi_CR * ( __exp1( - NH1 / Ncr )
                         + __exp1( - NH2 / Ncr ) )
               + xi_SLR;
        //  Direct - X veto:  the ray - traced X - ray
        //  photoionization dominates --  zero the diffuse
        //  estimate to avoid double counting.
        if( i_xray >= 0 )
        {
            const auto xi_direct
                = d.flx[ i_xray ] * flx_conv * sig_veto;
            if( xi_direct > xi )
                xi = 0;
        }
        return xi;
    };
};

////////////////////////////////////////////////////////////
//  One diffuse - ionization branch.   rate_coef returns
//  xi_eff * branch ( 1 / s,  CGS );  the base rate gives
//  res = xi_eff * branch * y_reactant.   Accumulation is
//  POST - COMMIT:  ppd_list_t calls rate_add / jac_add
//  AFTER the list's rate() / jac() reduced f_cache,  so
//  these write d.f / d.J directly with atomicAdd.

struct ion_diff_t
    : ::chemistry::reaction::std_t< ion_diff_t >
{
    ////////// Types //////////
    using super_t = ::chemistry::reaction::std_t
        < ion_diff_t >;
    using f_t     = type::float_t;

    ////////// Data //////////
    f_t branch;

    ////////// Device functions //////////
    template< class dat_T, class bdt_T,
              class eos_T, class rls_T >
    __device__ __forceinline__ f_t rate_coef
    ( dat_T       &   d, const bdt_T & bdt,
      const eos_T & eos, const rls_T & rls ) const
    {
        return branch
             * rls.ion.eval_xi( d, bdt, eos, rls.flx_conv );
    };

    template< class dat_T, class bdt_T,
              class eos_T, class rls_T >
    __device__ __forceinline__ f_t dlnk_dlnT
    ( dat_T       &    , const bdt_T & ,
      const eos_T &    , const rls_T &  ) const
    {
        return 0;   // xi_eff is T - independent
    };

    template< class dat_T, class bdt_T,
              class eos_T, class rls_T >
    __device__ __forceinline__ f_t rate_add
    ( dat_T       &   d, const bdt_T & bdt,
      const eos_T & eos, const rls_T & rls ) const
    {
        const auto res = rate_coef( d, bdt, eos, rls )
                       * d.y[ i_chem[ 0 ] ];
        if( res == 0 )
            return 0;
        for( int i = 0; i < n_chem; ++ i )
            atomicAdd( d.f + i_chem[ i ],
                       res * f_chem[ i ] );
        return res;
    };

    template< class dat_T, class bdt_T,
              class eos_T, class rls_T >
    __device__ __forceinline__ void jac_add
    ( dat_T       &   d, const bdt_T & bdt,
      const eos_T & eos, const rls_T & rls ) const
    {
        const auto res = rate_coef( d, bdt, eos, rls )
                       * d.y[ i_chem[ 0 ] ];
        if( __abs1( res ) < 1e-32f )
            return;
        for( int i = 0; i < n_chem; ++ i )
            atomicAdd( & d.J( i_chem[ i ], i_chem[ 0 ] ),
                       f_chem[ i ] * res
                       / ( d.y[ i_chem[ 0 ] ] + 1e-32f ) );
        return;
    };
};

};                              // namespace interm
