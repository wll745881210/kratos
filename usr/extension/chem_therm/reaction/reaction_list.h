#pragma once

#include "../../../../src/modules/chemistry/reaction/reaction_list.h"
#include "dust_reaction.h"
#include "photon_reaction.h"
#include "atom_cooling.h"
#include "mol_cooling.h"
#include "phot_cache.h"
#include "intp.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
//

struct list_t : public ::chemistry::reaction::list::std_t
< reac_t, list_t >
{
    ////////// Type //////////
    using super_t
    = ::chemistry::reaction::list::std_t< reac_t, list_t >;

    ////////// Problem-specific data //////////
    type::float_t       sig_dust_cgs;
    type::float_t         h0_col_cgs;
    type::float_t          col_ratio;
    type::float_t           flx_conv;
    type::float_t           * hv_erg;    
    type::float_t               T_eq;

    type::float_t     fac_dust_stiff;
    // If satisfies this, current step identified stiff

    ////////// Different types of reactions //////////
    int        n_hv;
    int      i_elec;
    int      n_cool;
    
    int l_ph_shield;
    int l_phot_reac;    
    int l_dust_reac;
    int l_atom_cool;
    int l_mole_cool;

    utils::const_list_t< dust_reac_t > dust_reac;    
    utils::const_list_t< phot_reac_t > phot_reac;
    utils::const_list_t< atom_cool_t > atom_cool;
    utils::const_list_t<  mol_cool_t >  mol_cool;

    ////////// Host-side Functions //////////
    __host__ virtual void  init_mol_cool
    ( const  input & args, ::chemistry::base_t & chem );
    __host__ virtual void  init_atom_cool
    ( const  input & args, ::chemistry::base_t & chem );
    __host__ virtual void  init_phot_reac
    ( const  input & args, ::chemistry::base_t & chem );
    __host__ virtual void  init_dust_reac
    ( const  input & args, ::chemistry::base_t & chem );
    __host__ virtual void  init_extra
    ( const  input & args, ::chemistry::base_t & chem );    
    __host__ virtual void  init
    ( const  input & args, ::chemistry::base_t & chem );

    __host__ virtual void print_extra
    ( const ::chemistry::base_t & chem );

    __host__ __device__ __forceinline__ int size(  ) const
    {
        return l_mole_cool;
    };

    __host__ virtual int  size_sh (  ) const
    {
        auto res = n_cool     * sizeof( intp_t<    > )
             + phot_reac.size * sizeof( phot_cache_t );
        return utils::align( res, 8 );
    };    

    ////////// Device-side Functions //////////
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void prep_dev
    ( dat_T       &   d , const bdt_T & b_d,
      const eos_T & eos ) const
    {
        __syncthreads(  );        
        eos.temp ( d  );
        d.n_elec = ( i_elec < 0  ? 0 : phys::cgs::NA * d.
                     rho_cgs * __max1( d.y[ i_elec ], 0 ) );
        
        const auto n_col_conv = eos.unit.rho0
            / (  eos.unit.l0  * eos.unit.l0 );
        for( int i = threadIdx.x; i < d.dim;
             i += blockDim.x )
        {
            if( i == d.i_en )
                continue;
            d.n_col[ i ] = b_d.rad.sp_col.at( d.idx )[ i ]
                  * n_col_conv * col_ratio / eos.mcgs( i );
            if( h0_col_cgs > 0 )
            {
                const auto n_col_tst
                    = __max1( d.y[ i ], 1e-38f ) *
                    h0_col_cgs * d.rho_cgs * phys::cgs::NA;
                d.n_col[ i ] = __max1
                    ( n_col_tst, d.n_col[ i ] );
            }
        }
        __syncthreads(  );
        type::float_t y_tot( 0 );
        for( int i  = 0; i < d.dim; ++ i )
            if ( i != d.i_en )
                y_tot += d.y[ i ];
        d.n_tot = y_tot * d.rho_cgs * phys::cgs::NA;

        d.n_hv = b_d.rad.n_fld(  );
        d. flx = b_d.rad.flx.at( d.idx );
        const auto * flx_src = b_d.rad.flx.at( d.idx );
        for( int n = threadIdx.x ; n < d.n_hv;
                 n += blockDim.x )
            d.flx[ n ] = flx_src[ n ];
        __syncthreads(  );
        
        const  int n_p( n_cool + phot_reac.size );
        for( int i = threadIdx.x; i < n_p; i += blockDim.x )
        {
            if( i  < mol_cool.size )
            {
                const auto & mc = mol_cool[ i ];
                d.f_cool[ mc.i_cool ].setup
                    ( d, b_d, eos, * this, mc );
            }
            else if( i < n_cool )
            {
                const auto & ac
                    = atom_cool[ i - mol_cool.size ];
                d.f_cool[ ac.i_cool ].setup
                    ( d, b_d, eos, * this, ac );
            }
            else
                phot_reac[ i - n_cool ].prepare
                         ( d, b_d, eos, * this );
        }
        if( fac_dust_stiff > 0 && threadIdx.x == 0 )
            ( * d.de_dt_d ) = 0;
        __syncthreads(  ) ;
        
        type::float_t flx_bol( 0 );
        for( int i_hv = 0; i_hv < d.n_hv; ++ i_hv )
            flx_bol += d.flx[ i_hv ] * flx_conv
                    * hv_erg[ i_hv ];
        d.T_dust = __sqrt1( __sqrt1
                 ( flx_bol / phys::cgs::sigma_sb ) );
        d.T_dust = utils::max( T_eq, d.T_dust );
        return;
    };

    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void jac_single
    ( dat_T       &   d, const int   & i ,
      const bdt_T & bdt, const eos_T & e ) const
    {
        if( i < reac.size )
            reac[ i ].jac ( d, bdt, e, * this );
        else if( i < l_phot_reac )
            phot_reac[ i -   reac.size ].
                jac( d, bdt, e, * this );
        else if( i < l_dust_reac )
            dust_reac[ i - l_phot_reac ].
                jac( d, bdt, e, * this );
        else if( i < l_atom_cool )
            atom_cool[ i - l_dust_reac ].
                jac( d, bdt, e, * this );
        else if( i < l_mole_cool )
            mol_cool [ i - l_atom_cool ].
                jac( d, bdt, e, * this );
        return;
    };
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void rate_single
    ( dat_T       &   d, const int   & i ,
      const bdt_T & bdt, const eos_T & e ) const
    {
        if( i < reac.size )
            reac[ i ].rate( d, bdt, e, * this );
        else if( i < l_phot_reac )
            phot_reac[ i -   reac.size ].
               rate( d, bdt, e, * this );
        else if( i < l_dust_reac )
            dust_reac[ i - l_phot_reac ].
               rate( d, bdt, e, * this );
        else if( i < l_atom_cool )
            atom_cool[ i - l_dust_reac ].
               rate( d, bdt, e, * this );
        else if( i < l_mole_cool )
            mol_cool [ i - l_atom_cool ].
               rate( d, bdt, e, * this );
        return;
    };


    template< class dat_T, class bdt_T,  class eos_T >
    __device__ bool stiff_temp
    ( dat_T &   d,  const bdt_T & bdt,
      const eos_T & eos ) const
    {
        if( this->fac_dust_stiff < 0 )
            return false;
        __syncthreads(  );

        auto de_dt_dust = ( * d.de_dt_d );
        auto de_dt_ext  = d.f[ d.i_en ] - de_dt_dust;

        // Are the cooling/heating too fast?
        const auto de_dt_max = __max1( fabs( de_dt_dust ) ,
                                       fabs( de_dt_ext  ) );
        const auto de_max( d.h * de_dt_max );
        bool res( ! __isnan( de_max ) && ! __isinf( de_max )
                  && d.y[ d.i_en ] < de_max );
        if( ! res )
            return res;
        
        // Do we have catastrophic cancellation?
        res &= ( fabsf( d.f[ d.i_en ] )
                 < fac_dust_stiff * fabsf( de_dt_dust ) );

        if( res )
        {
            const auto T0 = d.T;
            d.T = d.T_dust + de_dt_ext / de_dt_dust
                * ( d.T_dust - d.T );
            // if( threadIdx.x == 0 )
            // printf( "%g %g ( %g ) %g; %g %g %g\n", de_dt_ext,
            //         de_dt_dust, de_dt_dust / ( d.T_dust - T0 ),
            //         d.f[ d.i_en ], 
            //         T0, d.T, d.T_dust );
        }
        return res;
    };

    template< class dat_T, class bdt_T,  class eos_T >
    __device__ void rate ( dat_T &   d,  const bdt_T & bdt,
                           const eos_T & eos ) const
    {
        super_t:: rate( d, bdt, eos );
        if( stiff_temp( d, bdt, eos ) )
            if( threadIdx.x == 0 )
                d.f[ d.i_en ] = 0 ;
        __syncthreads(  );
        return;
    };

    template< class dat_T, class bdt_T , class eos_T >
    __device__ void jac  ( dat_T &   d , const bdt_T & bdt,
                           const eos_T & eos ) const
    {
        super_t ::jac ( d, bdt, eos ) ;
        if( stiff_temp( d, bdt, eos ) )
        {
            for( int i  = threadIdx.x ; i < d.dim;
                     i +=  blockDim.x )
            {
                d.J( i, d.i_en ) = 0;
                d.J( d.i_en, i ) = 0;
            }
            if( threadIdx.x  == 0 )
                d.f[ d.i_en ] = 0 ;
        }
        __syncthreads(  ) ;
        return;
    };
    
    // Like rate_single but returns the reaction rate value
    // (the `res` from rate()). Side effects on d.f are
    // still produced but discarded by the caller. Used only
    // by the diagnostic output kernel; never on the hot
    // solve path.
    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ type::float_t rate_val_single
    ( dat_T       &   d, const int   & i ,
      const bdt_T & bdt, const eos_T & e ) const
    {
        if( i < reac.size )
            return reac[ i ].rate( d, bdt, e, * this );
        else if( i < l_phot_reac )
            return phot_reac[ i -   reac.size ].
                          rate( d, bdt, e, * this );
        else if( i < l_dust_reac )
            return dust_reac[ i - l_phot_reac ].
                          rate( d, bdt, e, * this );
        else if( i < l_atom_cool )
            return atom_cool[ i - l_dust_reac ].
                          rate( d, bdt, e, * this );
        else if( i < l_mole_cool )
            return mol_cool [ i - l_atom_cool ].
                          rate( d, bdt, e, * this );
        return 0;
    };

    ////////// Prepare for Radiation  //////////
    template     < class bdt_T ,  class eos_T > __device__
    void prep_rad( const bdt_T &, const eos_T & ) const;
};

};                     // namespace extension::chemistry
