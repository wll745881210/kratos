#pragma once

#include "../../../../src/modules/particle/radiation/photon.h"

namespace extension::radiation
{
////////////////////////////////////////////////////////////
//

template< template< class T > class     base_T
        = particle::radiation::photon:: cart_t >
struct photon_t   :  base_T < photon_t< base_T > >
{
    ////////// Types //////////
    using   float_t = type:: float_t;
    using  float2_t = type::float2_t;        
    using   super_t = base_T< photon_t< base_T > >;
    using geo_loc_t = particle::geo_loc_t;

    ////////// Data //////////
    int               n_hv;
    int               n_sp;
    int             n_scat;
    float_t          tau_s;        // Remaining scat tau
    float_t          * dat;        // Extra data

    using super_t::      i;
    using super_t::   step;
    using super_t::   ib_l;
    using super_t::get_self_c;

    ////////// Special declarations //////////
    photon_t(  ) = default;
    // Has internal arrays, disable assign operator?

    ////////// Dual-purpose Data Access //////////
    __host__ __device__ __forceinline__
    int size_dat(  ) const
    {
        auto res( ( n_hv + n_sp ) * sizeof(  float_t ) );
        return utils::align  ( res, sizeof( float2_t ) );
    };
    __host__ __device__ int size(  ) const
    {
        return utils::align( size_dat( ) + sizeof( * this ),
                             sizeof( float2_t ) ) ;
    };
    
    __device__ void init( char * p_sh )
    {
        dat = ( float_t * ) ( p_sh );
        return;
    };
    __host__ __device__ float_t * dat_default(  ) const
    {
        constexpr auto s_self = utils:: align_c
                ( sizeof( photon_t ) , sizeof( float2_t ) );
        return ( float_t * )( ( char * )( this ) + s_self );
    };
    __host__ __device__ void set_offset(  )
    {
        dat = this->dat_default(  );
        return;
    };

    __host__ __device__ __forceinline__
    float_t & lumin ( const int & i )
    {
        return dat[ i ];
    }
    __host__ __device__ __forceinline__
    float_t & column( const int & i )
    {
        return dat[ i + n_hv ];
    }

    ////////// Dual-purpose Save/load
    __host__ __device__ __forceinline__
    void cp_shallow( photon_t & tgt )
    {
        auto * tgt_dat = tgt.dat;
        tgt = ( * this  );
        tgt.dat = tgt_dat;
        return;
    };

    __device__ __forceinline__
    void load( photon_t & src )
    {
        src.cp_shallow( * this );
        //  The  shared - memory  slot  stride  must  cover
        // the  ALIGNED  per - photon  dat  (  2026 - 10 -
        // 15  audit:   the  bare   n_hv + n_sp   stride  is
        // 4 B  short  for  odd   n_hv + n_sp   --  the
        // 8 - B - aligned  memcpy  then  overwrites  the
        // next  thread 's  lumin  with  this  photon 's
        // padding;  ~ 3 / 4  of  the  photons  lost
        // their  luminosity  in  the  bench ):
        dat += threadIdx.x * ( size_dat(  )
                               / sizeof( float_t ) );
        memcpy( dat, src.dat_default (  ), size_dat(  ) );
        return super_t::load( src );
    };
    __host__ __device__ __forceinline__
    void save( photon_t & tgt )
    {
        cp_shallow( tgt );
        memcpy( tgt.dat_default(  ), dat, size_dat(  ) );
        return super_t::save( tgt );;
    };
    __host__ __device__ __forceinline__
    void move( photon_t & tgt )
    {
        cp_shallow( tgt );
        memcpy( tgt  .dat_default(  ),
                this->dat_default(  ), size_dat(  ) );
        return super_t::move( tgt );
    };
    
    ////////// Device-side Fucntions //////////
    template < class x_T, class prx_T, class itg_T >
    __device__ __forceinline__  void proc_phys
    ( const int       & a_proc , const   x_T &  dl,
      const geo_loc_t &    g_l , const prx_T & prx,
      const itg_T     &    itg )
    {
        const auto * mfpa_i ( prx.rad.mfpa_i.at( i ) );
        auto       * flx_tgt( prx.rad.flx   .at( i ) );
        const auto dsi = dl[ a_proc ] / prx.geo.volume( i );

        for( int i_hv = 0; i_hv < n_hv; ++ i_hv )
        {
            const auto tau_t = __max1
                ( mfpa_i[ i_hv ] * dl[ a_proc ], 0 );
            const auto etaut ( __exp1 ( - tau_t  ) );
            const auto flx = lumin( i_hv ) * dsi *
              ( tau_t < 1e-2f ? 1 : ( 1 - etaut ) / tau_t );
            atomicAdd( flx_tgt + i_hv, flx );
            lumin( i_hv ) *= etaut;
        }
        //  The  VET  chi - moment  accumulation  (  2026 -
        // 10 - 15  fld-ppd-integration-plan.md § 3.19 ):
        // weight  w  =  the  band - summed  remaining
        // luminosity  (  the  photon 's  energy  weight );
        // direction  n  =  dir  (  unit ).   Accumulate
        // w  and  w  n_a n_b  into  chi_m  (  [ 0 ]  w,
        // [ 1.. 6 ]  the  lower  triangle ).   Gated  by
        // the  integrator 's  vet_gate  (  the  module
        // sets  it  from  [bench]  chi_vet;   0  =  the
        // legacy  deposit  only  --  bit  identical ):
        if( itg.vet_gate )
        {
            auto * cm( prx.rad.chi_m.at( i ) );
            //  The  DEPOSITED  energy  weight  (  the  § 3.19
            // fix:  the  LOCAL  source  importance  --  the
            // Jiang  formal - solution  logic.   The  passing
            // luminosity  weights  beamed  the  deep  cells
            // the  rays  merely  cross  (  midplane  550 K  vs
            // FLD  104 K );   the  deposit  dL  =  lumin  ( 1 -
            // e^-tau )  is  the  local  emission  driver ):
            float_t w( 0 );
            for( int i_hv = 0; i_hv < n_hv; ++ i_hv )
            {
                const auto tau_t = __max1
                    ( mfpa_i[ i_hv ] * dl[ a_proc ], 0 );
                w += lumin( i_hv )
                     * ( type::float_t( 1 )
                         - __exp1( - tau_t ) );
            };
            if( w > 0 )
            {
                const auto d0( this->dir[ 0 ] ),
                           d1( this->dir[ 1 ] ),
                           d2( this->dir[ 2 ] );
                atomicAdd( cm + 0, w );
                atomicAdd( cm + 1, w * d0 * d0 );
                atomicAdd( cm + 2, w * d0 * d1 );
                atomicAdd( cm + 3, w * d0 * d2 );
                atomicAdd( cm + 4, w * d1 * d1 );
                atomicAdd( cm + 5, w * d1 * d2 );
                atomicAdd( cm + 6, w * d2 * d2 );
            };
        };
        const auto *  hyd_u( prx.hyd.     u.at( i ) );
        const auto *  rho_s( hyd_u + prx. hyd.n_hva );
        auto       * sp_col( prx.rad.sp_col.at( i ) );
        for( int n = 0; n < n_sp; ++ n )
        {
            const auto col_old( column( n ) );
            const auto rho_i = __max1
                ( float_t( rho_s[ n ] ) , 0.f );
            column( n ) += dl[ a_proc ] * rho_i;
            utils::atomic_max
              ( sp_col + n, ( column( n ) + col_old ) / 2 );
        }
        return;
    };
    
    template< class map_T, class itg_T > __device__
    int proc( const map_T & bmap, const itg_T & itg )
    {
        auto & self = get_self_c(  );
        this->dest.i_rank = -1;
        geo_loc_t    g_l ;
        type::coord_t dl ;
        
        for( int  ib_old = -1; step > 0; -- step )
        {
            if( ib_l != ib_old && ib_l >= 0 )
            {
                g_l.load( bmap[ ib_l ].geo,      i );
                g_l.fix ( bmap[ ib_l ].geo, * this );
                ib_old = ib_l;
            }
            if( ! self.proc_step( dl, g_l, bmap, itg ) )
                break; 
        }
        return self.dest.i_rank;
    };
    
};                      // struct photon_t
};                      // namespace extension::radiation
