#pragma once

#include "../../../src/types.h"
#include <cmath>

namespace extension
{
////////////////////////////////////////////////////////////
// STD cooling curve utility

using type::float_t;

struct cool_t
{
    ////////// Data //////////
    int        n_T;
    float_t   lnT0;
    float_t   dlnT;
    float_t * lam0;
    float_t * lam1;

    ////////// Host-side Function //////////
    __host__ cool_t(  ) : n_T( 0 ) {  };
    __host__ virtual void read( const std::string & file )
    {
        std::ifstream fin( file.c_str(  ) );
        if( ! fin )
            throw std::runtime_error( "cool_file " + file );
        std::vector < float_t > lnT, lam0, lam1;
        while( true )
        {
            float_t lgT( 0 ), lglam0( 0 ), lglam1( 0 );
            fin  >> lgT >>    lglam0 >>    lglam1;
            if( ! fin )
                break;
            lnT .push_back( log( 10. ) *  lgT   );
            lam0.push_back( pow( 10. , lglam0 ) );
            lam1.push_back( pow( 10. , lglam1 ) );
        }
        lnT0 = lnT.at  ( 0 ) ;
        n_T  = lnT.size(   ) - 1;
        dlnT = lnT.at  ( 1 ) - lnT0;
        this->lam0 = new float_t[ lnT.size(  ) ];
        this->lam1 = new float_t[ lnT.size(  ) ];
        for( int i = 0; i < lnT.size(  ); ++ i )
        {
            this->lam0[ i ] = lam0[ i ];
            this->lam1[ i ] = lam1[ i ];
        }
        return fin.close(  );
    };
    __host__ virtual void set_mem( device::base_t & dev )
    {
        const auto n =  n_T + 1;
        auto  * lam0 =  dev.malloc_const  < float_t > ( n );
        auto  * lam1 =  dev.malloc_const  < float_t > ( n );
        dev.f_cc( lam0, this->lam0, sizeof( float_t ) * n );
        dev.f_cc( lam1, this->lam1, sizeof( float_t ) * n );
        
        delete  [  ] this->lam0;
        delete  [  ] this->lam1;
        this->lam0 = lam0;
        this->lam1 = lam1;
        return;
    };

    ////////// Dual-side functions //////////
    __host__ __device__ __forceinline__
    operator bool(  ) const
    {
        return ( n_T > 0 );
    };
    __forceinline__ __host__ __device__
    int idx( const float_t & lnT ) const
    {
        int i_l = ( lnT - lnT0 ) / dlnT;
        i_l     = ( i_l <    0 ?       0 : i_l );
        i_l     = ( i_l >= n_T ? n_T - 1 : i_l );
        return i_l;
    };
    __forceinline__ __host__ __device__
    float_t lnT_i( const int & idx ) const
    {
        return lnT0 + idx * dlnT;
    };
    __forceinline__ __host__ __device__ void cool_func
    ( float_t & lnlam, float_t & dlnlam_dlnT,
      const float_t & lnT, const float_t & z = 1 ) const
    {
        const auto &    i_l = idx ( lnT     );
        const auto & lam0_l = lam0[ i_l     ];
        const auto & lam1_l = lam1[ i_l     ];
        const auto & lam0_r = lam0[ i_l + 1 ];
        const auto & lam1_r = lam1[ i_l + 1 ];
        const auto & lnlam_l
              = log( lam0_l + z * ( lam1_l - lam0_l ) );
        const auto & lnlam_r
              = log( lam0_r + z * ( lam1_r - lam0_r ) );

        dlnlam_dlnT = ( lnlam_r - lnlam_l ) / dlnT;
        lnlam       = lnlam_l + dlnlam_dlnT
                    * ( lnT - lnT_i( i_l ) );
        return;
    };

    template< class f_T > __device__ void operator(  )
    ( f_T & T , f_T  t_rem , const f_T & nb_cgs,
      const f_T &   cv_cgs , const f_T &   z_z0,
      const int &   n_iter ) const
    {
        float_t lnT( log( T ) );
        float_t lnlam( -100 ), dlnlamdlnT( 0 );        
        for( int iter = 0; iter < n_iter; ++ iter )
        {
            cool_func( lnlam, dlnlamdlnT, lnT, z_z0 );
            if( lnlam < -87 )       // lam < 1e-38
                break ;

            const auto dlnT_dt = -__exp1( lnlam ) * nb_cgs
                * nb_cgs / ( __exp1( lnT ) * cv_cgs );
            // Modified Backward Euler to prevent unphysical
            // situations when ( ddlnT * dt ) > 1
            const auto ddlnT = dlnT_dt * ( dlnlamdlnT - 1 );

            auto h( t_rem );
            h = __min1( h, ( ddlnT > 0 ? 0.5f : 1.f )
                        /  __abs1( ddlnT ) );
            lnT += h * dlnT_dt / ( 1 - h * ddlnT );
            
            t_rem -= h;
            if( t_rem < 1e-3f * h )
                break;
        }
        T = exp( lnT );
        return;
    }
};
};                              // namespace utils
