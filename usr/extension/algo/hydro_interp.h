#pragma once

#include "interp.h"

namespace extension
{
////////////////////////////////////////////////////////////
// Hydrodynamic initialization interpolator

template< class f_T, int n_f = 3 >
struct hyd_val_t
{
    ////////// Data //////////
    using this_t = hyd_val_t< f_T , n_f >;
    static const int        n_fld = n_f  ;
    f_T                       val [ n_f ];

    ////////// Functions //////////
    __host__ __device__ hyd_val_t(  )
    {
        for( int n = 0; n < n_fld; ++ n )
            this->val[ n ] = 0;
        return;
    };
    __host__ __device__ hyd_val_t( const f_T & val )
    {
        for( int n = 0; n < n_fld; ++ n )
             this->val[ n ] = val;
        return;
    };

    template< class g_T > __host__ __device__
    void operator += ( const g_T & src )
    {
        for( int n = 0; n < n_fld; ++ n )
            val[ n ] += src [ n ];
        return;
    };
    template< class g_T >  __host__ __device__
    void operator *= ( const g_T & g )
    {
        for( int n = 0; n < n_fld; ++ n )
            val[ n ] *= g ;
        return;
    };

    __host__ __device__ const f_T & operator [  ]
    ( const int & n ) const
    {
        return val[ n ];
    };
    __host__ __device__ f_T & operator[  ] ( const int & n )
    {
        return val[ n ];
    };
};

template< class val_T, int dim_max >
struct interp_hyd_t : interp_t< val_T, dim_max >
{
    using   val_t = val_T;
    using super_t = interp_t< val_t, dim_max >;
    using super_t ::size;
    using super_t :: dim;

    ////////// Data //////////
    enum { press = 0, temp = 1 } therm_type ;

    __host__ __device__ __forceinline__
    constexpr int n_dim(  ) const
    {
        return dim;
    };
    
    __host__  virtual bool read_bin
    ( binary_io::base_t & bio, const std::string & pre )
    {
        if( ! bio.has_tag( pre + "n_pts" ) )
            return false;
        using x_t = typename super_t::x_t;

        dim = this->read_auto
            ( bio, this->n, pre + "n_pts" );
        if( dim > dim_max )
            throw std::runtime_error
                ( "More dims than dim_max" );        
        size = 1;
        for( int a = 0; a < dim; ++ a )
            size *= this->n [ a ];
        for( int a = 0; a < dim; ++ a )
        {
            this->x0[ a ] =  0;
            this->dx[ a ] = -1;
            this->xf[ a ] = new x_t [ this->n[ a ] ];
            this->read_auto( bio, this->xf[ a ], pre + "x_"
                             + std::to_string( a ) );
        }
        store_data( bio, pre );
        return true;
    }

    __host__  virtual void store_data
    ( binary_io::base_t & bio, const std::string & pre )
    {
        const auto n_fld( val_t::n_fld );

        using f_t = std::remove_reference_t
                  < decltype( * this->dat->val ) >;
        auto * b = new f_t[ size * n_fld ];
        therm_type
          = ( bio.has_tag( pre + "pre" ) ? press : temp );
        this->read_auto( bio, b           , pre + "rho" );
        if( therm_type == press )
            this->read_auto( bio, b + size, pre + "pre" );
        else
            this->read_auto( bio, b + size, pre +   "T" );
        this->read_auto( bio, b + 2 * size, pre + "vel" );

        this->dat = new val_t[ size ];
        for( decltype( size ) i = 0; i < size; ++ i )
            for( int n = 0; n < n_fld; ++ n )
                this->dat[ i ][ n ] = b[ n * size + i ];
        delete [  ] b;
        return ;
    };

    __host__ virtual void read_file
    ( const std::string & file, const std::string & pref )
    {
        try
        {
            super_t::read_file( file, pref );
        }
        catch( ... )
        {
            return;
        }
    };    

    __host__ __device__ operator bool(  ) const
    {
        return ( this->dat != nullptr );
    };
};

};                              // namespace utils
