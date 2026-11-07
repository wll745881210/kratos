#pragma once

namespace chemistry::hydro
{
////////////////////////////////////////////////////////////
// EoS with chemistry

struct cv_poly_t
{
    ////////// Data //////////
    int                 n_a;
    type::float_t     m_cgs;
    type::float_t clim[ 2 ];    // CV in limiting cases
    type::float_t Tlim[ 3 ];    // Temperature limits
    type::float_t  a_h[ 5 ];    // NASA polynomial
    type::float_t  a_l[ 5 ];    // NASA polynomial    
    
    ////////// Function //////////
    __host__ cv_poly_t(  ) : n_a( 0 )
    {
        clim[ 0 ] = 3 / 2.;
        clim[ 1 ] = 3;
        Tlim[ 0 ] = 1;
        Tlim[ 1 ] = 1e6;
        return;
    };
    
    template < class f_T > __host__ __device__
    type::float_t operator(  ) ( const f_T & T ) const
    {
        if( n_a < 1  || T < Tlim[ 0 ] )
            return clim[ 0 ];
        else if( T > Tlim[ 1 ] )
            return clim[ 1 ];
        f_T res( 0 ), T_n ( 1 );
        const auto * a( T > Tlim[ 2 ] ? a_h : a_l );
        for( int n = 0; n < n_a; ++ n )
        {
            res += a[ n ] * T_n;
            T_n *= T;
        }        
        return res;
    };

    template< class f_T > __host__ __device__
    type::float_t ddT( f_T T ) const
    {
        if( n_a < 1 )
            return clim[ 0 ];
        T = __max1( __min1( T, Tlim[ 1 ] ), Tlim[ 0 ] );
        f_T res( 0 ), T_n ( 1 );
        const auto * a( T > Tlim[ 2 ] ? a_h : a_l );        
        for( int n = 0; n < n_a; ++ n )
        {
            res += a[ n ] * T_n * ( n + 1 );
            T_n *= T;
        }
        return res;
    };

    __host__ void setup
    ( const input & pars, const species_t & sp )
    {
        m_cgs = sp .mass_cgs;
        
        std::vector< type::float_t >  a_l, a_h;
        if( ! pars( sp.name, "high",  a_h ) ||
            ! pars( sp.name, "low",   a_l ) || 
            ! pars( sp.name, "Tlim", Tlim ) )
        {
            clim[ 0 ] = ( sp.n_atom  <= 1 ? 3 :
                        ( sp.n_atom  >= 3 ? 6 : 5 ) ) / 2.;
            clim[ 1 ] = clim[ 0 ] // v-- Internal DoFs
                      + std::max( sp.n_atom - 1 , 0 ) / 2.;
            Tlim[ 0 ] = 1.0e2;
            Tlim[ 1 ] = 1.5e3;
            n_a       =     0;
            return;
        }
        n_a = a_h.size(  );
        if( n_a != a_l.size(  ) )
            throw std::runtime_error
                ( "Inconsistent dat size in cv_poly" );
        for( int i = 0; i < n_a; ++ i )
        {
            this->a_l[ i ] = a_l[ i ];
            this->a_h[ i ] = a_h[ i ];
        }
        clim[ 0 ] = ( * this )( Tlim[ 0 ] );
        clim[ 1 ] = ( * this )( Tlim[ 1 ] );
        return;
    };
};

struct cv_poly_list_t
{
    ////////// Data //////////
    int         n_sps;
    cv_poly_t * dat_d;
    cv_poly_t * dat_h;
    
    ////////// Host-side functions //////////
    __host__ cv_poly_list_t(  ) : dat_d( nullptr ) ,
                                  dat_h( nullptr ) {  };

    __host__ virtual void setup
    ( const input & args, const sp_vec_t & species,
      device::base_t & f_dev )
    {
        input pars( args.get< std::string >
                    ( "chemistry", "cv_file", "" ) );
        if( ! pars.read(  ) )
            return;

        n_sps = species.size(   );
        dat_h = f_dev.malloc_host< cv_poly_t > ( n_sps );
        std::vector< type::float_t > a;
        for( int n = 0; n < species.size(  ); ++ n )
            dat_h[ n ].setup( pars, species[ n ] ) ;

        auto s_dat = n_sps * sizeof( cv_poly_t );
        dat_d = f_dev.malloc_const < cv_poly_t >( n_sps );
        return  f_dev.f_cc( dat_d, dat_h, s_dat );
    };

    __host__ virtual void free( device::base_t & f_dev )
    {
        f_dev.free_host( dat_h );
        dat_h = nullptr;
        dat_d = nullptr;        
        return;
    };  // dat_d is on the constant memory, no need to free

    __host__ __device__ operator bool(  ) const
    {
        return dat_d != nullptr;
    };

    template < class v_T, class f_T > __host__
    type::float_t xn_h( const v_T & x, const f_T & T ) const
    {
        type::float_t norm( 0 ), tot( 0 );
        for( int n = 0; n < n_sps; ++ n )
        {
            const auto x_n( __max1( x[ n ], 0 ) );
            norm += x_n + type::float_min;
            tot  += x_n * dat_h[ n ]( T );
        }
        return tot / norm;
    };

    ////////// Device-side functions //////////
    template < class v_T, class f_T > __device__
    type::float_t xn( const v_T  &  x , const f_T & T ,
                      const bool & dT ) const
    {
        type::float_t norm( 0 ), tot( 0 );
        for( int n = 0; n < n_sps; ++ n )
        {
            const auto x_n( __max1( x[ n ], 0 ) );
            norm += x_n + type::float_min;
            tot  += x_n * ( dT ? dat_d[ n ].ddT( T )
                               : dat_d[ n ]( T ) ) ;
        }
        return tot / norm;
    };
    template< class v_T, class f_T > __device__ 
    type::float_t xn_d( const v_T & x, const f_T & T ) const
    {
        return xn( x, T, false );
    };
    template< class v_T, class f_T > __device__ 
    type::float_t dT_xn_d
    ( const v_T & x, const f_T & T ) const
    {
        return xn( x, T,  true );
    }; 
    template< class v_T > __device__
    type::float_t xn_d_simple( const v_T & x ) const
    {
        return xn( x, -1.f, false );
    };
};

};                         // namespace chemistry::hydro
