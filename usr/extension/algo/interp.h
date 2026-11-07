#pragma once

#include "../../../src/device/device.h"
#include "../../../src/io/binary/binary_io.h"
#include "../../../src/utilities/mapping/mapping.h"

namespace extension
{
////////////////////////////////////////////////////////////
// Utility for multi-dimensional interpolation

template< class val_T , int dim_max,
          class x_T = type::float_t, class int_T = int >
struct interp_t
{
    ////////// Type //////////
    enum   loc_t { on_host, on_device, on_const };
    enum   bnd_t {  extrap,      fill,  nearest };
    using    x_t =   x_T;
    using  val_t = val_T;
    using this_t = interp_t< val_T, dim_max, x_T, int_T >;

    ////////// Data //////////
    bool              moved;  // For proper vector behavior
    bool               ijkl;
    int                 dim;
    bnd_t               bnd;
    int_T              size;
    loc_t           dat_loc;
    val_T          fill_val;

    int        n[ dim_max ];
    x_T       x0[ dim_max ];
    x_T       dx[ dim_max ];
    x_T     * xf[ dim_max ];
    val_T           *   dat;
    device::base_t  * p_dev;  // For safety on global GRAM

    ////////// Functions //////////
    __host__ void move_from( this_t & src )
    {
        cp_shallow_from( src );
        src.  moved    =  true;
        this->moved    = false;
        return;
    };

    __host__ __device__ void init_default( const int & dim )
    {
        dat       = nullptr;
        ijkl      =    true;
        bnd       =  extrap;
        fill_val  =       0;
        moved     =   false;
        this->dim =     dim;
        for( int a = 0; a < dim; ++ a )
        {
            x0[ a ] =       0;
            dx[ a ] =      -1;
            xf[ a ] = nullptr;
        }
        dat_loc = on_host;
        p_dev   = nullptr;
        return;
    };        

    __host__ __device__ interp_t
    ( const int & dim = dim_max )
    {
        init_default( dim );
    }

    __host__ __device__ interp_t( const this_t &  src )
    {
        cp_shallow_from( src );
        return;
    };
    __host__ interp_t( this_t && src ) noexcept
    {
        move_from( src );
        return;
    };
    // __host__ ~interp_t(  )
    // {
        // if( this->moved )
        //     return;
        // if( dat_loc == on_host )
        //     this->free(  );
        // else if( dat_loc == on_device && p_dev != nullptr )
        //     this->free_device( * p_dev );
        // return;
    // };

    __host__ __device__
    void cp_shallow_from( const this_t & src )
    {
        this->fill_val  = src.fill_val;
        this->dat_loc   = src. dat_loc;
        this->p_dev     = src.   p_dev;
        this->ijkl      = src.    ijkl;
        this->size      = src.    size;
        this->dim       = src.     dim;
        this->bnd       = src.     bnd;
        this->dat       = src.     dat;
        for( int i = 0; i < dim_max; ++ i )
        {
            this->n [ i ] = src.n [ i ];
            this->x0[ i ] = src.x0[ i ];
            this->dx[ i ] = src.dx[ i ];
            this->xf[ i ] = src.xf[ i ];
        }
        return;
    };
    __host__ void cp_deep_from( const this_t & src )
    {
        cp_shallow_from( src );
        for( int a = 0; a < dim_max; ++ a )
            if( src.xf[ a ] != nullptr )
            {
                xf[ a ] = alloc_h < x_T >( n[ a ] );
                memcpy( xf[ a ] , src.xf [ a ],
                        n [ a ] * sizeof ( x_T  ) );
            }
        dat = alloc_h < val_T > ( size );
        memcpy( dat, src.dat, size * sizeof( val_T ) );
        return;
    };

    __host__ this_t & operator = ( const this_t & src )
    {
        cp_deep_from( src );
        return   ( * this );
    };                          // Shallow copy only!

    template< class T >
    __host__ T * alloc_h( const size_t & s )
    {
        return ( T * ) std::malloc( sizeof( T ) * s );
    };

    template< class v_T, class  i_T,  class u_T > __host__
    void setup( const v_T & x0, const v_T &  dx ,
                const i_T &  n, const u_T & dat )
    {
        size = 1;
        if  constexpr( dim_max == 1 )
        {
            this->x0[ 0 ] = x0;
            this->dx[ 0 ] = dx;
            this->n [ 0 ] =  n;
            this->size    =  n;
        }
        else
            for( int a = 0; a < dim; ++ a )
            {
                this->x0[ a ] = x0[ a ];
                this->dx[ a ] = dx[ a ];
                this->n [ a ] = n [ a ];
                this->size   *= n [ a ];
            }
        this -> dat = alloc_h< val_T > ( size );
        memcpy( this->dat,   & dat [ 0 ] ,
                size *sizeof ( val_T ) ) ;
        return;
    };

    template< class tgt_T, class src_T >
    __host__ void cp_auto( tgt_T & tgt, const src_T & src,
                           const size_t & n )
    {
        if( sizeof( tgt[ 0 ] ) == sizeof( src[ 0 ] ) )
            memcpy( tgt, src, n * sizeof( src[ 0 ] ) );
        else
            for( size_t i = 0 ; i < n ; ++ i )
                tgt[ i ] = src[ i ];
        return;
    };

    template< class tgt_T >  __host__
    size_t read_auto( binary_io::base_t & bio,
                      tgt_T * tgt, const std::string & tag )
    {
        if( ! bio.has_tag( tag ) )
            return 0;
        const auto & b_info =  bio. info ( tag );
        const auto & s_tgt( sizeof( tgt[ 0 ] ) );
        const auto u_size ( size_t( b_info.u_size ) );

        if( s_tgt == u_size )
            return bio.read ( tgt, tag );

        char * buf = alloc_h< char >( b_info.size );
        bio.read( buf, tag );
        const auto f_tgt = [ & ] ( const auto & i )
        {
            auto * p  ( buf + i * u_size );
            auto & res( tgt [ i ] );
            if constexpr( std::is_integral_v< tgt_T > )
            {
                if( u_size == sizeof( int ) )
                    tgt[ i ] = ( * reinterpret_cast
                                 < int    * > ( p ) );
                else
                    tgt[ i ] = ( * reinterpret_cast
                                 < size_t * > ( p ) );
            }
            else if constexpr
                  ( std::is_floating_point_v< tgt_T > )
            {
                if( u_size == sizeof( float ) )
                    tgt[ i ] = ( * reinterpret_cast
                                 < float  * >  ( p ) );
                else
                    tgt[ i ] = ( * reinterpret_cast
                                 < double * >  ( p ) );
                if( std::isnan( tgt[ i ] ) )
                    throw std::runtime_error( "Bin NaN" );
            }
        };
        const  auto n_dat( b_info.size / u_size );
        for( size_t i = 0; i < n_dat; ++ i )
            f_tgt( i ) ;
        std::free( buf );
        return n_dat;
    };

    template< class i_T, class vv_T, class u_T > __host__
    void  setup_non_uni( const i_T &   n, const vv_T & xf,
                         const u_T & dat )
    {
        this->size = 1;
        if  constexpr( dim_max == 1 )
        {
            this->size    = n;
            this->n [ 0 ] = n;
            this->xf[ 0 ] = alloc_h < x_T >  ( n );
            cp_auto( this->xf[ 0 ], & xf[ 0 ], n );
        }
        else
            for( int a = 0; a < dim; ++ a )
            {
                this->size   *= n[ a ];
                this->n [ a ] = n[ a ];
                this->xf[ a ] = alloc_h < x_T >  ( n[ a ] );
                cp_auto
                  ( this->xf[ a ], & xf[ a ][ 0 ], n[ a ] );
            }
        this->dat = alloc_h < val_T > ( size );
        cp_auto( this->dat, & dat[ 0 ], size );
        return;
    };

    __host__ virtual bool read_bin
    ( binary_io::base_t & bio,
      const std::string & pref = "" )
    {
        if( ! bio.has_tag( pref + "n_pts" ) )
            return false;

        // Read the data-layout flag.
        // ijkl = true ( default ): dim-0 slowest,
        // C-order ( n[ 0 ],n[ 1 ],n[ 2 ]).
        // ijkl = false: dim-0 fastest (similar to dat_3d.h)
        // ( n[ 2 ], n[ 1 ], n[ 0 ]) C-order - z slowest, x
        // fastest. This allows to choose the convention
        // matching the data without transposing.
        if( bio.has_tag( pref + "ijkl" ) )
        {
            int   ijkl = 1;
            read_auto( bio, & ijkl, pref + "ijkl" );
            this->ijkl = ( ijkl != 0 );
        }
        dim  = read_auto( bio, n , pref + "n_pts" );
        if( dim > dim_max )
            throw std::runtime_error
                ( "More dims than dim_max" );
        size = 1;
        for( int a = 0; a < dim; ++ a )
            size *= n [ a ] ;

        if( bio. has_tag( pref + "x0" ) &&
            bio. has_tag( pref + "dx" )  )
        {     // Uniform
            read_auto( bio, x0, pref + "x0" );
            read_auto( bio, dx, pref + "dx" );
        }
        else  // Non-uniform
            for( int a = 0; a < dim; ++ a )
            {
                x0[ a ] =  0;
                dx[ a ] = -1;
                xf[ a ] = alloc_h< x_T > ( n[ a ] );
                read_auto( bio, xf[ a ], pref + "x_" +
                           std::to_string( a ) );
            }
        dat = alloc_h < val_T > ( size );
        read_auto( bio, dat, pref + "data" );
        return true;
    };

    __host__ virtual void read_file
    ( const std::string & file ,
      const std::string & pref = "" )
    {
        binary_io::default_t bio;
        bio.open( file, "r" );
        bio.load(           );
        read_bin( bio, pref );
        return bio.close(   );
    };

    __host__ void set_fill( const val_T & fill_val )
    {
        this->fill_val = fill_val;
        bnd            =     fill;
        return;
    };
    __host__ void set_nearest(  )
    {
        bnd = nearest;
        return;
    };

    template< class mem_T, class cp_T > __host__
    void to_device( const mem_T & f_mem, const cp_T & f_cp )
    {
        val_T * q = ( val_T * )
              f_mem ( sizeof( val_T ) * size );
        f_cp( q, dat, sizeof( val_T ) * size );
        std::free( dat );
        dat  = q ;
        for( int a = 0; a < dim; ++ a )
        {
            if( xf[ a ] == nullptr )
                continue;
            x_T * p =  ( x_T * )
                f_mem( sizeof( x_T ) * n[ a ] );
            f_cp( p, xf[ a ], sizeof( x_T ) * n[ a ] );
            std::free( xf[ a ] );
            xf[ a ] = p;
        }
        return;
    };
    __host__ void to_device( device::base_t & dev )
    {
        dat_loc = on_device;
        return to_device( [ & ]  ( const size_t & s )
        {  return dev.malloc_device< char > ( s ) ; },
        [ & ] ( void * t, const void * r, const size_t & s )
        {  dev.f_cp( t, r, s ); } );
    };
    __host__ void to_const ( device::base_t & dev )
    {
        dat_loc = on_const;
        return to_device( [ & ]  ( const size_t & s )
        {  return dev.malloc_const < char > ( s ) ; },
        [ & ] ( void * t, const void * r, const size_t & s )
        {  dev.f_cc( t, r, s ); } );
    };

    template< class free_T >
    __host__ void free( const free_T & f_free )
    {
        if( dat != nullptr )
            f_free( dat );
        dat = nullptr;
        for( int a = 0; a < dim; ++ a )
            if( xf[ a ] != nullptr )
            {
                f_free( xf[ a ] );
                xf[ a ] = nullptr;
            }
        return;
    };
    __host__ void free_device( device::base_t & dev )
    {
        if( dat_loc !=  on_device )
            free( [ & ]( void * p ) { dev.free( p ); } );
        p_dev = nullptr;
        return;
    };
    __host__ void free(  )
    {
        return free( [ & ]( void * p ){ std::free( p ); } );
    };

    template < class g_T >
    __host__ void operator *= ( const g_T & f )
    {
        for( int i = 0; i < size; ++ i )
            dat[ i ] *= f;
        return;
    };

    ////////// Host-device dual //////////
    __host__ __device__ __forceinline__
    const x_T & x_max( const int ax ) const
    {
        return xf[ ax ][ n[ ax ] - 1 ];
    };
    __host__ __device__ __forceinline__
    const x_T & x_min( const int ax ) const
    {
        return xf[ ax ][ 0 ];
    };

    template< class g_T >__forceinline__ __host__ __device__
    int find_idx( const g_T & x, const int & a ) const
    {
        int  i( 0 ) ;
        if( dx[ a ] > 0 )
        {
            i = int( ( x - x0[ a ] ) / dx[ a ] );
            i = ( i < 0 ? -1 // v-- RHS out-of-bnd flag 
                        : ( i >= n[ a ] - 1 ? -2 : i ) );
        }
        else
        {
            if( xf[ a ] == nullptr )
                return -1;
            else if( x < xf[ a ][ 0 ] )
                return bnd != extrap ? -1 :          0;
            else if( x >= xf[ a ][ n[ a ] -  1 ] )
                return bnd != extrap ? -2 : n[ a ] - 2;

            for( ; i < n[ a ]  - 1 ; ++ i )
                if( xf[ a ][ i + 1 ] >= x )
                    break;
        }
        return  i;
    };

    __host__ __device__ __forceinline__ void idx_frac
    ( int & i, x_T & f, const x_T & x, const int & a ) const
    {
        if( n[ a ] == 1 )
        {
            i  = 0;
            f  = 1;
            return;
        }
        i = find_idx( x, a );
        if( i < 0 )
        {
            if( bnd == fill )
                return;
            else if( bnd == nearest )
            {
                f = ( i == -1 ? 0 :          1 );
                i = ( i == -1 ? 0 : n[ a ] - 2 );
                return;
            }
            else                // extrapolation
                i = ( i == -1 ? 0 : n[ a ] - 2 );
        }
        const auto xl( dx[ a ]  > 0 ? dx [ a ] * i + x0[ a ]
                     : xf[ a ][ i ] ) ;
        const auto d ( dx[ a ]  > 0 ? dx [ a ]
                     : xf[ a ][ i + 1 ] - xl ) ;
        f = ( x - xl ) / d ;
        return;
    };

    __host__ __device__ __forceinline__
    int_T idx( const int i[  ] ) const
    {
        int_T res ( 0 );
        for( int a = 0; a < dim; ++ a )
        {
            const int b ( ijkl ?  a : dim - 1 - a );
            res = res * n[ b ] + i[ b ];
#ifdef    __CPU_DEBUG__
            if( i[ a ] < 0 || i[ a ] >= n[ a ] )
                throw std::runtime_error( "interp idx" );
#endif // __CPU_DEBUG__
        }
        return res;
    };

    template < class loc_T >
    __host__ __device__ __forceinline__
    val_T operator(  ) ( const loc_T & x ) const
    {
        int  i [ dim_max ];
        return ( * this )( i, x );
    };

    template < class idx_T, class loc_T >
    __host__ __device__ __forceinline__
    val_T operator(  ) ( idx_T & i, const loc_T & x ) const
    {
        const int m_dim( 1 << dim );
        int di[ dim_max ], j[ dim_max ];
        x_T f [ dim_max ];
        for( int a = 0; a < dim; ++ a )
        {
            if  constexpr( dim_max == 1 )
                idx_frac ( i[ 0 ], f[ 0 ], x, 0 );
            else
                idx_frac ( i[ a ], f[ a ], x[ a ], a );
            if( i[ a ] < 0 )
                return fill_val;
        }
        val_T res ( 0 );
        for ( int l = 0; l < m_dim; ++ l )
        {
            utils::muldim( di, l, dim );
            x_T  w( 1 );
            for( int a = 0; a < dim; ++ a )
            {
                w *=   ( di[ a ] ? f [ a ] :   1 - f[ a ] );
                j[ a ] = i [ a ] + di[ a ] * ( 1 < n[ a ] );
            }
            auto s = dat[ idx( j ) ];
            s   *= w ;
            res += s ;
        }
        return res;
    };

    __host__ __device__ __forceinline__
    explicit operator bool (  ) const
    {
        return this->dat != nullptr;
    };    
};                              // class  interp_t
};                              // namespace extension
