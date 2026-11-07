#pragma once

#include "../../../utilities/phys/constants.h"
#include "../../../utilities/types/crtp.h"
#include "../../../utilities/functions.h"
#include "../../../io/args/input.h"
#include "../types.h"

namespace chemistry
{
////////////////////////////////////////////////////////////
// Declaration

class base_t;

namespace reaction
{
////////////////////////////////////////////////////////////
// Reactions

struct base_t
{
    //////////  Types  //////////
    using f_t = type::float_t;

    ////////// Data //////////
    f_t    *   pars; // Rate   parameters
    f_t    *   enes; // Energy Parameters
    int    * i_chem; // Indices of chemicals
    f_t    * f_chem; // Coefficient for all involved
    int      n_reac; // Reactants number
    int      n_chem; // Total number of species
    int      n_pars; // Number of rate   parameters
    int      n_enes; // Number of energy parameters
    // Eliminate "same reactant + product" cases
    enum { logarithm, poly } ene_mode;
    
    ////////// Host-side functions //////////
    __host__ base_t(  );
    __host__ virtual bool parse
    ( const std::string & l ,
      const input & rarg, const ::chemistry::base_t & che );
    __host__ virtual void setup_pars
    ( const ::chemistry::base_t &  chem,
            std::vector< int > & i_chem, const int & n_reac,
      const std::vector< f_t > &   pars,
      const std::vector< f_t > &   enes,
      const std::vector< f_t > & f_prod );
    __host__ virtual std::vector < f_t > reduce_reac
    ( const std::vector< int > & i_chem, const int & n_reac,
      const std::vector< f_t > & f_prod );
    __host__ virtual void setup             ( void   * p0 );
    __host__ virtual void relocate_shifted  ( void   * p0 );
    __host__ virtual size_t  set_shift( const size_t &  s );
    __host__ virtual void print( const sp_vec_t & species );
};

template< class derived_T = crtp::dummy_t >
struct std_t : base_t
{
    ////////// Type //////////
    __crtp_def_self__( std_t, derived_T );

    ////////// Device-side functions //////////
    __device__ __forceinline__
    f_t exp_( const f_t & x ) const
    {
        if( x < -85 )
            return 0;
        return __exp1( __min1( x, 74 ) );
    };
    __device__ __forceinline__
    f_t pow_( const f_t & x, const f_t & y ) const
    {
        return exp_( y * logf( __abs1( x ) ) );
    };

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    void prep( dat_T     & d, const bdt_T & ,
               const eos_T &, const rls_T & ) const {  };

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate_coef( dat_T     & d, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        return pars[ 0 ] *
        ( n_pars > 1 ? pow_( d.T / 300, pars[ 1 ] ) : 1 ) *
        ( n_pars > 2 ? exp_( -pars[ 2 ]     / d.T ) : 1 ) ;
    };

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t ene_reac( dat_T     & d, const bdt_T & ,
                  const eos_T &, const rls_T & ) const
    {
        if( n_enes == 0 )
            return 0;
        f_t res( 0 ), T_n( 1 );
        if( ene_mode == poly )
            for( int n = 0; n < n_enes; ++ n )
            {
                res += enes[ n ] * T_n;
                T_n *= d.T;
            }
        else // de = enes[ 0 ] + kb * T * ( enes[ 1 ]
        {    //    + enes[ 2 ] * log( T )** enes[ 3 ] )
            res = ( n_enes >= 2 ? enes[ 1 ] : 0 );
            if( n_enes >= 3 )
                res += enes[ 2 ] * pow_( log ( d.T ),
                   ( n_enes >= 4 ? enes[ 3 ] : f_t( 1 ) ) );
            if( res != 0 )
                res *= phys::cgs::kb * d.T;
            res = enes[ 0 ] + res;
        }   
        return res * phys::cgs::NA;
    };  // Output: "specific energy" conversion based on y

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t dlnk_dlnT( dat_T     & d, const bdt_T & ,
                   const eos_T &, const rls_T & ) const
    {
        return pars[ 1 ] + pars[ 2 ] / d.T;
    };

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate( dat_T       &   d, const bdt_T & bdt ,
              const eos_T & eos, const rls_T & rls ) const
    {
        f_t  res = get_self(  ).rate_coef
                 ( d, bdt, eos, rls );
        for( int i = 0; i < this->n_reac; ++ i )
             res *=  d. y [ this->i_chem [ i ] ]
                 * ( i > 0 ? d.rho_cgs : 1 );
        if( res == 0 )
            return 0 ;
        if( d.i_en > 0 )
        {
            const auto e_coef = get_self(  ).ene_reac
                              ( d , bdt, eos, rls );
            if constexpr( std::is_same_v< f_t, float > )
            {
                constexpr auto  f_max
                    = type::float_max  / 100 ;
                const auto tst = __abs1( res ) > 1 ?
                    __abs1( e_coef / ( f_max / res ) ) :
                    __abs1( e_coef /   f_max * res ) ;
                if( tst > 1 )
                    res = __abs1( f_max / e_coef )
                        * utils:: sgn ( res );
            }
            d.f_cache( d.i_en, res * e_coef );
        }
        for( int i = 0 ; i < n_chem; ++ i )
            d.f_cache( i_chem[ i ], res * f_chem[ i ] );
        return res;
    };

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    void jac( dat_T       &   d, const bdt_T & bdt ,
              const eos_T & eos, const rls_T & rls ) const
    {
        static constexpr f_t y_min( 1e-32f );

        const auto   & self = get_self(  );
        const auto r = self.rate( d, bdt, eos, rls );
        if( __abs1( r ) < y_min )
            return;

        f_t dr_de;
        if( d.i_en >= 0 && d.y[ d.i_en ] != 0 )
        {
            const auto & e = d.y[ d.i_en ];
            const auto e_r
                = self.ene_reac ( d , bdt, eos, rls );
            dr_de = r / ( e + y_min ) * eos.dlnT_dlne( d )
                  * self.dlnk_dlnT( d, bdt, eos, rls ) ;

            const f_t ene( r * e_r ) ;
            for ( int i = 0; i < n_reac; ++ i )
                atomicAdd( & d.J ( d.i_en,  i_chem[ i ] ),
                     ene / ( d.y[ i_chem[ i ] ] + y_min ) );
            atomicAdd
                  ( & d.J( d.i_en, d.i_en ), dr_de * e_r );
        }
        for( int i = 0; i < n_chem; ++ i )
        {
            for ( int j = 0;  j < n_reac; ++ j )
               atomicAdd( & d.J( i_chem[ i ], i_chem[ j ] ),
                          f_chem[ i ] * r /
                          ( d.y[ i_chem[ j ] ] + y_min ) );
            if( d.i_en >= 0 )
                atomicAdd( & d.J ( i_chem[ i ], d.i_en ),
                           dr_de * f_chem[ i ] );
        }
        return;
    };
};
};                         // namespace reaction
};                         // namespace chemistry
