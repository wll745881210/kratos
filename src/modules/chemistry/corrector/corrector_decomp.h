#pragma once
#include "../../../utilities/types/var_list.h"

namespace chemistry::solver::corrector
{
////////////////////////////////////////////////////////////
// Decomposition class

struct decomp_t
{
    //////////  Type  //////////
    using f_t = type::float_t;

    ////////// Data //////////
    int  i_map; // i_map 4 sp: indicate single-elem sp
    int   size; // i_map 4 el: idx   of single-elem sp
    f_t  * num;
    int  * idx;
    union
    {
        int    shift;
        int capacity;
    }   mem ;

    ////////// Host-side functions //////////
    __host__ decomp_t(  );
    __host__ void relocate_shifted    ( void       *  p0 );
    __host__ size_t set_shift         ( const size_t & s );
    __host__ void relocate( void *  n_new, void  * i_new );
    __host__ void add     ( const int & i, const f_t & n );
    __host__ void setup   ( void                   *  p0 );
};

////////////////////////////////////////////////////////////
// Decomposition lists

struct decomp_ls_t : utils::const_list_t< decomp_t, int >
{
    ////////// Type //////////
    using  super_t = utils::const_list_t< decomp_t, int >;
    ////////// Data //////////
    int        n_sp ;

    ////////// Functions //////////
    __host__ decomp_ls_t(  ) : super_t(  ), n_sp( 0 ) {   };
    __host__ void init( const  int & n_s, const int & n_e );
    __host__         decomp_t & s( const int & i ) ;
    __host__         decomp_t & e( const int & i ) ;
    __device__ const decomp_t & s( const int & i ) const
    {
        return ( * this )[ i ];        
    };
    __device__ const decomp_t & e( const int & i ) const
    {
        return ( * this )[ i + n_sp ];        
    };    
};
};                 // namespace chemistry::solver::corrector
