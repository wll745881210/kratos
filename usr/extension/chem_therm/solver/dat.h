#pragma once

#include "../../../../src/modules/chemistry/solver/dat.h"
#include "../reaction/intp.h"
#include "../reaction/phot_cache.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
//

template< class f_T >
struct dat_t : ::chemistry::solver::dat_t< f_T >
{
    ////////// Types //////////
    using     f_t = f_T;    
    using super_t
    = ::chemistry::solver::dat_t< type::float_t >;

    ////////// Data //////////
    f_t          * de_dt_d; // 
    f_t             T_dust; // Dust grain temperature
    f_t             n_elec; // n CGS of free electrons
    f_t              n_tot; // Total number density in CGS
    int               n_hv;
    f_t          *     flx; // Flux of radiation    
    f_t          *   n_col;
    intp_t<  >   *  f_cool; // Cooling interpolators
    phot_cache_t * p_cache; // Cache the photon data

    ////////// Host-side functions //////////
    __host__ static int size_sh
    ( const int  & n, const int & n_th,
      const bool & cache_f )    
    {
        const auto res = super_t::size_sh
          ( n, n_th, cache_f ) + ( n + 1 ) * sizeof( f_t );
        return utils::align( res, 8 );
    };  // n_col for the column density cache.
    
    ////////// Device-side functions //////////
    template< class rls_T > __device__
    dat_t( const int  & dim, char * p, const rls_T & rls,
           const bool & cache_f )
    {
        n_col = ( f_t * )( this->setup( dim, p, cache_f ) );
        de_dt_d = ( f_t * )         ( n_col      +    dim );
        f_cool  = ( intp_t<  >   * )( de_dt_d    +      1 );
        p_cache = ( phot_cache_t * )( f_cool + rls.n_cool );
        flx     = ( f_t * )( p_cache + rls.phot_reac.size );
        return;
    };
};

};                       // namespace extension::chemistry

