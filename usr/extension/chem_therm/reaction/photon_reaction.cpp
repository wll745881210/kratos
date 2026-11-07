#include "../../../../src/utilities/functions.h"
#include "../../algo/interp.h"
#include "../../radiation/radiation.h"
#include "../chemistry.h"
#include "reaction.h"
#include "reaction_list.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// Derived reaction type

__host__ bool phot_reac_t::parse_ph
( list_t      &  rls, const std::string         & label,
  const input & rarg, const ::chemistry::base_t & chem )
{
    std::vector< int >       i_chem;
    std::vector< std::string > reacs, prods;
    rarg( label, "reactants",  reacs );
    rarg( label, "products",   prods );

    bool is_photon_reac( false );
    for( const auto & l : { reacs, prods } )
        for( const auto & r :  l )
        {
            if( r.find( "PHOT" ) != std::string::npos )
                is_photon_reac = true;
            else
            {
                if( chem.species.exist( r ) )
                   i_chem.push_back( chem.species.at( r ) );
                else
                    return false;
            }
        }
    if( ! is_photon_reac )
        return false;
    
    const auto & hv
        ( dynamic_cast< const extension::chem_t & >
        ( chem ).p_rad->hv_eV );
    std::vector< f_t >  lg_hv, lg_sig, lg_ene, lg_mul;
    rarg( label, "lg_hv",          lg_hv  ) ; 
    rarg( label, "lg_sigma",       lg_sig ) ;
    rarg( label, "lg_energy",      lg_ene ) ;
    if( ! rarg( label, "lg_multi", lg_mul ) )
        for( int i = 0; i < lg_hv.size(   ) ; ++ i )
            lg_mul.push_back( 0 );

    extension::interp_t
        < type::float2_t, 1 > f_lgsig, f_lgene, f_lgmul;
    f_lgsig.set_fill( -1e3 );
    f_lgene.set_fill( -1e3 );
    f_lgmul.set_fill(    0 );    
    f_lgsig.setup_non_uni( lg_hv.size(  ), lg_hv, lg_sig );
    f_lgene.setup_non_uni( lg_hv.size(  ), lg_hv, lg_ene );
    f_lgmul.setup_non_uni( lg_hv.size(  ), lg_hv, lg_mul );

    //////////////////////////////
    // Set shielding
    std::vector< shield_t > shield;

    i_sh             = -1;
    n_shield         =  0;
    i_hv_sh_lim[ 0 ] =  0;
    i_hv_sh_lim[ 1 ] =  0;
    if( reacs[ 0 ] == "H2" &&
        prods[ 0 ] == "H"  && prods[ 1 ] == "H" ) // not H+
    {
        lg_hv.push_back( 11.2 ); // Not dissociating below
        lg_hv.push_back( 14.5 ); // Ionizing above
        sh_kind  = hydrogen;
        n_shield =        1;
    }   // H2 self-shielding does not need shield_t
    else if( rarg.found( label, "shield_sps" ) )
    {
        sh_kind    = fitted;
        std::vector< std::string > sh_sps  ;
        rarg( label, "shield_sps", sh_sps );
        for ( const   auto & sp :  sh_sps )
        {
            if( ! chem.species.exist( sp ) )
                continue;
            std::array < type::float_t , 6 > v_t  ;
            rarg( label, "shield_par_" + sp, v_t );
            auto & tmp =  shield.emplace_back(   );
            tmp.i_sp = chem.species.at( sp );
            tmp.N1 = v_t[ 0 ];
            tmp.a  = v_t[ 1 ];
            tmp.x  = v_t[ 2 ];
            tmp.N2 = v_t[ 3 ];
            tmp.b  = v_t[ 4 ];
            tmp.y  = v_t[ 5 ];
            ++ n_shield;
        }
    }
    else
        sh_kind  = none ;

    if( sh_kind != none )
    {
        const auto hv_sh_min = pow( 10, lg_hv     [ 0 ] );
        const auto hv_sh_max = pow( 10, lg_hv.back(   ) );
        for( int n = 0; n < hv.size(  ); ++ n )
        {
            if( hv[ n ] < hv_sh_min )
                i_hv_sh_lim[ 0 ] = n + 1 ;
            if( hv[ n ] > hv_sh_max )
            {
                i_hv_sh_lim[ 1 ] = n - 1;
                break;
            }
        }
        i_sh = ( rls.l_ph_shield ++ );
    }

    //////////////////////////////
    // Set the common parameters

    this->n_chem  = i_chem.size(  );
    this->n_reac  = this->n_chem - prods.size(  );
    this->n_pars  = hv    .size(  );
    this->n_enes  = hv    .size(  );

    const int s_i( sizeof(      int ) );
    const int s_f( sizeof(      f_t ) );
    const int s_s( sizeof( shield_t ) );
    this->shield = ( shield_t * ) malloc( size(  ) );
    this->  pars = ( f_t * )
        ( ( char * ) ( this->shield ) +
          utils::align( n_shield * s_s, s_f ) );
    this->  enes = this->pars + n_pars;
    this->  mult = this->enes + n_enes;    
    this->f_chem = ( f_t * ) ( this->mult + n_enes );
    this->i_chem = ( int * ) (     f_chem + n_chem );    
    for( int n = 0; n < this->n_pars; ++ n )
    {
        pars[ n ] = pow( 10, f_lgsig( log10( hv[ n ] ) ) );
        enes[ n ] = pow( 10, f_lgene( log10( hv[ n ] ) ) );
        mult[ n ] = pow( 10, f_lgmul( log10( hv[ n ] ) ) );
    }
    if( sh_kind == fitted )
        for( int n = 0; n < this->n_shield; ++ n )
            this->shield[ n ] = shield.at( n );

    std::vector< f_t > f_prod;
    rarg( label, "fac_prod", f_prod );
    auto  f_chem =  reduce_reac( i_chem, n_reac, f_prod );
    memcpy( this->i_chem, i_chem.data(  ), n_chem * s_i );
    memcpy( this->f_chem, f_chem.data(  ), n_chem * s_f );
    return  true;
}

////////////////////////////////////////////////////////////
// Memory access

__host__ void phot_reac_t::setup( void * p )
{
    const auto d_sh = utils::align
        ( n_shield * sizeof( shield_t ), sizeof( f_t ) );

    shield = ( shield_t * )
           ( ( char * ) ( p )  + intptr_t( enes ) );
    pars   = ( f_t  * ) ( ( char * ) ( shield ) + d_sh );
    enes   =   pars + n_pars;
    mult   =   enes + n_enes;
    f_chem = ( f_t  * ) (   mult + n_enes );
    i_chem = ( int  * ) ( f_chem + n_chem );
    return ;
}

__host__ void phot_reac_t::relocate_shifted( void * p0 )
{
    char  * p = ( char * )( p0 ) + intptr_t( enes );
    memcpy( p , shield, size(  ) );
    return std::free( shield );
}

__host__ size_t phot_reac_t::set_shift
( const size_t & shift )
{
    enes = ( f_t * ) ( intptr_t( shift  ) ) ;
    return shift + size(  );
}   // use enes to temporarily register the shift

__host__ size_t phot_reac_t::size(  )
{
    return utils::align( sizeof( shield_t ) * n_shield,
                         sizeof( f_t ) )
        +  sizeof( int ) *   n_chem
        +  sizeof( f_t ) * ( n_pars + 2 * n_enes + n_chem );
}

////////////////////////////////////////////////////////////
// Printing

__host__ void phot_reac_t::print
( const ::chemistry::sp_vec_t & sps )
{
    std::cout << "---------------------\n  hv + ";
    const auto p_sep = [ & ] ( const int i )
    {     return i == 0 ? "" :
               ( i != n_reac ? " + " : "  -->  " ); };
    const auto s_sep = [ & ] ( const int i, const int n )
    {     return i == n - 1 ? "" : ", "; };

    for( int i = 0; i < n_chem; ++ i )
        std::cout << p_sep( i ) << sps[ i_chem[ i ] ].name;
    std::cout << "\n  Sigma / cm^2      = [ ";
    for( int i = 0; i < n_pars; ++ i )
        std::cout << pars[ i ] << s_sep( i, n_pars );
    std::cout << " ]\n  Enes / (erg/reac) = [ ";
    for( int i = 0; i < n_enes; ++ i )
        std::cout << enes[ i ] << s_sep( i, n_enes );
    std::cout << " ]\n  Multiplier        = [ ";
    for( int i = 0; i < n_enes; ++ i )
        std::cout << mult[ i ] << s_sep( i, n_enes );    
    std::cout << " ]\n";
    if( sh_kind != none )
    {
        std::cout << "  Shielding involved: ";
        if( sh_kind == hydrogen )
            std::cout << "Standard H_2\n";
        else
        {
            std::cout << "Fitted shielding \n";
            for( int n = 0; n < n_shield; ++ n )
            {
                const auto & sh ( shield[ n ] );
                std::cout << "    " << sps[ sh.i_sp ].name
                          << ' '    << sh.N1 << ' ' << sh.a
                          << ' '    << sh.x  << ' ' << sh.N2
                          << ' '    << sh.b  << ' ' << sh.y
                          << '\n';
            }
        }
    }
    return;
}

};                      // namespace extension::chemistry
