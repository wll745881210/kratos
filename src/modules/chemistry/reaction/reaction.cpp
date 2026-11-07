#include <map>
 #include "../../../utilities/phys/constants.h"
#include "../../../utilities/string.h"
#include "../chemistry.h"

namespace chemistry::reaction
{
////////////////////////////////////////////////////////////
// Host-side functions

using f_t   = base_t::f_t;

__host__ base_t::base_t(  ) : f_chem( nullptr ),
                              i_chem( nullptr )
{
    return;
}

__host__ bool base_t::parse
( const std::string & label ,
  const input & rarg, const ::chemistry::base_t & chem )
{
    if( label.find( "reaction" ) == std::string::npos )
        return false;
    const auto f_ls = [ & ] ( const auto & kwd )
    {
        std::vector< std::string > res;
        rarg( label, kwd, res  ) ;
        return  res;
    };
    const auto reacs = f_ls ( "reactants"  );
    const auto prods = f_ls ( "products"   );
    const auto paras = f_ls ( "parameters" );
    const auto eners = f_ls ( "energy"     );
    const auto f_pds = f_ls ( "fac_prod"   );
    
    std::vector< int >           i_chem;
    std::vector< f_t > pars, enes, f_ps;

    for( const auto & l : {  reacs, prods } )
        for( const auto & r  :  l )
            if( chem.species.exist( r ) )
                i_chem.push_back( chem.species.at( r ) );
            else
                return false;
    const auto i_chem_old = i_chem;
    for( const auto  & p : paras )
        pars.push_back ( std::atof( p.c_str(  ) ) );
    for( const auto  & e : eners )
        enes.push_back ( std::atof( e.c_str(  ) ) );
    for( const auto  & f : f_pds )
        f_ps.push_back ( std::atof( f.c_str(  ) ) );
    setup_pars  ( chem, i_chem, reacs.size (  ) ,
                  pars,   enes, f_ps );

    ene_mode = poly;
    const auto emode_s = rarg.get< std::string >
               ( label, "ene_mode", "log" );
    if( emode_s.find( "log" ) != std::string::npos )
        ene_mode = logarithm;
    else if( emode_s.find( "pol" ) != std::string::npos )
        ene_mode = poly;    
    return true;
}

__host__ std::vector< f_t >   base_t::reduce_reac
( const std::vector < int > & i_chem , const int & n_reac,
  const std::vector < f_t > & f_prod )
{
    std::vector < f_t > f_chem;
    if( f_prod.size(  ) > 1 )
    {
        for( int i = 0; i < n_reac; ++ i )    
            f_chem.push_back(  -1 );
        for( const auto & f_p : f_prod )
            f_chem.push_back( f_p );
        return f_chem;
    }
    std::vector< int > n_ch_r( i_chem.size(  ) );
    for( int i = 0; i < n_reac; ++ i )
        n_ch_r[ i ] = -1;
    for( int i = n_reac; i < n_chem; ++ i )
    {
        n_ch_r[ i ] = 1;
        for( int j = 0; j < n_reac; ++ j )
        {
            if( i_chem[ j ] == i_chem[ i ] &&
                n_ch_r[ j ] <  0 )
            {
                ++ n_ch_r[ j ];
                -- n_ch_r[ i ];
                break;
            }
        }
    }
    for( const auto & n_ch_rs : n_ch_r )
        f_chem.push_back( n_ch_rs );
    return f_chem;
}

__host__ void base_t::setup_pars
( const ::chemistry::base_t &   chem ,
        std::vector < int > & i_chem , const int & n_reac,
  const std::vector < f_t > &   pars ,
  const std::vector < f_t > &   enes ,
  const std::vector < f_t > & f_prod )
{
    this->n_reac = n_reac;
    this->n_pars =   pars.size(  );
    this->n_enes =   enes.size(  );
    this->n_chem = i_chem.size(  ); // before reduce_reac
    auto  f_chem = reduce_reac( i_chem, n_reac, f_prod );

    // Merge duplicate product species (reactants untouched)
    {
        const int n_chem_old = i_chem.size(  );
        int n_chem_new = n_reac;
        for( int i = n_reac; i < n_chem_old; ++ i )
        {
            int  j;
            for( j = n_reac; j < n_chem_new; ++ j )
                if( i_chem[ i ] == i_chem[ j ] )
                {
                    f_chem[ j ] += f_chem[ i ];
                    break;
                }
            if( j == n_chem_new )
            {
                i_chem[ n_chem_new ] = i_chem[ i ];
                f_chem[ n_chem_new ] = f_chem[ i ];
                ++ n_chem_new;
            }
        }
        i_chem.resize( n_chem_new );
        f_chem.resize( n_chem_new );
    }
    this->n_chem = i_chem.size(  );
    
    const int s_i( sizeof( int ) );
    const int s_f( sizeof( f_t ) );
    this->  pars = ( f_t * ) malloc( s_i * n_chem +
                 s_f * ( n_chem + n_pars + n_enes ) );
    this->  enes = this->pars + n_pars;
    this->f_chem = this->enes + n_enes;    
    this->i_chem = ( int * ) ( this->f_chem + n_chem );
    memcpy( this->i_chem, i_chem.data(  ), n_chem * s_i );
    memcpy( this->f_chem, f_chem.data(  ), n_chem * s_f );
    memcpy( this->  pars,   pars.data(  ), n_pars * s_f );
    memcpy( this->  enes,   enes.data(  ), n_enes * s_f );
    return;
}

__host__ void base_t::print( const sp_vec_t & sps )
{
    std::cout << "---------------------\n  ";
    const auto p_sep  = [ & ] ( const int i )
    {     return i == 0 ? "" :
               ( i != n_reac ? " + " : "  -->  " ); };
    const auto s_sep  = [ & ] ( const int i, const int n )
    {     return i == n - 1 ? "" : ", "; };

    std::map< int, int > n_canceled;
    for( int j = 0; j < n_reac; ++ j )
        if( fabs( f_chem[ j ] ) < 0.01 )
            ++ n_canceled[ i_chem[ j ] ];

    const auto fmt_coef = []( const f_t v ) -> std::string
    {
        char buffer[ 12 ];
        const f_t tol = 0.01;
        if( fabs( v - round( v ) ) < tol )
            sprintf( buffer, "%d ", ( int ) round( v ) );
        else
            sprintf( buffer, "%0.1f ", v );
        return buffer;
    };

    const auto f_coef = [ & ] ( const int i )
    {
        std::string res( "" );
        f_t v = f_chem[ i ];
        if( i >= n_reac )
        {
            auto it = n_canceled.find( i_chem[ i ] );
            if( it != n_canceled.end(  ) )
                v += it->second;
        }
        else if( fabs( v ) < 0.01 )
            v = -1.0;
        if( fabs( fabs( v ) - 1 ) > 0.01 &&
            fabs( v ) > 0.01 )
            res = fmt_coef( v );
        return res;
    };
    
    for( int i = 0; i < n_chem; ++ i )
        std::cout << p_sep( i ) << f_coef( i )
                  << sps[ i_chem[ i ] ].name;
    std::cout << "  \n  Coef  = [ ";
    for( int i = 0; i < n_chem; ++ i )
        std::cout << f_chem[ i ] << s_sep( i, n_chem );
    std::cout << " ]\n  Pars  = [ ";
    for( int i = 0; i < n_pars; ++ i )
        std::cout <<   pars[ i ] << s_sep( i, n_pars );
    std::cout << " ]\n  Enes  = [ ";
    for( int i = 0; i < n_enes; ++ i )
        std::cout <<   enes[ i ] << s_sep( i, n_enes );
    std::cout << " ]" << std::endl;
    return;
}

__host__ void base_t::setup( void * p )
{
    pars   = ( f_t  * )
           ( ( char * ) ( p )  + intptr_t( enes ) );
    enes   = ( f_t  * ) (   pars + n_pars );
    f_chem = ( f_t  * ) (   enes + n_enes );
    i_chem = ( int  * ) ( f_chem + n_chem );
    return ;
}

__host__ void base_t::relocate_shifted( void * p0 )
{
    char  * p = ( char * )( p0 ) + intptr_t( enes ) ;
    memcpy( p , pars , n_chem            * sizeof( int ) +
            ( n_pars + n_enes + n_chem ) * sizeof( f_t ) );
    return std::free( pars );
}

__host__ size_t base_t::set_shift( const size_t & shift )
{
    enes = ( f_t * ) ( intptr_t( shift  ) ) ;
    return shift + sizeof( int )    * n_chem
         + sizeof( f_t ) * ( n_pars + n_enes + n_chem );
}   // use enes to temporarily register the shift

};                         // namespace chemistry::reaction
