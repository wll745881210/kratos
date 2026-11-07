#include "../../../device/device.h"
#include "../../../utilities/string.h"
#include "../utilities/periodic.h"
#include "corrector.h"
#include "../chemistry.h"

namespace chemistry::solver::corrector
{
////////////////////////////////////////////////////////////
// Host-side functions

__host__ void  base_t::init
( const  input & args, ::chemistry::base_t & mod )
{
    i_elec_s =  -1;
    i_elec_e =  -1;
    y_min    = args.get< type::float_t >
               ( "chemistry", "y_floor", 1e-34 );
    
    const auto   & sps ( mod.species );
    std::map< int, int > e_map ;
    for( const auto & s :  sps )
    {
        for( const auto & [ z, n_e ] : s.elem )
            e_map [ z ] = 0 ;
        if( s.name == "e-" || s.name == "e" )
            i_elec_s = s.idx;
    }
    int i_e_count ( 0 ) ;
    for( auto & [ e, i_e ] : e_map )
    {
        i_e = ( i_e_count ++ );
        if( e == 0 )            // 0 means free charge
            i_elec_e = i_e;
    }

    mat_int_t sp_elem( e_map.size(  ), sps. size(  ) );
    for( size_t i = 0; i < sps.size(  ); ++ i )
        for( const auto & [ z, n_e ] : sps[ i ].elem )
            sp_elem( e_map[ z ], i ) = n_e;
    n_sp = sp_elem.n;
    n_el = sp_elem.m;
    
    if( args.get< bool >( "chemistry", "stoi_proj", false) )
        stoi.init ( * mod.p_dev, sp_elem );
    if( args.get< bool >( "chemistry", "corr_neg",  true ) )
        set_decomp( mod,  e_map, sp_elem );
    return;
}

__host__ void  base_t::set_decomp
( ::chemistry::base_t        &    chem ,
  const std::map< int, int > &    emap ,
  const mat_int_t            & sp_elem )
{
    const auto & sps( chem.species );
    auto  &    f_dev( * chem.p_dev );
    decomp.init( n_sp, n_el );
    for( int i = 0; i < n_sp; ++ i )
        decomp.s( i ).i_map = -1 ;

    std::map< int, int > e_map_sps, e_map_inv;
    for( const auto & [ z, i_e ] : emap )
    {
        e_map_inv[ i_e ] = z;
        for( const auto & sp : sps )
        {
            auto s = ( sp.name == "e-" ? "e" : sp.name );
            if( s == chem.elem_dat.elem( z ).name )
            {
                e_map_sps[ i_e ] = sp.idx ;
                decomp. s( sp.idx ).i_map = i_e;
            }   // Skip single-elem sp. for element decomp.
        }
    }
    for( const auto & [ z, i_e ] : emap )
        if( e_map_sps.find( i_e ) == e_map_sps.end(  ) )
        {
            int  n_el_min( 16384 ) , i_el_min( -1 );
            for( const auto & sp : sps )
            {
                if( sp.elem.size(  ) > 1 )
                    continue;
                auto it_e  = sp.elem.find( z )  ;
                if(  it_e != sp.elem.end (   ) &&
                     it_e->second < n_el_min )
                {
                    n_el_min = it_e->second;
                    i_el_min = sp.idx;
                }
            }
            if( i_el_min < 0 )
                throw std::runtime_error( "corr decomp" );
            e_map_sps[ i_e ] = i_el_min;
        }   // If no single-element species...
    for( int i = 0; i < n_sp; ++ i )
    {
        const  auto  & sp( chem.species.at( i ) );
        auto   & s = decomp.s( i );
        for( int j = 0; j < n_el; ++ j )
            if(  sp_elem( j, i )  != 0 )
            {
                auto num = type::float_t( sp_elem( j, i ) )
                           / sp_elem( j, e_map_sps[ j ] ) ;
                if( j == i_elec_e && i != i_elec_s )
                    num *=  -1 ;
                s.add( e_map_sps[ j ], num );
                decomp.e( j ). i_map = e_map_sps[ j ] ;
                if( decomp.e( j ).i_map != i )
                    decomp.e( j ).add( i, sp_elem( j, i ) );
            }   // The queue for compensating the negative
    }           // elem-species with most abundant ones
    decomp.regularize( f_dev );
    return;
}

};                 // namespace chemistry::solver::corrector
