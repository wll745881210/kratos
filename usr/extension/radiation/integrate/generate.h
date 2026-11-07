#pragma once

#include "../radiation.h"
#include "../../chem_therm/chemistry.h"
#include "../../../../src/utilities/phys/unit.h"

namespace extension::radiation
{
////////////////////////////////////////////////////////////
//

template< class derived_T = crtp::dummy_t >
struct gen_t : crtp::helper_t
     < gen_t , derived_T, particle::generate::base_t >
{
    ////////// Type and data //////////
    __crtp_def_self__( gen_t, derived_T );
    using super_t = crtp::helper_t
        < gen_t, derived_T, particle::generate::base_t >;
    
    void *         data_h;
    int            n_scat;
    size_t         s_data;
    coord_t          dir0;

    int        n_ray_cell;  // Rays per cell side
    int            ax_inj;  // Normal axis of injection
    bool              rhs;  // RHS?

    phys::unit_t< type::float2_t > unit;
    std ::vector< type:: float_t > flux;    

    using super_t::step_max;
    using super_t::   n_par;    
    
    __host__ gen_t(  ) : data_h( nullptr ), s_data( 0 ) { };

    __host__ virtual void init
    ( const input & args, particle::base_t & mod ) override
    {
        super_t::init( args, mod );
        unit   . init( args );
        
        n_scat = args.get< int >
               ( "radiation", "n_scat",      1 );
        n_ray_cell = args.get< float_t >
               ( "radiation", "n_ray_cell",  1 );
        step_max = args.get< int > 
               ( "radiation", "step_max", 1024 );

        const auto flag_inj = args.get< std::string >
            ( "radiation", "injection", "z_out" );
        ax_inj = int( flag_inj[ 0 ] - 'x' );
        rhs    = ( flag_inj.find( "out" ) !=
                   std::string::npos );

        dir0 = coord_t( { 0., 0., -1. } );
        args ( "radiation", "dir", dir0 );
        const auto dir_norm = 1 / dir0.norm(  );
        for( int a = 0; a < 3; ++ a )
            dir0[ a ] *= dir_norm;
        data_h = nullptr ;
        
        args( "radiation", "flux_cgs", flux );
        const auto r_fac = args.get< type ::float2_t >
            ( "radiation", "flux_scale", 1 );
        for( auto & f_s : flux )
            f_s *= unit.derive( 0, 2, 1 ) / r_fac;
        return;
    };

    __host__ virtual void finalize
    ( particle::base_t & mod ) override
    {
        if( data_h != nullptr && s_data > 0 )
            mod.p_dev->free_host( data_h );
        return;
    };    

    template< class par_T > __host__
    void set_par( par_T & par )
    {
        if( par.n_hv > flux.size(  ) )
            throw std::runtime_error( "Error: lumin size" );
        
        const auto ds = par.proper;
        for( int n = 0; n < par.n_hv; ++ n ) // to code unit
            par.lumin ( n ) = flux[ n ] * ds;
        for( int n = 0; n < par.n_sp; ++ n )
            par.column( n ) = 0;

        par.step      = step_max;
        par.n_scat    =   n_scat;
        par.dest.todo = particle::to_keep;
        const auto  y = std::rand(  )
                      / ( float( RAND_MAX ) + 1.f );
        par.tau_s     = -logf( 0.99999f - 0.9999f * y );
        return;
    };

    template< class par_T, class geo_T, class fun_T >
    __host__ void pre_proc_par
    ( par_T & par, const geo_T & geo, 
      particle::base_t & mod, const fun_T & f_cache )
    {
        const int ax_p1 ( ( ax_inj + 1 ) % 3 );
        const int ax_p2 ( ( ax_inj + 2 ) % 3 );

        auto & i( par.i );
        type::coord_t xf,  dx;
        for( int a = 0; a < 3; ++ a )
        {
            xf[ a ] = geo.x_fc( a, i[ a ] );
            dx[ a ] = geo.dx_f( a, i      );
        }
        const auto ds = geo.surface( i, ax_inj );
        par.proper =  ds  / pow( n_ray_cell, 2 );
        // Use proper temporarily
        for( int j = 0; j < n_ray_cell; ++ j )
            for( int k = 0; k < n_ray_cell; ++ k )
            {
                par.x[ ax_p1 ] = xf[ ax_p1 ] + dx[ ax_p1 ]
                               * ( j + 0.5 ) / n_ray_cell;
                par.x[ ax_p2 ] = xf[ ax_p2 ] + dx[ ax_p2 ]
                               * ( k + 0.5 ) / n_ray_cell;
                f_cache( par );
            }
        return;
    };

    template< class com_T >
    __host__ size_t set_id_offset( com_T & f_com )
    {
        std::vector< size_t > n_par_all( f_com.n_rank(  ) );
        n_par_all[ 0 ] = n_par;
        if( f_com.rank(  ) == 0 )
            for( int i = 1; i < f_com.n_rank(  ); ++ i )
                f_com.irecv_h( & n_par_all[ i ], sizeof
                               ( n_par ), i, 0 );
        else
            f_com.isend_h( & n_par, sizeof( n_par ), 0, 0 );
        f_com. wait_all_h(  );

        size_t id_offset( 0 );
        if( f_com.rank(  ) == 0 )
            for( int i = 1; i < f_com.n_rank(  ); ++ i )
            {
                n_par_all[ i ] += n_par_all[ i - 1 ];
                f_com.isend_h ( & n_par_all[ i - 1 ],
                                sizeof( size_t ), i, 0 );
            }
        else
            f_com.irecv_h
                ( & id_offset, sizeof( size_t ), 0, 0 );
        f_com.wait_all_h(  );
        return id_offset;
    };

    template< class pol_T > __host__ void
    generate_once ( pol_T & pool, particle::base_t & mod )
    {
        if( n_ray_cell == 0 )
        {
            data_h = this;
            return;
        }
        if( mod.p_mesh->p_com->is_root(  ) )
            std::cout << "Generating/copying photons once"
                      << " ... "<< std::flush;

        auto & rad  = dynamic_cast
             < ::extension::radiation_t & > ( mod );
        auto & chem = ( * rad .p_che );
        const auto n_hv = rad .hv_eV  .size(  );
        const auto n_sp = chem.species.size(  );

        using par_t = typename pol_T::par_t;
        std::vector< par_t > par_cache;
        const int ax      ( ax_inj ) ;
        const int ax_p1 ( ( ax_inj + 1 ) % 3 );
        const int ax_p2 ( ( ax_inj + 2 ) % 3 );
        for( auto & d : mod )
        {
            const auto & geo( d.h(  ).geo );
            const auto & reg( d.h(  ).reg );
            const auto & reg_max = mod.p_mesh->tree
                       (  ).region_max( reg.level ) ;
            if( reg[ ax_inj ] != reg_max [ ax_inj ] - 1 )
                continue;
            
            type::idx_t i_s( { 0, 0, 0 } );
            type::idx_t i_e( { 0, 0, 0 } );
            
            i_s[ ax    ] = rhs ? geo.n_ceff[ ax ] - 1 : 0;
            i_e[ ax    ] = i_s[ ax ] + 1;
            i_e[ ax_p1 ] = geo.n_ceff[ ax_p1 ];
            i_e[ ax_p2 ] = geo.n_ceff[ ax_p2 ];

            par_t       par;
            par.n_hv = n_hv;
            par.n_sp = n_sp;
            par.dest.todo  = particle::to_keep;
            for( int a = 0; a < 3; ++ a )
                par.dir[ a ]  = dir0[ a ];
            
            par.ib_l = d.b_w.get(  ).id_l;
            if( rhs )
                par.x[ ax ] = geo.x_fc( ax, i_e[ ax ] ) 
                            - geo.dx_f( ax, i_e ) * 1e-2;
            else
                par.x[ ax ] = geo.x_fc( ax, i_s[ ax ] ) 
                            + geo.dx_f( ax, i_s ) * 1e-2;
            for( auto i : utils::loop_t( i_s, i_e ) )
            {
                for( int a = 0; a < 3; ++ a )
                    par.i[ a ] = i[ a ];
                get_self_c(  ).pre_proc_par
                ( par, geo, mod, [ & ] ( auto & p )
                {
                    p.id = par_cache.size (   );
                    par_cache . push_back ( p );
                }   );
            }
        }
        n_par = par_cache.size(  );
        pol_T pool_h;
        pool  .set_n_int( n_hv, n_sp );
        pool_h.set_n_int( n_hv, n_sp );
        pool_h.reserve_host( mod,  n_par + 1 );
        auto & par_tmp     ( pool_h[ n_par ] ); // tmp var
        par_tmp.n_hv =    n_hv;
        par_tmp.n_sp =    n_sp;
        par_tmp.set_offset(  );

        auto & f_com( * mod.p_mesh->p_com );
        const auto id_offset = set_id_offset( f_com );

        for( size_t i = 0; i < n_par; ++ i )
        {
            par_cache.at( i ).cp_shallow( par_tmp );
            get_self_c  (   ).set_par   ( par_tmp );
            par_tmp.id += id_offset;
            par_tmp.save( pool_h [ i ] );
        }

        pool.set_mem( mod, n_par );
        s_data = pool.size   (   );
        data_h = &  pool_h   [ 0 ];
        auto n_par_global( n_par );
        f_com.reduce_all ( n_par_global, comm::sum );
        if( mod.p_mesh->p_com->is_root(  ) )
            std::cout << " Done with " << n_par_global
                      << " photons. "<< std::endl;
        return;
    };

    template< class pol_T, class  map_T > __host__
    void generate ( pol_T & pool, const map_T & bmap,
                    particle::base_t  & mod )
    {
        if( data_h == nullptr )
            generate_once( pool, mod );
        pool.n_par = n_par;
        return mod.p_dev->a_cp
             ( & pool[ 0 ], data_h, s_data, mod.stream );
    };
};                      // struct gen_t

};                      // namespace extension::radiation
