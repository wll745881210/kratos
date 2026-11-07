#include "boundary/keeper.h"
#include "types/block_data.h"
#include "operation/operation.h"
#include "multigrid.h"

namespace multigrid
{
////////////////////////////////////////////////////////////
// Functions

void base_t::read( const input & args )
{
    if( ! p_opt )
        enroll<   >(  );
    cycle          =  0;
    defect_lim     = ( typeid( type::float_t ) ==
                       typeid( double ) ? 5e-6  : 5e-4 );
    n_th           = args.get< int >
                   ( "multigrid", "thread_limit",   32 );
    n_iter         = args.get< int >
                   ( "multigrid", "n_iter",          1 );
    n_smooth       = args.get< int >
                   ( "multigrid", "n_smooth" ,       1 );
    defect_lim     = args.get< type::float_t >
                   ( "multigrid", "dtol",   defect_lim );
    cycle_calc_ave = args.get< int >
                   ( "multigrid", "cycle_calc_ave",  1 );
    cycle_per_deep = args.get< int >
                   ( "multigrid", "cycle_per_deep",  1 );
    dl_scratch     = args.get< int >
                   ( "multigrid", "dl_scratch",      1 );
    iter_first     = args.get< int >
                   ( "multigrid", "iter_first",     20 );
    is_test        = args.get< bool >
                   ( "multigrid", "is_test",     false );
    to_print       = args.get< bool >
                   ( "multigrid", "print_info",   true );

    const auto phy_bnd_type = args.get< std::string >
             ( "multigrid", "phy_bnd_type",  "free" );
    if( phy_bnd_type.find( "fre" ) != std::string::npos )
        p_opt->phy_bnd = operation::base_t::phy_bnd_t::free;
    return p_opt->init( args );
}

void base_t::finalize ( mesh::mesh_t & mesh )
{
    p_opt->finalize                ( * this );
    return mesh::mod_base_t::finalize( mesh );
}

void base_t::smooth_lvlchg
( mesh::mesh_t & mesh, const int & l, const dlvl_t & flag )
{
    auto & bdk = dynamic_cast
               < boundary::keeper_t & > ( * p_bdk );
    if( flag < 0 )              // Finer
    {
        int f = ( opt_flag_t::subtract_fld * need_sub_src )
              * ( i_iter == 0 && l == lvl_max );
        this->map_mesh( [ & ] ( auto & reg, auto & d )
        {
            f_refine ( d, l, opt_flag_t( f ) );
        }   );
        bdk.act_bnd_l( mesh, l, dlvl_t::finer );
    }
    const bool to_defect = l == 0 &&
        ( flag == dlvl_t::finer || flag == dlvl_t::same );
    for( int i_s = 0; i_s < n_smooth; ++ i_s )
    {
        bool defect( i_s == n_smooth - 1 && to_defect );
        int f = ( opt_flag_t:: subtract_src * to_sub_src )
              | ( opt_flag_t::output_defect *     defect );
            
        for( const auto s : { rb_t::red, rb_t::black } )
        {
            this->map_mesh( [ & ] ( auto & reg, auto & d )
            {
                f_smooth( d, l, s, opt_flag_t( f ) );
            }   );
            bdk.act_bnd_l( mesh, l, dlvl_t::same );
        }
        to_sub_src = false; // Don't sub_src afterwards
    }
    if( flag > 0 )              // Coarser
    {
        this->map_mesh( [ & ] ( auto & reg, auto & d )
        {
            f_coarse( d, l );
        }   );
        bdk. act_bnd_l( mesh, l - 1, dlvl_t::coarser );
        this->map_mesh( [ & ] ( auto & reg, auto & d )
        {
            f_residual( d, l - 1 );
        }   );
    }
    this->sync_streams( mesh );
    return;
}

int base_t::lvl_base( mesh::mesh_t & mesh )
{
    return proxy_default_t<  >::  dual
                 ( * begin(  )->second ).n_lvl(  );
}

int base_t::lvl_lim ( mesh::mesh_t & mesh )
{
    auto l = lvl_base( mesh ) - 2;
    if( cycle  % cycle_per_deep != 0 )
        l = std::max ( dl_scratch, l );
    return l;
}

////////////////////////////////////////////////////////////
// Interface

float_t base_t::v_cycle( mesh::mesh_t & mesh )
{
    lvl_min = -lvl_lim( mesh );
    for( int lvl = 0; lvl > lvl_min; -- lvl )
        smooth_lvlchg( mesh, lvl, dlvl_t ::coarser );
    smooth_lvlchg( mesh, lvl_min, dlvl_t ::   same );
    for( int lvl = lvl_min  +  1; lvl <= 0; ++ lvl )
        smooth_lvlchg( mesh, lvl, dlvl_t ::  finer );
    this->sync_streams(  mesh );
    return    f_defect( false );
};

float_t base_t::t_cycle( mesh::mesh_t & mesh )
{
    lvl_min = -lvl_lim( mesh );
    lvl_max = 0;
    
    for( int lvl = lvl_max;     lvl >  lvl_min; -- lvl )
        smooth_lvlchg( mesh,    lvl ,  dlvl_t::coarser );
    smooth_lvlchg( mesh,    lvl_min ,  dlvl_t::   same );
    for( int lvl = lvl_min + 1; lvl <= lvl_max; ++ lvl )
        smooth_lvlchg( mesh,    lvl ,  dlvl_t::  finer );
    for( const auto & d : ( * this ) )
        p_dev->sync_stream( d.stream );

    const auto def = f_defect( false );
    if( mesh.p_com->is_root(  ) )    
        std::cout << " rel. defect = " << def << std::endl;
    return def;
};

void base_t::step( mesh::mesh_t & mesh )
{
    this->sync_streams ( mesh );
    need_sub_src = mesh.tree(  ).periodic.prod(  ) ||
      p_opt->phy_bnd != operation::base_t::phy_bnd_t::fixed;
    
    if( cycle % cycle_calc_ave == 0 )
        f_calc_ave ( mesh ) ;
    
    auto it_max( cycle % cycle_per_deep == 0 ? n_iter : 1 );
    it_max =   ( cycle == 0 ? iter_first : it_max );

    to_sub_src = need_sub_src;
    float_t def( FLT_MAX );
    for( i_iter = 0;
         def > defect_lim && i_iter < it_max;   ++ i_iter )
         def = is_test ? t_cycle ( mesh ) : v_cycle( mesh );
    if( to_print && mesh.p_com->is_root(  ) )
        std::cout << "MG with "  << i_iter << " iters, "
                  << " defect = "<< def << std::endl;
    if( is_test )
        f_defect( true );
    ++ cycle;
    return this->sync_streams( mesh );
}
};                              // namespace multigrid
