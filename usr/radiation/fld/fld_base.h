#pragma once

#include "../../../src/modules/multigrid/multigrid.h"
#include "../../../src/mesh/types/dat_3d.h"

namespace rad::fld
{

////////////////////////////////////////////////////////////
//  FLD multigrid driver base:  multigrid::base_t plus the
// knobs the FLD driver needs  ( the mirror of the nonideal
// ni_mg_base_t;  the core multigrid module stays the
// pristine Poisson driver ).
//    sub_src_off --  the FLD operator  ( I - dt D lap +
// sig )  is SPD  ( identity - shifted ),  so the mean -
// subtraction projection must stay OFF.
//    print_iter / stag --  the defect trace / stagnation
// break of the core  ( the  [multigrid]  par names ).

struct fld_mg_base_t : multigrid::base_t
{
    ////////// Data //////////
    bool                     print_iter;
    bool                     sub_src_off;
    type::float_t                  stag;

    ////////// Functions //////////
    fld_mg_base_t(  )
    {
        print_iter  = false;
        sub_src_off = true;
        stag        = type::float_t( 1.5 );
        return;
    };

    virtual void read( const input & args ) override
    {
        multigrid::base_t::read( args );
        print_iter = args.get< bool >
                   ( "multigrid", "print_iter",  false );
        stag       = args.get< type::float_t >
                   ( "multigrid", "stag", 1.5 );
        return;
    };

    virtual void step( mesh::mesh_t & mesh ) override
    {
        this->sync_streams ( mesh );
        need_sub_src = ( mesh.tree(  ).periodic.prod(  ) ||
          p_opt->phy_bnd
          != multigrid::operation::base_t::phy_bnd_t::fixed )
          && ! sub_src_off;

        if( cycle % cycle_calc_ave == 0 )
            f_calc_ave ( mesh ) ;

        auto it_max( cycle % cycle_per_deep == 0
                     ? n_iter : 1 );
        it_max =   ( cycle == 0 ? iter_first : it_max );

        to_sub_src = need_sub_src;
        type::float_t def( FLT_MAX ), def_prev( FLT_MAX );
        for( i_iter = 0;
             def > defect_lim && i_iter < it_max;   ++ i_iter )
        {
             def = is_test ? t_cycle ( mesh )
                           : v_cycle ( mesh );
             if( print_iter && to_print
                 && mesh.p_com->is_root(  ) )
                 std::cout << "FLD MG iter " << i_iter
                           << " :  defect = " << def
                           << std::endl;
             if( stag > 0 && i_iter > 0
                 && def >= stag * def_prev )
                 break;
             def_prev = def;
        }
        if( to_print && mesh.p_com->is_root(  ) )
            std::cout << "FLD MG with "  << i_iter
                      << " iters,  defect = " << def
                      << std::endl;
        ++ cycle;
        return this->sync_streams( mesh );
    };
};

};                          // namespace rad::fld
