#pragma once

#include "../../mesh/mesh.h"
#include "../../mesh/block/block.h"
#include "boundary/keeper.h"
#include "boundary/comm.h"
#include "operation/operation.h"
#include "types/block_data.h"
#include "types/dual_data.h"
#include <memory>

namespace multigrid
{
////////////////////////////////////////////////////////////
// Types for block access

template < class bdt_T = data::block_data_t >
struct proxy_default_t : mesh::block::proxy_t< bdt_T >
{
    static data::dual_t & dual( mesh::block::dual_t & d )
    {
        return dynamic_cast< data::dual_t & > ( d );
    };
};

////////////////////////////////////////////////////////////
// Default and basic multi-grid driver

using namespace mesh;
using type:: float_t;

class base_t : public mesh::mod_base_t
{
    ////////// Initializer //////////
public:
    base_t(  ) : n_fld( 1 ) {  };
   ~base_t(  )              {  };
    virtual void read    ( const input  & );
    virtual void finalize( mesh::mesh_t & );

    ////////// Operations //////////
protected:                      // Data
    int                                   n_th;
    int                                  n_fld;
    int                                  cycle;
    int                                 i_iter;    
    int                                 n_iter;
    int                                lvl_min;
    int                                lvl_max;
    int                               n_smooth;
    int                             dl_scratch;
    int                             iter_first;
    int                         cycle_calc_ave;
    int                         cycle_per_deep;
    bool                          need_sub_src;
    bool                            to_sub_src;
    bool                               is_test;
    bool                              to_print;    
    float_t                         defect_lim;
    std::shared_ptr< operation::base_t > p_opt;
protected:                      // Modules
    using f_act_t = std::function
    < void( block::dual_t &,  const int & ) >;
    std::function
    < void ( block::dual_t &,     const  int & ,
             const  rb_t &, const opt_flag_t & ) > f_smooth;
    std::function< float_t( const bool   & ) >     f_defect;
    std::function< void ( block::dual_t &, const int &,
                          const opt_flag_t & ) >   f_refine;
    f_act_t                                        f_coarse;
    f_act_t                                      f_residual;
    std::function< void   ( mesh::mesh_t & ) >   f_calc_ave;

protected:                      // Function
    virtual void smooth_lvlchg
     ( mesh::mesh_t &, const int &, const dlvl_t & );
    virtual int lvl_lim ( mesh_t & );
    virtual int lvl_base( mesh_t & );
public:                         // Function
    virtual float_t v_cycle    ( mesh::mesh_t & );
    virtual float_t t_cycle    ( mesh::mesh_t & );
    virtual void    step       ( mesh::mesh_t & );
    virtual int     space_order(    )    { return  1; };

    ////////// Type interfaces //////////
public:                         // Functions
    template< class prx_T =        proxy_default_t<  > ,
              class opt_T = operation::operation_t<  > ,
              class com_T = boundary::comm_t < prx_T > >
    void enroll (   )
    {
        p_opt = std::make_shared< opt_T   > (         );
        auto & opt = static_cast< opt_T & > ( * p_opt );
        f_refine = [ & ]( block::dual_t & b, const int & l,
                          const opt_flag_t & f )        
        {
            opt.flag = f;
            auto & d  ( prx_T::dual( b ) );
            const auto & b_f( prx_T::d( d.d_mg( l     ) ) );
            const auto & b_c( prx_T::d( d.d_mg( l - 1 ) ) );
            p_dev->launch( kernel::refine, opt.rsc_refine
                   ( b_c, n_th ), d.stream, b_c, b_f, opt );
            p_dev->event_record  ( d.event, d.stream );
        };
        f_coarse = [ & ]( block::dual_t & b, const int & l )
        {
            auto & d  ( prx_T::dual( b ) );
            const auto & b_f( prx_T::d( d.d_mg( l     ) ) );
            const auto & b_c( prx_T::d( d.d_mg( l - 1 ) ) );
            p_dev->launch( kernel::coarse,   opt.rsc_coarse
                   ( b_c, n_th ), d.stream, b_c, b_f, opt );
            p_dev->event_record  ( d.event, d.stream );
        };
        f_residual = [ & ]
        ( block::dual_t & b, const int & l )
        {
            auto & d  ( prx_T::dual( b ) );
            const auto & b_c( prx_T::d( d.d_mg( l ) ) );
            p_dev->launch( kernel::residual, opt.rsc_coarse
                   ( b_c, n_th ), d.stream, b_c, opt );
            p_dev->event_record ( d. event, d.stream );
        } ;       
        f_smooth = [ & ]( block::dual_t & b, const int & l,
                      const rb_t & s, const opt_flag_t & f )
        {
            opt.flag  = f;
            auto & d  ( prx_T::dual( b ) );
            const auto & bdt( prx_T::d ( d.d_mg( l ) ) );
            p_dev->launch( kernel::smooth, opt.rsc_smooth
                    ( bdt, n_th ), d.stream, bdt, s, opt );
            p_dev->event_record  ( d.event,  d.   stream );
        };
        f_calc_ave = [ & ] ( mesh::mesh_t & mesh )
        {
            this->map_mesh( [ & ] ( auto & reg, auto & d )
            {
                const auto & bdt( prx_T::d( d.d(  ) ) );
                const auto & n_ceff  ( bdt.geo.n_ceff );
                const auto bl = dim3( n_ceff[ 2 ] );
                const auto th = dim3( n_ceff[ 1 ] );
                auto sh = th.x * ( 1 + 2 *  opt.n_fld )
                        * type::size ( opt. src_ave_d );
                auto lp = std::make_tuple( bl, th, sh );
                p_dev->launch( kernel::calc_ave, lp,
                               d.stream, bdt, opt );
            }   );
            opt.stat_src_ave( * this );
        };
        f_defect = [ & ] ( const bool & output )
        {
            if( output )
                map_mesh( [ & ] ( auto & reg, auto & d )
            {
                const auto & bdt( prx_T::d( d.d(  ) ) );
                const auto & n_ceff  ( bdt.geo.n_ceff );
                auto bl = dim3( n_ceff[ 1 ] , n_ceff[ 2 ] );
                auto th = dim3( n_ceff[ 0 ] ); 
                auto sh = th.x * opt.n_sh_smooth( bdt )
                        * type::size( opt.defect_d );
                auto lp = std::make_tuple( bl, th, sh );
                p_dev->launch( kernel::defect, lp,
                               d.stream, bdt, opt );
            }   );
            return opt.stat_defect( * this );
        };

        if( ! f_bdt_yield )
            enroll_dat< typename prx_T:: bdt_t ,
                                 data ::dual_t > (  );
        block_yield = [ & ]
        ( mesh::block_t & b, mesh::mesh_t & mesh )
        {
            auto p = block_yield_base( b  , mesh );
            auto &   d = prx_T::dual ( * p      );
            auto & b_h = prx_T:: ref ( d.h(   ) );
            auto & b_d = prx_T:: ref ( d.d(   ) );
            b_d.mgd.order(  ) = space_order  (  );
            b_h.mgd.order(  ) = space_order  (  );
            b_d.mgd.n_fld(  ) = n_fld;
            b_h.mgd.n_fld(  ) = n_fld;
            opt.    n_fld     = n_fld;
            opt.setup_mem  ( * this );
            return p ;
        };
        if( ! p_bdk )
            enroll_bdk < boundary::keeper_t > (  )
                      -> enroll_comm< com_T > (  );

        return prx_T::set_module( * this );
    };
};                              // class base_t
};                              // namespace multigrid
