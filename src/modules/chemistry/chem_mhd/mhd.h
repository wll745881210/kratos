#pragma once

#include "../../dynamics/mhd/mhd.h"
#include "../../dynamics/mhd/types/block_data.h"
#include "../../dynamics/mhd/boundary/comm.h"
#include "reconstruct.h"
#include "riemann.h"

namespace chemistry
{
class base_t;
};

namespace chemistry::mhd
{
//////////////////////////////////////////////////////////
// MHD with chemistry enabled

struct block_data_t : dynamics::mhd::block_data_t
{
    ////////// Type and data //////////
    using super_t = dynamics::mhd::block_data_t;
    mesh::dat_3d_t< type::float_t > T_hyd;

    ////////// Functions //////////
    __host__ virtual void setup
    ( const mesh::f_new_t & f_new )
    {
        T_hyd.init( f_new, geo.n_ceff, 0, 1, 1 );
        return super_t::setup( f_new );
    };

    __host__ virtual void write
    ( const mesh::f_write_t & f_w )
    {
        T_hyd.write( f_w, "hydro_T_" );
        return   super_t::write( f_w );
    };

    __host__ virtual void free
    ( const mesh::f_free_t  & f_free )
    {
        T_hyd.free( f_free );
        return super_t::free( f_free );
    };

    __host__ virtual void copy_input
    ( const  mesh::     f_cp_t & f_cp ,
      mesh::block::base_data_t &  tgt )
    {
        T_hyd.cp_to( static_cast
                   < block_data_t & > ( tgt ).T_hyd, f_cp );
        return super_t::copy_input( f_cp, tgt );
    };
};

struct proxy_t : mesh::block::proxy_t< block_data_t >
{
    using bdt_t = block_data_t;
};

class base_t : public dynamics::mhd_t
{
    ////////// Chemistry //////////
protected:                      // Data
    type::float2_t                            T0;
    type::float2_t                    mu_cgs_ave;
    std ::vector  < type::     float2_t >  x_spe;
public:                         // Data
    std ::weak_ptr< ::chemistry::base_t >  q_che;
protected:                      // Functions
    virtual void read_chem( const input &  args );

    ////////// Interfaces //////////
public:                         // Functions
    virtual void read  ( const input & );
    virtual void init  ( const input &, mesh::mesh_t & );
    virtual void init_cond   ( mesh :: block::dual_t & );
    virtual int  intra_order   (  );
    virtual void enroll_default(  );
    virtual void parasite
    ( std::weak_ptr< mesh::mod_base_t > );

    template
    < class prx_T,
      class eos_T,
      class itg_T = dynamics::mhd::integrator::rk2_t<  > >
    void enroll_chem_mhd(  )
    {
        enroll_dat< typename prx_T::bdt_t > (  );
        return dynamics::mhd_t::enroll< prx_T, itg_T,
            eos_T, rec_t, rie_t<  >,
            dynamics::mhd::boundary::comm_t< prx_T > > (  );
    };
};

};                         // namespace chemistry::mhd
