#pragma once

#include "../../../src/modules/particle/radiation/radiation.h"
#include "../../../src/modules/chemistry/chem_hydro/hydro.h"
#include "block/block_data.h"
#include "particle/photon.h"
#include "block/proxy.h"
#include "particle/particle_pool.h"
#include "comm/comm.h"
#include "integrate/integrator.h"

namespace extension
{
////////////////////////////////////////////////////////////
//
class hydro_chem_t;

class chem_t;

using pol_t  = radiation::pool_t< radiation::photon_t<  > >;

class radiation_t : public particle::radiation::base_t
{
protected:                      // Data
    bool                         is_first_run;
protected:                      // Functions
    virtual void init_cond
    ( mesh::block::dual_t & d ) override;
public:                         // Data
    int                           n_sp; // species
    int                           n_sh; // shielding
    std::vector< type::float_t > hv_eV;
    std::shared_ptr< chem_t >    p_che;
    std::shared_ptr< ::chemistry::hydro::base_t >  p_hyd;

    std::function< void( mesh::mesh_t & ) > f_reset;
public:
    virtual void    read( const  input & args ) override;
    virtual void    init
    ( const input & args, mesh::mesh_t & mesh ) override;
    virtual void    save( mesh::mesh_t & mesh,
                          binary_io::base_t & ) override;
    virtual void    step( mesh::mesh_t & mesh ) override;
    // virtual void   reset( mesh::mesh_t & mesh ) ;        

    template < class prx_T, class pol_T ,
               class gen_T, class itg_T > 
    void enroll_extension(  )
    {
        enroll_dat< typename prx_T::bdt_t > (    );
        block_yield = [ & ]
        ( mesh::block_t & b, mesh::mesh_t & mesh )
        {
            auto p = block_yield_base ( b , mesh );
            auto & b_h   =  prx_T::ref( p->h(  ) );
            auto & b_d   =  prx_T::ref( p->d(  ) );
            b_h.rad.n_fld(  ) = n_hv;
            b_d.rad.n_fld(  ) = n_hv;
            b_h.rad.n_spe(  ) = n_sp;
            b_d.rad.n_spe(  ) = n_sp;
            b_h.rad.n_shl(  ) = n_sh;
            b_d.rad.n_shl(  ) = n_sh;
            b_h.rad.order(  ) = space_order(  );
            b_d.rad.order(  ) = space_order(  );
            return p ;
        };
        f_reset = [ & ] ( mesh::mesh_t & mesh )
        {
            for( auto & d : ( * this ) )
            {
                d.d(  ).dt = mesh. p_cyc ->dt(  );
                const auto & rad = prx_T::ref( d.d( ) ).rad;
                p_dev->f_mset( rad.   flx.   dat, 0,
                               rad.   flx.n_size, stream );
                p_dev->f_mset( rad.sp_col.   dat, 0,
                               rad.sp_col.n_size, stream );
                //  The  VET  moment  accumulator  (  § 3.19;
                // zero  unless  accumulated  --  the  FLD  /
                // chemistry  legs  see  a  zero  field ):
                p_dev->f_mset( rad.  chi_m.   dat, 0,
                               rad.  chi_m.n_size, stream );
            }
        };
            
        using map_t = particle ::block_map_t< prx_T >;
        using com_t = radiation::     comm_t< pol_T >;
        return particle::base_t::enroll
        < prx_T, pol_T, gen_T, itg_T, map_t, com_t > (  );
    };
};

};                          // namespace extension
