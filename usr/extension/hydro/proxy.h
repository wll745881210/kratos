#pragma once

#include "../proxy/proxy_base.h"
#include "../radiation/radiation.h"
#include "../../../src/modules/chemistry/chem_hydro/hydro.h"

namespace extension::hydro
{
////////////////////////////////////////////////////////////
//

using type::    idx_t;
using type::  float_t;
using type::  coord_t;
using type:: float2_t;

template< class pol_T = pol_t >
struct prx_t : dynamics::hydro::proxy_t, prx_base_t
{
    struct bdt_hyd_t : ::chemistry::hydro::block_data_t
    {
        ////////// Data //////////
        int   id_l;
        struct
        {
            pol_T                        pool;
            mesh::dat_3d_t< float_t  >    flx;
            mesh::dat_3d_t< float_t  > mfps_i;
            mesh::dat_3d_t< float_t  > mfpa_i;
            mesh::dat_3d_t< float_t  > sp_col;
            mesh::dat_3d_t< float_t  > shield;

            __host__ __device__ int n_fld(  ) const
            {
                return flx.n_int;
            };
        }   rad;

        // Per-reaction rate storage (lazy, default off).
        mesh::dat_3d_t< float_t > reac_rate;

        // multigrid::data::base_t   mgd;
        // mesh::dat_3d_t< float_t > acc;

        ////////// Function //////////
        __host__ bdt_hyd_t(  ) {  };
        
        template   < class bdt_T > __host__
        bdt_hyd_t( const bdt_T   & s )
        {
            // Shallow-copying the hydrodynamic part
            auto & self = static_cast
            < chemistry::hydro::block_data_t & > ( * this );
            auto & src  = static_cast
            < const chemistry::hydro::block_data_t & >( s );
            self        = src;

            // Shallow-copy the data related
            // this->mgd  = d_mgd( s.reg ).mgd;
            auto & b_d_par = d_par( s.reg );
            rad.flx    = b_d_par.rad.   flx;
            rad.mfps_i = b_d_par.rad.mfps_i;
            rad.mfpa_i = b_d_par.rad.mfpa_i;
            rad.sp_col = b_d_par.rad.sp_col;
            rad.shield = b_d_par.rad.shield;
            reac_rate = static_cast
                < const bdt_hyd_t & >( s ).reac_rate;

            // Shallow-copy of the photon pool
            auto & par = dynamic_cast
            < radiation_t & >( * p_par.lock( ) );
            rad.pool
                = dynamic_cast< pol_T & > ( * par.p_pol );
            auto & mesh( * p_par.lock(  )->p_mesh );
            id_l = mesh.block( s.reg ).id_l;
            return;
        };
    };

    using bdt_t = bdt_hyd_t;
    
    template     < class bdt_T > __host__
    static auto d( const bdt_T & bdt )
    {
        return bdt_hyd_t( bdt );
    };

    static __host__
    void set_module( mesh::mod_base_t & mod )
    {
        p_hyd = mod.weak_from_this(  );
        return;
    };
};

};                         // namespace extension::hydro
