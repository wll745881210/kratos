#pragma once

#include "../../usr/extension/chem_therm/reaction/reaction_list.h"
#include "disk_dust.h"
#include "disk_ionize.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  The PPD reaction list:  the extension ( chem_therm )
//  list,  with
//  ( a ) the dust temperature from the disk_dust_t provider
//      instead of the radiation flux ( prep_dev override;
//      prep_dev is non - virtual but the solver dispatches
//      it on the STATIC list type );
//  ( b ) the analytic diffuse ionization ( App A.1,
//      disk_ionize.h ):  the branch reactions in the extra
//      ion_reac list,  appended to d.f / d.J AFTER the base
//      rate() / jac() reduced f_cache ( the base CRTP self
//      is pinned to the extension list type,  so extending
//      the rate_single dispatch ranges would NOT dispatch --
//      hence the post - commit accumulation ).
//
//  THE UPGRADE POINTS:  T_dust -> self - consistent dust
//  RT in disk_dust_t::eval;  xi_eff -> runtime particle /
//  Monte - Carlo ionization in disk_ionize_t::eval_xi.
//
//  ***  The dat object d is PER - THREAD ( its scalar
//  members live in registers,  NOT shared memory ):
//  d.T_dust must be written by ALL threads redundantly.
//  A single - thread write + __syncthreads leaves the other
//  threads' copies at the base value:  the dust reaction sees
//  T_dust = T_eq = 1 and cools the disk to the floor.
//  ***
//
//  The base init_phot_reac leaves n_hv / hv_erg
//  UNINITIALIZED when no photo_reac_file is given ( the
//  planet pgens always have one --  a latent bug there );
//  the constructor pins them to the empty state.

struct ppd_list_t : extension::chemistry::list_t
{
    ////////// Types //////////
    using super_t = extension::chemistry::list_t;

    ////////// Data //////////
    disk_dust_t                      dust;
    disk_ionize_t                     ion;
    utils::const_list_t< ion_diff_t > ion_reac;
    type::float_t                T_cap_lo;
    type::float_t                T_cap_hi;

    ////////// Functions //////////
    __host__ ppd_list_t(  )
    {
        n_hv   =       0;
        hv_erg = nullptr;
        return;
    };

    __host__ virtual void init
    ( const input & args, ::chemistry::base_t & chem )
    override
    {
        //  init_extra BEFORE the base init:  the core
        //  std_t::init returns early when no reaction_file
        //  is given,  skipping init_extra ( n_cool / i_elec
        //  / T_eq / fac_dust_stiff would stay garbage ).   And
        //  init_extra is idempotent for the scalar settings but
        //  n_cool is a COUNTER incremented by init_mol /
        //  atom_cool later in the base init --  so it must run
        //  FIRST:  re - running it after the base init wipes
        //  n_cool and leaves the cooling interpolators
        //  uninitialized while dispatched.
        init_extra( args, chem );
        super_t::init( args, chem );
        dust.read( args );
        ion .read( args );
        //  WBG19 regular_temp:  clamp T to [ T_cap_lo,
        //  T_cap_hi ]  at chemistry entry ( prep_dev,
        //  before the solve ).
        T_cap_lo = args.get< type::float_t >
                   ( "chemistry", "T_cap_lo", 10   );
        T_cap_hi = args.get< type::float_t >
                   ( "chemistry", "T_cap_hi", 5e5  );
        if( ion.on )
        {
            ion.setup( chem );
            //  The diffuse - ionization branches are read
            //  from [chemistry] ion_reac_file ( the same
            //  section format as data / chemistry / *.dat,
            //  rate_type = ION_DIFF,  parameters[ 0 ] = the
            //  branch factor ).   Without a file,  fall
            //  back to the three WBG19 branches (
            //  mhd_wind.cpp:776-830 ).
            std::vector< ion_diff_t > ion_h;
            auto add_branch = [ & ]
            ( const std::string & reac,
              const std::vector< std::string > & prods,
              const type::float2_t & branch )
            {
                if( ! chem.species.exist( reac ) )
                    return;
                std::vector< int > i_c;
                i_c.push_back( chem.species.at( reac ) );
                bool ok( true );
                for( const auto & p : prods )
                    if( chem.species.exist( p ) )
                        i_c.push_back( chem.species.at( p ) );
                    else
                        ok = false;
                if( ! ok )
                    return;
                std::vector< type::float_t > pars( 1, 1 );
                std::vector< type::float_t > enes;
                std::vector< type::float_t > f_p
                    ( prods.size(  ), 1 );
                auto & r = ion_h.emplace_back(  );
                r.setup_pars( chem, i_c, 1, pars, enes, f_p );
                r.branch = branch;
                return;
            };
            input iarg( args.get< std::string >
                ( "chemistry", "ion_reac_file", "" ) );
            iarg.set_comm( chem.p_mesh->p_com );
            if( iarg.read(  ) )
            {
                for( const auto & r : iarg.get_prefixes(  ) )
                {
                    const auto rt = iarg.get< std::string >
                        ( r, "rate_type", "" );
                    if( rt.find( "ION_DIFF" )
                        == std::string::npos )
                        continue;
                    std::vector< std::string > reacs, prods;
                    std::vector< type::float2_t > pars;
                    iarg( r, "reactants",  reacs );
                    iarg( r, "products",   prods );
                    iarg( r, "parameters", pars  );
                    if( reacs.size(  ) != 1 || pars.empty(  ) )
                        throw std::runtime_error
                            ( "ion_reac_file:  ION_DIFF "
                              "needs 1 reactant + branch" );
                    add_branch( reacs[ 0 ], prods,
                                pars[ 0 ] );
                }
            }
            else
            {
                if( chem.p_mesh->p_com->is_root(  ) )
                    std::cout << "ion_reac_file absent:  "
                        "using the three WBG19 default "
                        "branches" << std::endl;
                add_branch( "H2", { "H", "H+", "e-" },
                            1.0  );
                add_branch( "H",  { "H+", "e-" },
                            0.5  );
                add_branch( "He", { "He+", "e-" },
                            0.84 );
            }
            ion_reac.init( ion_h.size(  ) );
            for( size_t i = 0; i < ion_h.size(  ); ++ i )
                ion_reac[ i ] = ion_h.at( i );
            const auto on_const = args.get< bool >
                ( "chemistry", "reac_on_const",  true );
            ion_reac.regularize( * chem.p_dev, on_const );
            if( chem.p_mesh->p_com->is_root(  ) )
                std::cout << "=======================\n"
                    << "# of diffuse ionization branches: "
                    << ion_reac.size
                    << "\n======================="
                    << std::endl;
        }
        return;
    };

    template< class dat_T, class bdt_T, class eos_T >
    __device__ __forceinline__ void prep_dev
    ( dat_T       &   d , const bdt_T & b_d,
      const eos_T & eos ) const
    {
        super_t::prep_dev( d, b_d, eos );
        //  T_dust from the provider,  NOT from the
        //  radiation flux.   d is PER - THREAD ( registers,
        //  not shared memory ):  EVERY thread evaluates --
        //  a single - thread write + __syncthreads would
        //  leave the other threads at the base value.
        d.T_dust = dust.eval( b_d, d.idx );
        //  The T clamp ( WBG19 regular_temp ):  d.y is in
        //  SHARED memory --  single - thread write + sync;
        //  d.T is per - thread --  every thread sets it.
        if( d.T > T_cap_hi || d.T < T_cap_lo )
        {
            const auto T_new = d.T > T_cap_hi ? T_cap_hi
                                              : T_cap_lo;
            if( threadIdx.x == 0 )
                d.y[ d.i_en ] *= T_new / d.T;
            d.T = T_new;
            __syncthreads(  );
        }
        return;
    };

    //  rate / jac:  the base commits f_cache into d.f /
    //  d.J;  the ionization branches then accumulate
    //  directly ( see the header comment ).
    template< class dat_T, class bdt_T, class eos_T >
    __device__ void rate
    ( dat_T       &   d , const bdt_T & bdt,
      const eos_T & eos ) const
    {
        super_t::rate( d, bdt, eos );
        if( ion.on )
        {
            for( int i = threadIdx.x; i < ion_reac.size;
                 i += blockDim.x )
                ion_reac[ i ].rate_add( d, bdt, eos,
                                        * this );
            __syncthreads(  );
        }
        return;
    };

    template< class dat_T, class bdt_T, class eos_T >
    __device__ void jac
    ( dat_T       &   d , const bdt_T & bdt,
      const eos_T & eos ) const
    {
        super_t::jac( d, bdt, eos );
        if( ion.on )
        {
            for( int i = threadIdx.x; i < ion_reac.size;
                 i += blockDim.x )
                ion_reac[ i ].jac_add( d, bdt, eos,
                                       * this );
            __syncthreads(  );
        }
        return;
    };
};

};                              // namespace interm
