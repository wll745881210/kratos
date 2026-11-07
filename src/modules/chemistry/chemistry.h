#pragma once

#include "../../utilities/string.h"
#include "chem_hydro/hydro.h"
#include "solver/solver.h"
#include "reaction/reaction_list_base.h"
#include "utilities/periodic.h"
#include <map>
#include <set>

namespace chemistry
{
////////////////////////////////////////////////////////////
// Types

namespace reaction::list
{
template< class     rec_T = reaction::  std_t<  >,
          class derived_T = crtp    ::dummy_t   >
struct    std_t;
};

////////////////////////////////////////////////////////////
// Chemistry module

class base_t : public mesh::mod_base_t
{
    ////////// Initializer //////////
protected:                      // Data and Types
    bool               initialized;
    using mesh::mod_base_t::update;
public:                         // Functions
    base_t(  ) : mod_base_t(  ),  initialized( false ) {  };
    virtual void read  ( const input & );
    virtual void init  ( const input &, mesh::mesh_t & );
    virtual void step                 ( mesh::mesh_t & );
    virtual void update               ( mesh::mesh_t & );
    virtual void finalize             ( mesh::mesh_t & );
    virtual void parasite( std::weak_ptr< mod_base_t > );

    ////////// Chemistry Data //////////
public:                        // Data
    sp_vec_t     species;
    elem_dat_t  elem_dat;
protected:                      // Function
    virtual void parse_species( const input & args );
    virtual void parse_species_single
    ( const input & args, const std::string &   sp );

    ////////// Chemical reactions //////////
public:                         // Module
    std::shared_ptr< solver  ::      base_t > p_sol;
    std::shared_ptr< dynamics::eos ::base_t > p_eos;
    std::shared_ptr< reaction::list::base_t > p_rls;

    ////////// Interface //////////
protected:                      // Module
    std::function< void( mesh::block::dual_t & ) >   intg;
    std::function< void( const input  &    ) > setup_mods;
public:                         // Function
    template< class prx_T = chemistry::hydro::proxy_t     ,
              class rls_T = reaction ::list ::  std_t<  > ,
              class eos_T = hydro    ::         eos_t<  > ,
              class sol_T = solver   ::     default_t<  > >
    void enroll(  )
    {
        p_rls  = std::make_shared< rls_T   > (         );
        p_sol  = std::make_shared< sol_T   > (         );
        intg = [ & ] ( mesh::block::dual_t & d )
        {
            auto & eos = dynamic_cast< eos_T & >( * p_eos );
            auto & rls = dynamic_cast< rls_T & >( * p_rls );
            auto & sol = dynamic_cast< sol_T & >( * p_sol );
            sol.template integrate
                         < prx_T > ( * this, d, rls,  eos );
        };
        setup_mods = [ & ] ( const input & args )
        {
            auto & eos = dynamic_cast< eos_T & >( * p_eos );
            eos.set_chem( args, species, * this->p_dev );
        };
        return prx_T::set_module( * this );
    };
};

};                              // namespace chemistry
