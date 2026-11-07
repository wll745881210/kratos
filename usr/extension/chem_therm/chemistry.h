#pragma once

#include <map>
#include <regex>
#include <vector>
#include "../../../src/modules/chemistry/chemistry.h"

namespace extension
{
////////////////////////////////////////////////////////////
// Chemistry with details

template< class bdt_T, class eos_T, class rls_T >
__global__ void ker_prep_rad
( const bdt_T bdt, const eos_T eos, const rls_T rls )
{
    return rls.prep_rad( bdt, eos );
};

class radiation_t;

struct dust_pop_t
{
    type::float_t   r_cgs;  // grain radius      [cm]
    type::float_t rho_cgs;  // solid density [g/cm^3]
    // Per-population dust grain parameters.  Each
    // population is keyed by its base element name
    // (e.g. "C", "Si", "Fe").
    // Input syntax: dust_pop_C = 1e-5 3.0 (radius [cm],
    // density [g/cm^3])
    // Species without a specific pop entry (incl. "Gr") use
    // global defaults.
};

class chem_t : public ::chemistry::base_t
{
    ////////// Dust grains //////////
public:                         // Data
    // Dust grain parameters; defaults first, then dict 
    type::float_t                      r_dust_cgs; 
    type::float_t                    rho_dust_cgs; 
    std::map< std::string, dust_pop_t > dust_pops;
public:                         // Functions
    bool is_grain_species( const std::string & name ) const;
    type::float_t get_sigma_dust_cgs
    ( const std::string & name ) const;
    dust_pop_t get_dust_params
    ( const std::string & name ) const;

    ////////// Generic //////////
public:                         // Data
    int                                  n_shield;
    type::float_t                      h_cool_cgs;
    type::float_t                      mu_ave_cgs;
    bool                      output_reac_rates;
    std::shared_ptr< radiation_t >          p_rad;
protected:                      // Functions
    virtual void parse_species_single
    ( const input & args, const std::string & sp ) override;
    void parse_species_gr
    ( const input & args, const std::string & sp_str );
public:
    virtual void prep_rad( mesh::mesh_t & mesh );
    virtual void compute_rates( mesh::mesh_t & mesh, binary_io::base_t & bio ) { };  // pgen overrides with correct eos/prx types
    virtual void free_reac_rates( mesh::mesh_t & ) { };  // pgen overrides to free reac_rate
    virtual void read    ( const  input & args )  override;
    virtual void step    ( mesh::mesh_t & mesh )  override;
    virtual void init
    ( const input &, mesh::mesh_t & ) override;
    virtual void save
    ( mesh::mesh_t & mesh, binary_io::base_t & bio ) override;
    virtual void finalize( mesh::mesh_t & mesh ) override;
};                              // clas chem_t
};                              // namespace extension
