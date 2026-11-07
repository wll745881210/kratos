#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <future>

#include "../../types.h"
#include "../../device/device.h"
#include "../../io/binary/binary_io.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Types

using type:: float_t;
using type::float2_t;

class mesh_t;

namespace cycle
{
////////////////////////////////////////////////////////////
// Evolution cycle counter of the mesh

struct base_t
{
    ////////// Types //////////
protected:                      // Types
    using clock = std::chrono::high_resolution_clock;
    using clk_t = decltype( clock::now(  ) );

    ////////// Con-/destructor and initializer //////////
public:                         // Functions
    base_t(  );
    virtual void init( const input & args, mesh_t & mesh );
    virtual void finalize(  );

    ////////// Evolution control //////////
public:                         // Data
    bool         dt_reduced;    // Avoid re-calling redc_dt
    int         n_cycle_lim;
    int             i_cycle;
    float2_t          t_lim;
    float2_t              t;
    //  The dt of the CURRENT step -- what  b_d.dt_0  is set from,  and what
    //  the print / fuse read.  It is the step ACTUALLY taken:  with
    //  [ cycle ] t_lim_top_up = 1  the last step is shortened to land
    //  exactly on t_lim,  so this holds the shortened value.  The
    //  sub-step factor is applied on top of it by the dynamics module:
    //      b_d.dt = b_d.dt_0 * itg.step_size( stp ).
    float_t           dt_cur;
    float_t       *    p_dt;
    //  [ 0 ] the last CFL dt,  UNCLAMPED  ( the checkpoint / forensics
    //  value --  a restart must not inherit an artificial top-up dt );
    //  [ 1 ] the next step's device upper bound  ( the step taken
    //  x dt_expand ).
    float_t       *  p_dt_h;
    float_t       dt_expand;
    //  [ cycle ] t_lim_top_up  ( default false ):  false = do NOT shorten the
    //  last step to land exactly on t_lim  ( t may overshoot by at most one
    //  dt );  true = top up to t_lim exactly  ( cosmology / external
    //  timelines ).   docs/analysis/cycle-top-up-design.md.
    bool             top_up;
    device::stream_t stream;
protected:                      // Module
    std::shared_ptr< device::base_t > p_dev;
protected:                      // Functions
    virtual bool step          ( mesh_t & mesh );
public:                         // Interface
    virtual void evolve        ( mesh_t & mesh );
    virtual void redc_dt_start ( mesh_t & mesh );
    virtual void redc_dt_finish( mesh_t & mesh );    
    virtual float_t   dt(  ) const;

    ////////// Signals //////////
protected:
    static  void sighdl( int    signum );

    ////////// Speed tests //////////
protected:                      // Data
    clk_t       wt_prev;
    int       i_cycle_0;
    float2_t n_cell_tot;
protected:
    template< class c_T >
    float_t dt_wt( const c_T & dt );
    float_t dt_wt(                );

    ////////// Info output //////////
protected:                      // Data
    int n_display_freq;
    int t_display_prec;
    std::string prefix;
protected:                      // Functions
    virtual bool update_tout
    ( float2_t & tout, const float2_t &   dt );
    virtual void print_info  ( mesh_t & mesh );

    ////////// Binary IO //////////
protected:                      // Data
    bool  final_output;
    float2_t dt_output;
    float2_t  t_output;
    int       i_output;
public:                         // Functions
    virtual std::string file_name( mesh_t & mesh );
    virtual void load ( binary_io::base_t &  bio );
    virtual void save ( binary_io::base_t &  bio );
};

};                              // namespace cycle
};                              // namespace mesh
