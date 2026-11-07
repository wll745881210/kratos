#pragma once

#include "edge_mg_data.h"
#include "edge_mg_device.h"

////////////////////////////////////////////////////////////
//  Interface of the edge-centred implicit diffusion solver.
//
//  This header is the ONLY thing a pgen ( or the module's
//  ni_t ) may include:  declarations and plain data,  no
//  device code,  no templates ( rule C7 of docs/mhd/
//  nonideal/edge-mg-foundations.md section 3.3 ).  All the
//  kernels and the driver live in edge_mg.cpp.

class input;

namespace device
{
class base_t;
}

namespace mhd_ee::ni::edge_mg
{
using type::float_t;


////////////////////////////////////////////////////////////
//  Knobs.  All under  [nonideal]  with the  ee_  prefix.
//  Every one of them defaults to the OFF value, except
//  ee_method:  it defaults to 5 ( the device matrix-free
//  backend,  the shipped one ),  with a loud fallback to 2
//  for the two knobs that backend cannot carry
//  ( ee_refine,  ee_a_bridge ).  See par_t::read().

struct par_t
{
    ////////// Data //////////
    int       enable;    // ee_enable   ( the whole leg )
    int        solve;    // ee_solve    0 = the S0' diagnostic ( bit-identical ),
                         //  1 = assemble + solve + write
                         //  back
    int      subcell;    // ee_subcell  0 = one tensor a cell,  1 = + Taylor
    int      ee_coarsen; // cdir bits: 1 = r, 2 = theta, 4 = phi
    int      ee_verb;    // print the diagnostic every cycle
    int      ee_iter;    // iteration ( or cycle ) cap
    int     ee_nlevel;   // MG levels
    int    ee_n_smooth;  // smoothing sweeps per level
    //  bc_edge: 0 legacy / 1 n x E = 0 on wall-tangent
    //  edges / 2 n x B = 0 ( Dirichlet edge elimination ).
    int      bc_edge;
    //  ee_writeback: 0 = strong -dt d1( eta J ) increment;
    //  1 = b + P_out( a_new ) - P_out( a_old ),  cached
    //  between steps ( one representation solve per step ).
    int  ee_writeback;
    //  ee_dt:  0 = the cycle dt ( the parabolic cap still
    //  applies );  > 0 = force this dt and relax the cap.
    //  For the STS / implicit A/B only.
    float_t   ee_dt;
    //  ee_a_bridge:  A7.1 representation IN,  M1 a = M12 b,
    //  when no edge state is available ( first step /
    //  restart ).  bf is converted from B.n to a flux with
    //  the CT face areas.
    int  ee_a_bridge;
    float_t    ee_tol;   // relative residual target
    //  ee_coarse_iter / ee_coarse_tol:  the cap and the
    //  tolerance of the COARSEST level's nested PCG.  A solver
    //  setting,  not a method setting:  the coarse solve only
    //  conditions the V-cycle,  so shrinking it costs accuracy
    //  nothing and can pay 3x of a step.  Defaults are the
    //  measured pair ( STATUS.md ( 90 ) ).
    int   ee_coarse_iter;
    float_t ee_coarse_tol;
    ////////// Functions //////////
    __host__ par_t(  );
    __host__ void read( const input & );
};

////////////////////////////////////////////////////////////
//  The driver.  One instance per enrolled integrator,  so
//  the per-block arrays live in the block data ( fld_t )
//  and only the small reductions live here.

class driver_t;           // the step's driver ( edge_mg_driver.h )

class solver_t
{
public:
    ////////// Data //////////
    par_t               par;
    bool               ready;
    float_t *        p_dmax;   // device: max |div b| ( smoke )
    float_t        div_dmax;   // host  : last read-back
    //  the last solve, so a gate ( or a log ) can quote its
    //  cost
    int               last_it;
    int            last_nred;
    int           last_ncomm;
    float_t         last_res;
    driver_t *         p_dr;   // nullptr until the first ee_solve step
    ////////// Functions //////////
    __host__ solver_t(  );
    __host__ ~solver_t(  );
    __host__ void  read   ( const input & );
    __host__ void  setup  ( device::base_t & );
    __host__ void  step   ( device::base_t &, const void *,
                            view_t & );
    __host__ void  finalize( device::base_t & );
    //  the step's driver,  created on first use.  Only a GATE
    //  calls this,  to install the hierarchy report
    //  ( driver_t::audit,  usr_ext/mhd_ee/ee_audit.cpp );
    //  step( ) below is the production caller.
    __host__ driver_t * driver( void );
    //  the array traffic of the LAST implicit step, i.e. P5's
    //  exit meter ( zero reads and zero writes is the gate ).
    __host__ void  traffic_read
    ( int & n_read, int & n_write, long & b_read,
      long & b_write ) const;
    __host__ bool  traffic_on( void ) const;
};


////////////////////////////////////////////////////////////
//  P5's exit meter:  the array traffic of the LAST implicit
//  step ( solver_t::traffic_read above ).
//
//  The criterion is reads = 0 AND writes = 0:  inside one
//  implicit step,  no per-cell or per-face array may cross
//  the bus.  O(1) scalars do not count -- reading an
//  iteration count or a residual back is scheduling,  not a
//  violation of H1.  The setup traffic ( grid,  coefficients,
//  the numbering ) moves once and is not counted either.
//
//  P4/P5 read this from gate_e2e ( M9.5 );  the number they
//  must drive to zero is what the shipped step prints today.
//
//  The step state's host mirrors,  for a gate that must PROVE
//  a step copied nothing ( a hand-written meter can always be
//  wrong,  and it cannot see storage it never instruments ),
//  are NOT declared here:  that accessor is
//  driver_t::mirror( ) in edge_mg_driver.h,  which only a gate
//  includes.

////////////////////////////////////////////////////////////
//  The one piece of glue the module side needs:  build the
//  flat view from a block-data object.
//
//  This template is instantiated in whichever TU calls it
//  ( the module's ni_t::diffuse,  i.e. once per pgen ),  so
//  it must stay HOST-ONLY and free of device code --  the
//  point of the view is that no kernel ever sees bdt_T
//  ( rule C2,  foundations section 3.3 ).  The member names
//  below are the mhd block-data contract,  not a pgen's.

template< class bdt_T > __host__
inline view_t make_view( const bdt_T & b_d )
{
    view_t v;
    v.geo = b_d.geo;
    v.bf  = b_d.fld.bf;
    v.w   = b_d.hyd.w;
    v.a   = b_d.fld.a_edge;
    v.eta = b_d.fld.eta_cc;
    v.dt  = b_d.dt;
    return v;
};

};  // namespace edge_mg
