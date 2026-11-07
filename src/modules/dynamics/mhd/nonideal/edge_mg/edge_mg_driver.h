#pragma once

#include <cstddef>
#include <vector>

#include "edge_mg_iface.h"
#include "edge_mg_assemble.h"
#include "edge_mg_cplx.h"
#include "edge_mg_data.h"
#include "edge_mg_dev_vec.h"
#include "edge_mg_device.h"
#include "edge_mg_eta_dev.h"
#include "edge_mg_geo.h"
#include "edge_mg_global.h"
#include "edge_mg_grad.h"
#include "edge_mg_hier.h"
#include "edge_mg_mf.h"
#include "edge_mg_op.h"
#include "edge_mg_rep_dev.h"
#include "edge_mg_solve.h"
#include "edge_mg_wb_dev.h"

////////////////////////////////////////////////////////////
//  The implicit step's driver:  the state it keeps across
//  steps,  and every host-side action of one step.
//
//  This header is INTERNAL to the module -- it is not the
//  pgen's interface ( that is edge_mg_iface.h,  rule C7 ).
//  Nothing here defines a body:  the declarations are
//  satisfied by edge_mg_driver.cpp,  and only the two
//  templates ( the geometry helpers ) are instantiated
//  there.
//
//  `solver_t` owns one of these ( edge_mg_iface.h ),  so the
//  whole host side of the leg is a member of one type
//  instead of a scatter of namespace-level functions.

namespace mhd_ee::ni::edge_mg
{

////////////////////////////////////////////////////////////
//  The state of one implicit step.  It is created once
//  (`new`, never copied) and lives across steps,  which is
//  what makes the matrix-free backend's x a WARM START.

struct step_state_t
{
    bool                  built { false };
    int                  n_edge { 0 };
    int                  n_cell { 0 };
    int                 n_face  { 0 };
    cplx_t                   cx;
    std::vector< int > keep, comp, keep_f;
    int               n_compact { 0 };
    //  the compact M1,  for the right-hand side M1 a_old and
    //  for the bf -> a bridge ( the representation-in ).
    csr_t                    M1;
    bool         bridge_done { false };
    //  device scratch ( owned here,  freed by driver_t::free )
    void *                p_blk { nullptr };
    void *                p_lat { nullptr };
    void *                p_gr  { nullptr };
    void *                p_xc  { nullptr };
    //  host mirrors.  A matrix-free step does NOT touch them;
    //  test_mode 15 checks that by poisoning them
    //  ( dead-ends M81 ).
    std::vector< cell_blk_t > blk_h;
    std::vector< float2_t >   lat_h;
    std::vector< float_t  >   eta_h;
    std::vector< float_t  >     a_h;
    std::vector< float2_t >    bf_h;
    std::vector< float_t >       far;
    //  the matrix-free backend:  the device hierarchy is
    //  extracted once from the assembled one,  and the
    //  assembled one is then not built at all ( see
    //  edge_mg_mf.h ).
    mf_t                     mf;
    bool                mf_ready { false };
    //  the device tail:  the representation bridge ( K6 ) and
    //  the face write-back ( K7 ) run on the arrays the step
    //  already owns,  so a step copies no per-cell or
    //  per-face array at all.
    rep_dev_t              rep_dev;
    wb_dev_t                    wb;
    wb0_dev_t                  wb0;
    bool               tail_ready { false };
    float_t *           p_rep_old { nullptr };
    float_t *           p_rep_new { nullptr };
};

////////////////////////////////////////////////////////////
//  One implicit step of one block,  and the state it
//  carries.  The seven phases of a step are private
//  methods here,  in order:
//
//  step() -> _allocate,  _lattice,  ( _gradient ),
//            _blocks,  _build ( once ),  _solve,  _tail
//
//  and the setup-time jobs it calls ( the complex,  the
//  face areas,  the device eta assembler,  the M12 product,
//  the bridge,  the G-AGG check ) are private methods too.
//  The one exception is the traffic meter:  it is part of
//  the step state,  not of a phase.

class driver_t
{
public:
    ////////// Data //////////
    //  the state ( the gates read its mirrors through
    //  mirror( ),  nothing else does )
    step_state_t                  st;
    ////////// Functions //////////
    __host__ driver_t(  );
    __host__ ~driver_t(  );
    //  one step:  assemble,  solve,  write back.
    __host__ void step
    ( device::base_t & dev, const void * stream, view_t & v,
      const par_t & par, solve_stat_t & out );
    //  release every device buffer ( the solver's finalize )
    __host__ void free(  );
    //  the array traffic of the LAST step ( the P5 exit meter )
    __host__ void traffic
    ( int & n_read, int & n_write, long & b_read,
      long & b_write ) const;
    __host__ bool traffic_on(  ) const;
    //  the step state's host mirrors,  for a gate that must
    //  PROVE a step copied nothing.  `what` is 0..5 for blk_h
    //  / lat_h / eta_h / a_h / bf_h / far,  6 for the host
    //  solution vector,  7 for the matrix-free device vector;
    //  `n` is the size and `on_dev` says whether `pp` is a
    //  device pointer.
    __host__ void mirror
    ( const int & what, void ** pp, int & n, int & on_dev );
    //  A GATE's report on the hierarchy it just built.  The
    //  module asserts the two invariants of check_hierarchy( )
    //  and throws;  the numbers a human reads -- the per-level
    //  ladder,  the Galerkin image,  the smoother's lambda --
    //  are a diagnostic,  and they live in the pgen that wants
    //  them ( usr_ext/mhd_ee/ee_audit.cpp installs this ).
    //  The cycle's scalar is passed in,  so the report never
    //  runs the cycle twice.
    //  Production leaves the hook null and prints nothing.
    using hier_audit_t = void (*)
    ( void * ctx, device::base_t & dev, const hier_t & h,
      const std::vector< op_lvl_t > & op, mf_t & mf,
      const double & contraction );
    __host__ void audit( void * ctx, hier_audit_t fn );

private:
    ////////// The meter //////////
    bool                    tr_on { false };
    int                tr_n_read { 0 };
    int               tr_n_write { 0 };
    long                 tr_b_read { 0 };
    long                tr_b_write { 0 };
    //  the device this driver last ran on ( for the destructor
    //  only:  a step may be the last thing before teardown )
    device::base_t *          p_dev { nullptr };
    //  the gate's hierarchy report ( null in production )
    void *            p_audit_ctx { nullptr };
    hier_audit_t           p_audit { nullptr };
    __host__ void read_arr
    ( const std::size_t & n, const std::size_t & el );
    __host__ void write_arr
    ( const std::size_t & n, const std::size_t & el );
    ////////// The setup-time jobs //////////
    __host__ void release( device::base_t & );
    __host__ void build_complex
    ( cplx_t & c, const mesh::geo_t & geo, const idx_t & n_ceff,
      const bool ( & per )[ 3 ] ) const;
    template< class geo_T >
    __host__ void face_areas
    ( const geo_T & g, const idx_t & i,
      float2_t ( & area )[ 6 ] ) const;
    template< class geo_T >
    __host__ void cells_dev_eta
    ( device::base_t & dev, const geo_T & g, const cplx_t & c,
      const float2_t * p_eta, const float2_t * p_deta,
      const float2_t * p_xc, cell_blk_t * p_dev,
      int * p_n_thr );
    __host__ void prod_rhs_m12
    ( const mesh::geo_t & geo, const idx_t & n_ceff,
      const bool ( & per )[ 3 ], const cell_blk_t * p_blk,
      const float2_t * p_b, float2_t * p_r ) const;
    static __host__ bool a_all_zero
    ( const std::vector< float_t > & a );
    __host__ float_t a_from_bf
    ( device::base_t & dev, const view_t & v, const cplx_t & c,
      const geo_sph_t & g, const bool ( & per )[ 3 ],
      const float_t & tol );
    __host__ void check_hierarchy
    ( device::base_t & dev, const hier_t & h,
      const std::vector< op_lvl_t > & op, mf_t & mf );
    ////////// The seven phases of one step //////////
    __host__ void _allocate
    ( device::base_t & dev, const view_t & v, const cplx_t & c );
    __host__ void _lattice
    ( device::base_t & dev, const view_t & v );
    __host__ void _gradient
    ( device::base_t & dev, const geo_sph_t & g, const cplx_t & c,
      float2_t *& p_deta, float2_t *& p_xc );
    __host__ void _blocks
    ( device::base_t & dev, const geo_sph_t & g, const cplx_t & c,
      const float2_t * p_deta, const float2_t * p_xc );
    __host__ void _build
    ( device::base_t & dev, view_t & v, const par_t & par,
      const geo_sph_t & g, const cplx_t & c,
      const bool ( & per )[ 3 ], const float_t & dt );
    __host__ void _bridge_in
    ( device::base_t & dev, const view_t & v, const par_t & par,
      const geo_sph_t & g, const cplx_t & c,
      const bool ( & per )[ 3 ] );
    __host__ void _tail_setup
    ( device::base_t & dev, view_t & v, const par_t & par,
      const geo_sph_t & g );
    __host__ void _solve
    ( device::base_t & dev, const view_t & v, const par_t & par,
      const float_t & dt, solve_stat_t & out );
    __host__ void _tail
    ( device::base_t & dev, view_t & v, const par_t & par,
      const float_t & dt, solve_stat_t & out );
};

};  // namespace mhd_ee::ni::edge_mg