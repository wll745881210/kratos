#pragma once

#include "edge_mg_cplx.h"
#include "edge_mg_csr.h"
#include "edge_mg_geo.h"

////////////////////////////////////////////////////////////
//  ============ THE SEAM,  AFTER THE SHAKE-OUT
//  ============
//
//  The implicit step asks exactly ONE question:
//
//  solve A x = b , A = M1 + dt K ( SPD, compact )
//
//  Everything to the LEFT of that line is the OPERATOR: the
//  assembly,
//  the eta lattice and its sub-cell gradient, the pole
//  fold, the
//  right-hand-side rule, the write-back, the SMR exchange.
//  NOTHING on
//  the left may know how the question is answered.
//
//  Everything to the RIGHT is a BACKEND, and a backend is
//  disposable by
//  design. This header used to carry five of them ( PCG,
//  the host edge
//  MG, MGPCG, the Pi composite and the device MGPCG ) plus
//  the `ctx_t`
//  that held the host hierarchy. The 2026-11-06 decision
//  retired all of
//  them: the shipped backend is `mf_t` in edge_mg_mf.h, a
//  device-resident
//  matrix-free V-cycle whose per-level blocks come from the
//  same canonical
//  `P` ( see edge_mg_hier.h ). What survives here is the
//  CONTRACT and
//  the PODs the driver fills in.
//
//  WHAT A BACKEND MUST SATISFY ( the contract, in order of
//  importance )
//
//  (1) A is SPD and is reachable ONLY through a matvec. A
//  backend that
//  needs the entries is not a backend, it is a new
//  operator.
//  (2) Its reported residual is || b - A x || / || b ||
//  computed with
//  the SAME A, so the number is comparable across
//  backends.
//  (3) x is a WARM START. A backend IMPROVES x, it never
//  resets it.
//  The implicit step reuses the previous step's solution,
//  and that
//  is what makes "iterations per step" a meaningful number
//  ( the M9
//  gate ). `mf_t::solve( ...,  warm )` is the only place
//  that decides
//  this now, and `warm = true` is what the driver passes.
//  (4) It reports its cost: iterations AND the number of
//  GLOBAL
//  reductions (`n_red`). The reduction count is the
//  quantity that
//  actually caps scalability on many devices.
//
//  A NOTE FOR THE NEXT BACKEND. The host objects that used
//  to be the
//  cross-check ( `mg_t` + `csr_t::mul` ) are gone from the
//  production
//  tree; the cross-check now lives in the reference TU (
//  edge_mg_ref.cpp )
//  and in test_mode 15, which rebuilds the INDEPENDENT
//  assembled operator
//  with `op_build` ( edge_mg_op.h ) and solves with
//  `csr_t` + `cg`.

namespace mhd_ee::ni::edge_mg
{
using type::float_t;

#if PRECISION == 2
constexpr float_t edge_mg_default_tol = 1.e-10;
constexpr float_t edge_mg_default_ee_tol = 1.e-12;
#else
constexpr float_t edge_mg_default_tol = 1.e-6;
constexpr float_t edge_mg_default_ee_tol = 1.e-5;
#endif

//////////////////////////////////////////////////////////
//  Knobs, per solve.  `method` is gone: there is one
//  backend.
struct solve_par_t
{
    int       it_max { 200 };
    float_t       tol { edge_mg_default_tol };
};

//  The Krylov statistics a device solve returns:  iterations,
//  dot products ( the quantity that caps scalability ) and the
//  recursive residual.  It used to live in `edge_mg_krylov.h`,
//  which had two CG implementations -- a host vector one ( no
//  clients left ) and `pcg_dev_t`.  Since K6 computes its own
//  scalars on the device ( `edge_mg_rep_dev.h` ), the header is
//  gone and this is the one statistics type the device solvers
//  share.
struct pcg_stat_t
{
    int        it { 0 };
    int     n_red { 0 };
    float_t   res { 0. };
};

struct solve_stat_t
{
    int         it   { 0 };
    int       n_red  { 0 };  // scalar dot products
    int       n_comm { 0 };  // batch readbacks on device
    float_t     res   { 0. };
};

};  // namespace mhd_ee::ni::edge_mg