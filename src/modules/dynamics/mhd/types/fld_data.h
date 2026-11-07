#pragma once

#include "../../hydro/types/hyd_data.h"

namespace dynamics::mhd
{
////////////////////////////////////////////////////////////
// Types

using namespace mesh;

struct fld_t
{
    ////////// Types //////////
    using  f_t = type:: float_t;
    using f2_t = type::float2_t;

    ////////// Data //////////
    dat_3d_t< f2_t > bf;
    dat_3d_t<  f_t > ec;      // Electric field on cell
    dat_3d_t<  f_t > ee;      // Electric field on edge
    //  Non-ideal MHD ( modules/dynamics/mhd/nonideal ):
    //  diffusion emf on the ee edge lattice, kept separate
    //  from ee ( which holds ideal + diffusion ) so the
    //  Poynting energy flux sees the diffusion part only;
    //  cell-centred separable heating diagnostics
    //  ( [0] Ohmic eta_O J^2, [1] ambipolar eta_A J_perp^2;
    //  the Hall term heats nothing since
    //  J . ( J x B ) = 0 ).
    //  Allocated only when a nonideal integrator enrolled
    //  itself via the flag below ( its init runs before
    //  mesh generation, hence before any setup( ) ).
    dat_3d_t<  f_t > e_dif;
    dat_3d_t<  f_t > j_edge;  // currents on the ee lattice
    //  The edge operator's per - edge diagonal
    //  1 + dt eta |( C2 C1 1 )_e|,  for the diagonal -
    //  preconditioned Richardson.  Same lattice as e_dif /
    //  j_edge.
    dat_3d_t<  f_t > e_diag;
    //  The HX/AMS auxiliary space -- NODAL ( corner ) arrays on
    //  the same box as e_dif:  nod_w = G_inc^T b ( the
    //  auxiliary right - hand side ),  nod_p = the nodal
    //  potential being solved for.
    dat_3d_t<  f_t > nod_w, nod_p;
    dat_3d_t<  f_t > heat;
    //  The STS sweep's RKC recursion:  B_{ j - 2 }  storage
    //  ( the face field,  single slice ).
    dat_3d_t< f2_t > bf_prev;
    //  Implicit ( Gauss-Seidel ) solver work array:  the
    //  face-centred unknown x ( same face box as bf,
    //  single slice ).   Allocated only for the implicit
    //  nonideal mode ( implicit.h ).
    dat_3d_t< f2_t > cg_x;
    //  Divergence - projection scratch:  the potential and its
    //  Jacobi partner,  plus the face - flux imbalance
    //  S = V ( D.B )  the projection solves for.  All three are
    //  float2_t ( the face field already is ) and they carry NO
    //  ghost ring -- an out - of - mesh potential is 0 (
    //  Dirichlet ) and the curvilinear phi axis is periodic.
    dat_3d_t< f2_t > phi;
    dat_3d_t< f2_t > prhs;
    //  Edge-centred implicit tensor diffusion
    //  ( modules/dynamics/mhd/nonideal/edge_mg ):  the edge
    //  1-form work array ( n_int = 3,  one ghost ring,  the
    //  same lattice as ee / j_edge ) and the per-cell
    //  symmetric eta tensor in Voigt order {xx,yy,zz,xy,xz,
    //  yz}.  Allocated only when that leg enrolled itself.
    dat_3d_t<  f_t > a_edge;
    dat_3d_t<  f_t > eta_cc;
    inline static bool need_ni     = false;
    inline static bool need_ni_imp = false;
    inline static bool need_ni_proj = false;
    inline static bool need_ni_ee  = false;

    ////////// Functions //////////
    __host__ __device__ __forceinline__ int & order(  )
    {
        return bf.n_gh[ 0 ];
    };
    __host__ __device__ __forceinline__
    const int & order(  ) const
    {
        return bf.n_gh[ 0 ];
    };

    __host__ virtual int & n_stp (          ) const;
    __host__ virtual std::string   prefix(  ) const;
    __host__ virtual void setup  ( const    f_new_t & f_n  ,
                                   const      geo_t & geo );
    __host__ virtual void free   ( const   f_free_t & f_f );
    __host__ virtual void read   ( const   f_read_t & f_r );
    __host__ virtual void write  ( const  f_write_t & f_w );
    __host__ virtual void copy_to( const     f_cp_t & f_c  ,
                                   fld_t     &  tgt ) const;
};

};                              // namespace dynamics::mhd
