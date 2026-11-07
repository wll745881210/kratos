#pragma once

#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../../../../../src/types.h"
#include "../../../../../../src/device/device.h"

//  the __global__ layer of this file lives in the kernels/ subdirectory here.
#include "kernels/ni_mg_kernels.h"

////////////////////////////////////////////////////////////
//  The SCALAR ( float_t ) device-vector primitives.  They
//  are separate from the multigrid ops of edge_mg_device.h
//  for the reason recorded at 9z-n: a dot is a GLOBAL
//  reduction, not a multigrid op.
//
//  Why a second file instead of overloading the first one:
//  `vec_t` is
//  float2_t ( the complex E-field of the production leg ),
//  and a solver
//  vector is a plain float_t array. They cannot share a
//  buffer type, and
//  templating one on the other would put the complex
//  arithmetic into every
//  CG iteration. So the two layers are separate, and the
//  REDUCTION SCHEME
//  is deliberately identical -- the same fixed two-level
//  tree:
//
//  stage 1 block b sums its own contiguous slice in index
//  order,
//                serially,  into p_part[ b ]
//  stage 2 the host sums p_part[ 0 .. n_blk-1 ] in index
//  order
//
//  Determinism is a hard requirement ( 9z-d ): a Krylov
//  iteration whose
//  dot product can come out differently twice cannot be
//  gated bitwise, and
//  cannot be reproduced from a checkpoint. n_blk is a
//  CONSTANT of the
//  solve,  not a tuning knob,  for exactly that reason.

namespace mhd_ee::ni::edge_mg
{
using type::float_t;


////////////////////////////////////////////////////////////
//  y = x

struct real_copy_t
{
    const float_t * p_x;
    float_t       * p_y;

    __device__ void apply( const int & i ) const
    {
        p_y[ i ] = p_x[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim, const int & n ) const;
};

////////////////////////////////////////////////////////////
//  y += a x ( covers x += al p with a = al, and r -= al Ap
//  with
//  a = -al: ONE kernel for both CG updates )

struct real_axpy_t
{
    const float_t * p_x;
    float_t       * p_y;
    float_t           a;

    __device__ void apply( const int & i ) const
    {
        p_y[ i ] += a * p_x[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim, const int & n ) const;
};

////////////////////////////////////////////////////////////
//  y = x + be y ( the search-direction update; a plain y =
//  x + be y
//  cannot be written as an axpy of a THIRD buffer without
//  one more copy )

struct real_xpay_t
{
    const float_t * p_x;
    float_t       * p_y;
    float_t          be;

    __device__ void apply( const int & i ) const
    {
        p_y[ i ] = p_x[ i ] + be * p_y[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim, const int & n ) const;
};

////////////////////////////////////////////////////////////
//  y = a x element-wise ( the Jacobi scaling inside
//  Chebyshev )

struct real_dmul_t
{
    const float_t * p_a;
    const float_t * p_x;
    float_t       * p_y;

    __device__ void apply( const int & i ) const
    {
        p_y[ i ] = p_a[ i ] * p_x[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim, const int & n ) const;
};

////////////////////////////////////////////////////////////
//  y = a x + be y ( the Chebyshev direction update, which
//  needs a scale
//  on BOTH the new term and the carried one -- `real_xpay`
//  is the a = 1 case )

struct real_axpby_t
{
    const float_t * p_x;
    float_t       * p_y;
    float_t           a;
    float_t          be;

    __device__ void apply( const int & i ) const
    {
        p_y[ i ] = a * p_x[ i ] + be * p_y[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim, const int & n ) const;
};

////////////////////////////////////////////////////////////
//  stage 1 of the reduction: one serial partial sum per
//  block

struct real_dot_part_t
{
    const float_t * p_x;
    const float_t * p_y;
    int              n;
    int      chunk_len;
    float_t *     p_part;

    __device__ void apply(  ) const
    {
        const int b = int( blockIdx.x );
        if( int( threadIdx.x ) != 0 )
        {
            return;
        }
        const int i0 = b * chunk_len;
        //  NOT std::min: constexpr __host__ only, so nvcc
        //  refuses it from a
        //  __device__ function ( foundations rule C9 )
        const int i1 = ( i0 + chunk_len < n ) ? ( i0 + chunk_len ) : n;
        float_t s( 0. );
        for( int i = i0; i < i1; ++ i )
        {
            s += p_x[ i ] * p_y[ i ];
        }
        p_part[ b ] = s;
        return;
    };
};




////////////////////////////////////////////////////////////
//  Stage 2 of the reduction AND the coefficient updates:
//  one thread,  on the device,  so the host never finishes a
//  reduction and never computes a CG scalar.  The ops that
//  need only one thread ( sum of the partials,  alpha,  beta,
//  the residual,  omega ) all come through `real_one_k`.

struct real_sum_part_t
{
    const float_t * p_part;
    int              n_blk;
    float_t *          p_out;

    __device__ void apply(  ) const
    {
        float_t s( 0. );
        for( int b = 0; b < n_blk; ++ b )
        {
            s += p_part[ b ];
        }
        p_out[ 0 ] = s;
        return;
    };
};

////////////////////////////////////////////////////////////
//  The same two stages as `real_dot`,  but the scalar STAYS
//  on the device:  no `f_cp`,  no sync,  and the partials are
//  summed in the same order,  so the value is bit-identical
//  to the old host-side finish.

__host__ void real_dot_dev
( device::base_t & dev, const float_t * p_x, const float_t * p_y,
  float_t * p_part, const int & n, const int & n_blk,
  float_t * p_out, const int & n_th_lim = 256 );

////////////////////////////////////////////////////////////
//  The updates that read a scalar FROM THE DEVICE,  so the
//  host does not have to know it.  `s` is a sign (+-1) and
//  therefore a constant,  not an arithmetic input.

////////////////////////////////////////////////////////////
//  The SCALAR half of a CG iteration,  as one-thread ops on
//  `real_one_k` ( so the host never computes in a step ).
//
//  They used to sit in `edge_mg_mf.h` next to the multigrid;
//  they are generic --  b2 guard,  alpha with a curvature
//  breakdown flag,  beta,  residual -- so they live here,  and
//  both the matrix-free MG and the K6 face solve use them.
//
//  Every coefficient is read BY the vector kernels from the
//  device,  so an update can never race the scalar that feeds
//  it.
struct cg_b2s_t
{
    const float_t * p_b2;
    float_t       * p_b2s;

    __device__ void apply(  ) const
    {
        p_b2s[ 0 ] = ( * p_b2 ) > float_t( 0. )
                   ? ( * p_b2 ) : float_t( 1. );
        return;
    };
};

struct cg_alpha_t
{
    const float_t * p_rz;
    const float_t * p_pap;
    float_t       * p_al;
    float_t       * p_ok;

    __device__ void apply(  ) const
    {
        const float_t ok = ( * p_pap ) > float_t( 0. );
        p_ok[ 0 ] = ok;
        p_al[ 0 ] = ok ? ( * p_rz ) / ( * p_pap ) : float_t( 0. );
        return;
    };
};

struct cg_beta_t
{
    const float_t * p_rn;
    float_t       * p_rz;
    float_t       * p_be;

    __device__ void apply(  ) const
    {
        p_be[ 0 ] = ( * p_rn ) / ( * p_rz );
        p_rz[ 0 ] = * p_rn;
        return;
    };
};

struct cg_res_t
{
    const float_t * p_rr;
    const float_t * p_b2s;
    float_t       * p_res;

    __device__ void apply(  ) const
    {
        p_res[ 0 ] = sqrt( ( * p_rr ) / ( * p_b2s ) );
        return;
    };
};

struct real_axpy_dev_t
{
    const float_t * p_x;
    float_t       * p_y;
    const float_t * p_a;
    float_t           s;
    //  When the coefficient comes from a device scalar that may also
    //  carry a breakdown flag ( the CG's `alpha` ),  the update reads
    //  the flag and does NOTHING on a breakdown.  Without the guard a
    //  zero coefficient times a NaN search direction would still write
    //  NaN ( 0 * NaN = NaN ),  so "nothing moved" would not hold.
    const float_t * p_ok { nullptr };

    __device__ void apply( const int & i ) const
    {
        if( p_ok != nullptr && !( * p_ok > float_t( 0. ) ) )
        {
            return;
        }
        p_y[ i ] += s * ( * p_a ) * p_x[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim, const int & n ) const;
};

struct real_xpay_dev_t
{
    const float_t * p_x;
    float_t       * p_y;
    const float_t * p_be;
    const float_t * p_ok { nullptr };

    __device__ void apply( const int & i ) const
    {
        if( p_ok != nullptr && !( * p_ok > float_t( 0. ) ) )
        {
            return;
        }
        p_y[ i ] = p_x[ i ] + ( * p_be ) * p_y[ i ];
        return;
    };

    __host__ std::tuple< dim3, dim3, int >
    resource( const int & n_th_lim, const int & n ) const;
};

__host__ void real_axpy_dev
( device::base_t & dev, const float_t * p_x, float_t * p_y,
  const float_t * p_a, const float_t & s,
  const int & n, const int & n_th_lim = 256,
  const float_t * p_ok = nullptr );

__host__ void real_xpay_dev
( device::base_t & dev, const float_t * p_x, float_t * p_y,
  const float_t * p_be, const int & n, const int & n_th_lim = 256,
  const float_t * p_ok = nullptr );

////////////////////////////////////////////////////////////
//  host entries. No sync inside: the caller decides when to
//  wait for the
//  bus ( the CG needs the scalar before it can go on, the
//  axpys do not ).

__host__ void real_copy
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const int & n,
  const int & n_th_lim = 256 );

__host__ void real_axpy
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const float_t & a,
  const int & n, const int & n_th_lim = 256 );

__host__ void real_xpay
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const float_t & be,
  const int & n, const int & n_th_lim = 256 );

__host__ void real_dmul
( device::base_t & dev, const float_t * p_a, const float_t *
    p_x,
  float_t * p_y, const int & n, const int & n_th_lim = 256 );

__host__ void real_axpby
( device::base_t & dev, const float_t * p_x, float_t * p_y,
    const float_t & a,
  const float_t & be, const int & n, const int & n_th_lim =
      256 );

//  the scalar: stage 1 on the device into a caller-owned
//  n_blk buffer, then
//  the host sums those n_blk numbers IN ORDER. Same value
//  twice, always.
__host__ float_t real_dot
( device::base_t & dev, const float_t * p_x, const float_t *
    p_y,
  float_t * p_part, const int & n, const int & n_blk,
  const int & n_th_lim = 256 );

////////////////////////////////////////////////////////////
//  Batched real dot products
////////////////////////////////////////////////////////////

struct real_dot_batch_stat_t
{
    int n_dot  { 0 };  //  scalar dot products
    int n_comm { 0 };  //  one readback if nonempty
};

//  The workspace owns all buffers.  Unchanged pointers need
//  no repeated copy;  copying is disabled.
struct real_dot_batch_ws_t
{
    int  n_dot;  //  active prefix
    int  n_cap;  //  allocated pointer count
    int  n_blk;

    //  pp_x has n_cap entries;  partials have n_cap*n_blk.
    const float_t **  pp_x;
    float_t *         p_part;
    std::vector< float_t > h_part;

    real_dot_batch_ws_t(  )
    {
        n_dot = 0;
        n_cap = 0;
        n_blk = 0;
        pp_x = nullptr;
        p_part = nullptr;
        return;
    };

    real_dot_batch_ws_t
    ( const real_dot_batch_ws_t & ) = delete;
    real_dot_batch_ws_t & operator =
    ( const real_dot_batch_ws_t & ) = delete;

    __host__ void free( device::base_t & dev );

    //  Pointer upload is setup, not a dot batch.  Vector
    //  contents change freely;  pointer changes need set_x.
    __host__ void set_x
    ( device::base_t & dev,
      const std::vector< const float_t * > & p_x );

    //  Select an uploaded CGS2 prefix without copying.
    __host__ void set_active( const int & n_active );

    __host__ void init
    ( device::base_t & dev,
      const std::vector< const float_t * > & p_x,
      const int & n_blk_ );
};

//  Reproduce real_dot per j: serial contiguous chunks in
//  increasing index order, with row-major partial sums.
struct real_dot_batch_part_t
{
    const float_t * const * pp_x;
    const float_t *          p_y;
    int                       n;
    int                chunk_len;
    int                    n_blk;
    float_t *              p_part;

    __device__ void apply(  ) const
    {

        const int flat = int( blockIdx.x );
        const int b = flat % n_blk;
        const int j = flat / n_blk;
        if( int( threadIdx.x ) != 0 )
        {
            return;
        }
        const int i0 = b * chunk_len;
        const int i1 = ( i0 + chunk_len < n )
                     ? ( i0 + chunk_len ) : n;
        const float_t * p_x = pp_x[ j ];
        float_t s( 0. );
        for( int i = i0; i < i1; ++ i )
        {
            s += p_x[ i ] * p_y[ i ];
        }
        p_part[ j * n_blk + b ] = s;
        return;
    };
};


//  One kernel and one readback per batch.  Host block-order
//  sums match real_dot;  n_comm counts this communication.
//  Any MPI allreduce remains a separate caller operation.
__host__ real_dot_batch_stat_t real_dot_batch
( device::base_t & dev, real_dot_batch_ws_t & ws,
  const float_t * p_y, const int & n,
  std::vector< float_t > & dots, const int & n_th_lim = 256
      );

};  // namespace edge_mg
