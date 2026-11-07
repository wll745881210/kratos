//  ni_mg_kernels.h -- the __global__ layer of the matrix-free multigrid kernels.
//  Only kernels live here ( the project rule:  `__global__` appears nowhere else ).
#pragma once

namespace mhd_ee::ni::edge_mg
{


//============================================================
//  from edge_mg_dev_vec.h
//============================================================

template< class op_T >
__global__ void real_dot_batch_part_k
( const op_T op, const int n_task )
{
    if( int( blockIdx.x ) < n_task )
    {
        op.apply(  );
    }
    return;
};

template< class op_T >
__global__ void real_one_k( const op_T op )
{
    op.apply(  );
    return;
};

template< class op_T >
__global__ void real_dot_part_k( const op_T op, const int n_blk )
{
    if( int( blockIdx.x ) < n_blk )
    {
        op.apply(  );
    }
    return;
};

template< class op_T >
__global__ void real_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

//============================================================
//  the device-vector primitives ( see edge_mg_dev_vec.h )
//============================================================

template< class op_T >
__global__ void dot_part_k( const op_T op, const int n_blk )
{
    if( int( blockIdx.x ) < n_blk )
    {
        op.apply(  );
    }
    return;
};

template< class op_T >
__global__ void v_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

//============================================================
//  from edge_mg_device.h
//============================================================

template< class op_T >
__global__ void matvec_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

template< class op_T >
__global__ void blk_scale_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

template< class op_T >
__global__ void blk_solve_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T >
__global__ void mf_edge_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T >
__global__ void matvec_t_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

template< class op_T >
__global__ void rhs_m12_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T >
__global__ void blk_assemble_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    //  Record who owns the LAST cell. gridDim*blockDim only
    //  reports the
    //  requested geometry; a backend that runs a single
    //  thread would still
    //  report it. If cell n_cell-1 is owned by thread
    //  n_cell-1, that thread
    //  really did run.
    if( flat == n_cell - 1 && op.p_n_thr != nullptr )
    {
        *op.p_n_thr = flat + 1;
    }
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

//============================================================
//  from edge_mg_mf.h
//============================================================

template< class op_T >
__global__ void expand_a_k( const op_T op, const int n_edge )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int e = flat; e < n_edge; e += stride )
    {
        op.apply( e );
    }
    return;
};

template< class op_T >
__global__ void gather_a_k( const op_T op, const int n_edge )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int e = flat; e < n_edge; e += stride )
    {
        op.apply( e );
    }
    return;
};

template< class op_T >
__global__ void resid_fix_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

template< class op_T >
__global__ void blk_blk_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T >
__global__ void mf_blk_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T > __global__ void diag_inv_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

template< class op_T > __global__ void diag_upd_k( const op_T op, const int n )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int i = flat; i < n; i += stride )
    {
        op.apply( i );
    }
    return;
};

template< class op_T > __global__ void blk_diag_k( const op_T op,
                                                   const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

template< class op_T >
__global__ void blk_agg_k( const op_T op, const int n_cell )
{
    const int flat   = int( blockIdx.x * blockDim.x + threadIdx.x );
    const int stride = int( blockDim.x * gridDim.x );
    for( int ic = flat; ic < n_cell; ic += stride )
    {
        op.apply( ic );
    }
    return;
};

}  // namespace mhd_ee::ni::edge_mg
