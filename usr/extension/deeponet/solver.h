#pragma once

#include "../chem_therm/solver/solver.h"
#include "model.h"
#include "nn_bat.h"

#include <map>
#include <set>
#include <type_traits>
#include <vector>
#include <cstdio>

// Per-cell kernel prints are DIAGNOSTICS ONLY: CUDA printf
// links its machinery into the whole kernel ( registers +
// binary size even when the call is never executed ), so the
// final/production build compiles them OUT.  Define
// DNN_KERNEL_PRINTF ( or build with -DDNN_KERNEL_PRINTF ) to
// re-enable them for debugging.  Production observability is
// via the cnt[] counters below ( host-read in free( ) ).
// #define DNN_KERNEL_PRINTF

// cnt[] slots ( device unsigned long long[ 16 ], allocated
// lazily in integrate( ), zeroed once, host-read + freed in
// free( ) ):
//   0  di_cells    full-budget di_step invocations
//   1  di_sub      total substeps those consumed
//   2  di_lu       LU failures inside them
//   3  packed      cells queued for the NN
//   4  nn_commit   NN commits
//   5  rn_commit   renormalized commits ( kind-2 below sum_fb )
//   6..9           guard kind 1..4 trips
//   10 fb          full-DI cleanups ( guard fallbacks )
//   11 probe_ok    probe-converged direct commits
//   12 grind_rn    RN / clamped commits of guard-tripped
//                  cells whose cleanup-DI already failed
//                  dnn_grind_cap times ( anti-grind, rc-40 )
//   13 fb_failed   cleanup solves that exhausted the budget
//                  or hit an LU failure ( grind increments )

namespace extension::deeponet
{
//////////////////////////////////////////////////////////
// Per-cell stiffness counters ( substeps the direct
// integrator needed last time ). Host-side registry keyed
// by the block dual; the device pointers are passed to
// the kernel as an extra argument.

__host__ inline std::map< const void *, int * > &
stp_registry( )
{
    static std::map< const void *, int * > reg;
    return reg;
};

//////////////////////////////////////////////////////////
// Device helpers

__device__ __forceinline__ float nn_gelu
( const float & x )
{
    return 0.5f * x * ( 1.f + erff( x * 0.70710678118f ) );
};

// Box-Cox forward, floored exactly like python/train.py
__device__ __forceinline__ float nn_bc_fwd
( const float & x, const float & lam )
{
    const float xf = fmaxf( x, 1e-30f );
    if( fabsf( lam ) < 1e-8f )
        return logf( xf );
    return ( powf( xf, lam ) - 1.f ) / lam;
};

// Box-Cox inverse, floored at zero exactly like train.py
__device__ __forceinline__ float nn_bc_inv
( const float & v, const float & lam )
{
    if( fabsf( lam ) < 1e-8f )
        return expf( v );
    return powf
        ( fmaxf( 1.f + lam * v, 0.f ), 1.f / lam );
};

// One MLP layer for a BATCH of cells, computed cooperatively
// by the whole block. Work items are ( cell, neuron ) pairs:
// threads stride linearly, so consecutive threads read
// consecutive W columns ( coalesced ) and adjacent activations.
// Weights are read once per ( neuron, i ) row and reused across
// the batch cells in registers/loop. src: [ n_cell x n_in ],
// dst: [ n_cell x n_out ], both in shared memory.
// Synchronizes before returning.
__device__ __forceinline__ void nn_layer_b
( const float * src, float * dst,
  const int & n_cell, const int & n_in, const int & n_out,
  const float * w_dev,
  const int & off_w, const int & off_b,
  const bool & act )
{
    const float * W = w_dev + off_w;
    const float * b = w_dev + off_b;
    const int n_it( n_cell * n_out );
    for( int it( threadIdx.x ); it < n_it; it += blockDim.x )
    {
        const int c( it / n_out );
        const int o( it % n_out );
        float acc = b[ o ];
        const float * s( src + c * n_in );
        for( int i = 0; i < n_in; ++ i )
            acc += s[ i ] * W[ i * n_out + o ];
        dst[ c * n_out + o ] = act ? nn_gelu( acc ) : acc;
    };
    __syncthreads( );
    return;
};

//////////////////////////////////////////////////////////
// Kernel A ( probe + DI ). Runs the direct integrator,
// but for in-envelope cells first tries probe_try solve( )
// attempts: cells that converge early commit via DI ( work
// kept, zero waste ); cells that fail the attempts pack
// their z-scored inputs into the global slab and defer to
// the batched-NN kernel. pass = 0 : normal; pass = 1 :
// cleanup, only cells whose ok[ c ] == 2 run DI.

template< class sol_T, class bdt_T, class rls_T,
          class eos_T, class spc_T >
__global__ void ker_di
( const sol_T sol, const bdt_T bdt, const rls_T rls,
  const eos_T eos, int * stp, bat_t bat, const int pass,
  const spc_T work )
{
    return sol( bdt, rls, eos, stp, bat, pass, work );
};

template< class sol_T, class bdt_T, class rls_T,
          class eos_T, class spc_T >
__global__ void ker_di_cleanup
( const sol_T sol, const bdt_T bdt, const rls_T rls,
  const eos_T eos, int * stp, bat_t bat, const int pass,
  const spc_T work )
{
    return sol( bdt, rls, eos, stp, bat, pass, work );
};

//////////////////////////////////////////////////////////
// Kernel B ( batched NN + commit ). Persistent grid over
// the candidate list ( device-side count, no host sync ).
// Each block handles one batch of up to n_bat cells: MLP
// computed cooperatively over ( cell, neuron ) items,
// weights read once per block from global, trunk shared.

template< class sol_T, class bdt_T, class eos_T >
__global__ void ker_nn
( const sol_T sol, const bdt_T bdt, const eos_T eos,
  bat_t bat )
{
    return sol.nn_bat( bdt, eos, bat );
};

//////////////////////////////////////////////////////////
// Solver

template
< class cor_T = ::chemistry::solver::corrector :: base_t,
  class dat_T = extension::chemistry::dat_t< type::float_t >,
  class derived_T = crtp :: dummy_t >
struct solver_t : public extension::chemistry::solver_t
      < cor_T, dat_T, solver_t< cor_T, dat_T, derived_T > >
{
    ////////// Types //////////
    __crtp_def_self__( solver_t, derived_T, cor_T, dat_T );
    using super_t = extension::chemistry::solver_t
        < cor_T, dat_T, this_t >;
    using base_t = ::chemistry::solver::base_t;

    ////////// Data ( kernel-visible: keep POD ) //////////
    bool    use_dnn;
    int     n_stiff;    // substeps needed to count as stiff
    bool    nn_checked; // n_hv validated against the model
    bool    dnn_print;  // per-block debug prints
    bool    dnn_force;  // bypass T/dt/stiffness gate ( all NN )
    bool    no_fb;      // no DI fallback on sanity violation
    float   sum_tol;    // | sum X - 1 | tolerance
    float   sum_fb;     // | sum X - 1 | fallback threshold
                        // ( renormalize-commit below this )
    int     nn_pad;     // alignment slack before scratch
    int     nn_sh_off;  // scratch byte offset in var_sh
    int     probe_try;  // DI attempts before routing to NN
    int     probe_iter; // probe substep budget ( 0 = full iter_max )
    float   dTlg_cap;   // per-step | d lgT | guard ( dex )
    float   T_route_hi; // hot-cell router: T above this stays on
                        // DI ( < 0 = off, use nn.T_hi ) -- NN-
                        // -committed hot states drift OOD and
                        // the cleanup di_step on them collapses
                        // ( h -> 0, 1e5 substeps, never conver-
                        // ges ), so hot cells must never leave
                        // the DI trajectory
    int     n_bat;      // cells per NN batch
    int     nn_th;      // threads for the NN kernel
    int     dnn_dump;   // DAgger pair dumping ( 0 / 1 ):
                        // every NN-registered cell gets a
                        // shadow FULL-budget DI solve from the
                        // pristine state ( label ), while the
                        // NN commit stays the real transition.
                        // Offline-only: costs one full DI solve
                        // per registration.
    int     dump_cap;   // buffer capacity ( pairs )
    float * dump_buf;   // device rows [ n_in + n_out + 4 ]:
                        // in, out, dt, cell, kind, conv
    int   * dump_cnt;   // device pair counter ( accumulates
                        // across cycles; written at free( ) )
    unsigned long long * cnt;
                        // device event counters [ 16 ] ( see
                        // the slot table at the top of this
                        // file ); production observability
                        // without device printf
    int     grind_cap;  // anti-grind ( rc-40 ): after this many
                        // budget-exhausted DI solves for a cell
                        // ( each ~iter_max substeps burned ),
                        // ( a ) its kernel-B guard trips
                        // RN-commit instead of falling back --
                        // the cell stays on the NN trajectory;
                        // ( b ) kernel-A full-budget DI solves
                        // drop to the probe_iter budget (
                        // bounded partial commits ).  A
                        // converged solve resets the counter.
                        // 0 = off ( v1 semantics ).  Default 2.
    int   * grind;      // device per-cell budget-exhaustion
                        // counters [ n_cell ]
    nn_t    nn;

    solver_t( )
        : use_dnn( false ), n_stiff( 4 ),
          nn_checked( false ),
          dnn_print( false ), dnn_force( false ),
           no_fb( false ), sum_tol( 0.02f ),
           sum_fb( -1.f ),
          nn_pad( 0 ), nn_sh_off( 0 ),
          probe_try( 1 ), probe_iter( 0 ), dTlg_cap( 1.f ),
          T_route_hi( -1.f ),
          n_bat( 8 ), nn_th( 128 ),
          dnn_dump( 0 ), dump_cap( 100000 ),
          dump_buf( nullptr ), dump_cnt( nullptr ),
          cnt( nullptr ), grind_cap( 2 ),
          grind( nullptr ) { return; };

    ////////// Host-side interface //////////
    __host__ virtual void init
    ( const input & args, ::chemistry::base_t & mod ) override
    {
        // Full solver init: kind ( kr / ross ), expl_lim,
        // base_t::init ( iter_max, tolerances, n_th,
        // cache_rate ), and corr.init ( cooling / dust /
        // T_eq ). Calling base_t::init directly skipped
        // this chain and left the corrector uninitialized
        // ( runaway heating, wrong integrator ).
        super_t::init( args, mod );
        // work_glob is supported: the grid-stride loop in
        // operator(  ) moves the per-cell workspace to the
        // global slab ( reserved tail at nn_sh_off is now
        // unused scratch — kept for size_sh stability ), so
        // large networks run the same code path as shared
        // mode.
        use_dnn = args.get< bool >
            ( "chemistry", "use_dnn", false );
        n_stiff = args.get< int >
            ( "chemistry", "dnn_N_stiff", 4 );
        dnn_print = args.get< bool >
            ( "chemistry", "dnn_print", false );
        dnn_force = args.get< bool >
            ( "chemistry", "dnn_force", false );
        no_fb = args.get< bool >
            ( "chemistry", "dnn_no_fallback", false );
        sum_tol = float( args.get< type::float_t >
            ( "chemistry", "dnn_sum_tol", 0.02 ) );
        sum_fb = float( args.get< type::float_t >
            ( "chemistry", "dnn_sum_fb", -1.0 ) );
        if( sum_fb < 0.f )
            sum_fb = sum_tol;   // v1 semantics: always fallback
        probe_try = args.get< int >
            ( "chemistry", "dnn_probe_try", 1 );
        probe_iter = args.get< int >
            ( "chemistry", "dnn_probe_iter", 0 );
        dTlg_cap = float( args.get< type::float_t >
            ( "chemistry", "dnn_dTlg_cap", 1.0 ) );
        T_route_hi = float( args.get< type::float_t >
            ( "chemistry", "dnn_T_route_hi", -1.0 ) );
        grind_cap = args.get< int >
            ( "chemistry", "dnn_grind_cap", 2 );
        n_bat = args.get< int >
            ( "chemistry", "dnn_batch", 8 );
        nn_th = args.get< int >
            ( "chemistry", "dnn_threads", 128 );
        dnn_dump = args.get< int >
            ( "chemistry", "dnn_dump", 0 );
        dump_cap = args.get< int >
            ( "chemistry", "dnn_dump_cap", 100000 );
        if( n_bat < 1 )
            n_bat = 1;
        if( nn_th < 32 )
            nn_th = 32;
        const auto dt_ovr = args.get< type::float_t >
            ( "chemistry", "dnn_dt_max", -1 );
        if( use_dnn )
        {
            const auto path = args.get< std::string >
                ( "chemistry", "dnn_file", "" );
            if( path.empty( ) )
                throw std::runtime_error
                    ( "[deeponet] use_dnn set but "
                      "dnn_file missing" );
            nn_load( path, *( mod.p_dev ),
                     mod.species, nn );
            if( nn.n_spe + 1 != this->dim )
                throw std::runtime_error
                    ( "[deeponet] model species count does "
                      "not match the solver dimension" );
            if( dt_ovr > 0 )
                nn.dt_max = float( dt_ovr );
            // Keep the batched kernel's shared memory under
            // the opt-in ceiling ( ~160 KB on sm_80 ).
            while( this->n_bat > 1
                   && this->nn_smem( ) > 160 * 1024 )
                this->n_bat >>= 1;
        };
        return;
    };

    __host__ virtual void free
    ( device::base_t & dev ) override
    {
        // DAgger dump: drain the device buffer to
        // dnn_dump.bin ( CWD = the run directory ) before
        // anything is torn down. Header: 6 ints ( magic,
        // n_pair, n_in, n_out, n_spe, n_hv ); rows: float32
        // [ in( n_in ), out( n_out ), dt, cell, kind, conv ],
        // all in the RAW dataset convention ( eps cgs,
        // rho code units, X renormalized, flx raw bands ).
        if( this->dnn_dump && this->dump_buf != nullptr )
        {
            int n_pair( 0 );
            dev.cp( & n_pair, this->dump_cnt, 1 );
            if( n_pair > this->dump_cap )
                n_pair = this->dump_cap;
            const size_t stride = size_t( this->nn.n_in )
                                  + this->nn.n_out + 4;
            std::vector< float > host
                ( size_t( n_pair ) * stride, 0.f );
            if( n_pair > 0 )
                dev.cp( host.data( ), this->dump_buf,
                        host.size( ) );
            std::FILE * f = std::fopen
                ( "dnn_dump.bin", "wb" );
            if( f == nullptr )
                std::printf
                    ( "[deeponet] dnn_dump: FAILED to open "
                      "dnn_dump.bin for writing\n" );
            else
            {
                const int hdr[ 6 ]
                    { 0x6e6e6470, n_pair, this->nn.n_in,
                      this->nn.n_out, this->nn.n_spe,
                      this->nn.n_hv };
                std::fwrite( hdr, sizeof( int ), 6, f );
                if( ! host.empty( ) )
                    std::fwrite( host.data( ),
                                 sizeof( float ),
                                 host.size( ), f );
                std::fclose( f );
                std::printf
                    ( "[deeponet] dnn_dump: wrote %d pairs "
                      "to dnn_dump.bin\n", n_pair );
            };
            dev.free_device( this->dump_buf );
            dev.free_device( this->dump_cnt );
            this->dump_buf = nullptr;
            this->dump_cnt = nullptr;
        };
        // Event-counter summary ( production observability,
        // host-side print only -- no device printf anywhere
        // in the final build ).
        if( this->cnt != nullptr )
        {
            unsigned long long h[ 16 ] = { 0 };
            dev.cp( h, this->cnt, 16 );
            std::printf
                ( "[deeponet] totals: probe_ok=%llu packed=%llu "
                  "nn=%llu rn=%llu fb=%llu "
                  "guards(k1..k4)=%llu/%llu/%llu/%llu "
                  "grind_rn=%llu fb_failed=%llu "
                  "di_cells=%llu di_sub=%llu di_lu=%llu\n",
                  h[ 11 ], h[ 3 ], h[ 4 ], h[ 5 ], h[ 10 ],
                  h[ 6 ], h[ 7 ], h[ 8 ], h[ 9 ],
                  h[ 12 ], h[ 13 ],
                  h[ 0 ], h[ 1 ], h[ 2 ] );
            dev.free_device( this->cnt );
            this->cnt = nullptr;
        };
        if( this->grind != nullptr )
        {
            dev.free_device( this->grind );
            this->grind = nullptr;
        };
        nn_free( dev, this->nn );
        auto & reg = stp_registry( );
        for( auto & kv : reg )
            if( kv.second != nullptr )
                dev.free_device( kv.second );
        reg.clear( );
        auto & breg = bat_registry( );
        for( auto & kv : breg )
            bat_free( dev, kv.second );
        breg.clear( );
        return;
    };

    // Solver-owned shared bytes: the extension region
    // ( dat + flx ), then alignment slack ( nn_pad, set
    // by integrate ), then the network scratch.
    __host__ virtual int size_sh( ) const override
    {
        auto res = utils::align( super_t::size_sh( ), 8 );
        // Always reserve pad + flags ( 8 B ) + NN scratch:
        // the kernel's gate writes flags[ 0 ] unconditionally,
        // so use_dnn = 0 must still land in-bounds. ( An
        // earlier version gated this on use_dnn, writing
        // past the dynamic-smem allocation in pure-DI runs
        // and corrupting neighboring blocks' workspaces. )
        res += this->nn_pad + utils::align
            ( nn.n_scr_f * 4 + 8, 8 );
        return res;
    };

    // One-time dynamic-smem opt-in per KERNEL ( the three
    // launches use different smem sizes; a single done-flag
    // would skip the batched kernel's opt-in and make its
    // > 48 KB launch fail with an invalid-argument error ).
    template< class K > __host__ void opt_smem
    ( K k, const int & s_sh )
    {
        if( s_sh <= 48 * 1024 )
            return;
        static std::set< const void * > reg;
        if( reg.count( ( const void * ) k ) )
            return;
        reg.insert( ( const void * ) k );
#if defined( __CUDACC__ )
        cudaFuncSetAttribute( ( const void * ) k,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            s_sh );
#elif defined( __HIPCC__ ) && ! defined( __HIP_CPU_RT__ )
        hipFuncSetAttribute( ( const void * ) k,
            HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            s_sh );
#endif
        return;
    };

    // Lazy per-block stiffness counters.
    template< class mod_T > __host__ int * stp_lookup
    ( mod_T & mod, mesh::block::dual_t & d )
    {
        auto & reg = stp_registry( );
        const auto itr = reg.find( & d );
        if( itr != reg.end( ) )
            return itr->second;
        const auto & b( d.b_w.get( ) );
        int n = 1;
        for( int a = 0; a < 3; ++ a )
            n *= b.geo.n_ceff[ a ];
        auto * stp = mod.p_dev->template malloc_device< int >( n );
        mod.p_dev->f_mset( stp, 0, n * sizeof( int ), 0 );
        reg.emplace( & d, stp );
        return stp;
    };

    template < class prx_T, class mod_T,
                class rls_T, class eos_T >
    __host__ void integrate
    ( mod_T & mod, mesh::block::dual_t & d,
      const rls_T & rls, const eos_T & eos )
    {
        this->n_hv = rls.n_hv;
        if( this->use_dnn && ! this->nn_checked )
        {
            this->nn_checked = true;
            if( nn.n_hv != this->n_hv )
                throw std::runtime_error
                    ( "[deeponet] radiation band count "
                      "mismatch: model has "
                      + std::to_string( nn.n_hv )
                      + ", run has "
                      + std::to_string( this->n_hv ) );
            if( ! nn.ready )
                throw std::runtime_error
                    ( "[deeponet] use_dnn but no model loaded" );
        };
        // Scratch offset: after the rls, eos, and extension
        // solver regions, 8-aligned. nn_pad keeps size_sh
        // consistent with this offset.
        this->nn_pad = ( 8 - ( rls.size_sh( )
            + eos.size_sh( ) + super_t::size_sh( ) ) % 8 ) % 8;
        this->nn_sh_off = rls.size_sh( ) + eos.size_sh( )
            + super_t::size_sh( ) + this->nn_pad;

        const auto & bdt( prx_T::d( d.d( ) ) );
        rls.prep_host( * this, bdt, eos );
        const auto rsc = this->resource
            ( d, rls.size_sh( ) + eos.size_sh( ) );

        using bdt_t = std::decay_t
            < decltype( prx_T::d( d.d( ) ) ) >;
        using wg_t = ::chemistry::solver::work_glob_t;
        const wg_t::wdt_t work
            ( wg_t::arg( this->p_work_glob, bdt. reg ) );
        this->template opt_smem
            ( ker_di< this_t, bdt_t, rls_T, eos_T,
                      wg_t::wdt_t >,
              std::get< 2 >( rsc ) );
        this->template opt_smem
            ( ker_di_cleanup< this_t, bdt_t, rls_T,
                              eos_T, wg_t::wdt_t >,
              std::get< 2 >( rsc ) );
        this->template opt_smem
            ( ker_nn< this_t, bdt_t, eos_T >,
              this->nn_smem( ) );

        auto * stp = this->stp_lookup( mod, d );
        auto & bat = bat_lookup( mod, d, this->nn.n_in );
        // DAgger dump buffer: one global allocation for the
        // whole run ( pairs accumulate across cycles and
        // blocks ); the counter is zeroed ONCE here, not per
        // cycle. freed + written to dnn_dump.bin in free( ).
        if( this->use_dnn && this->dnn_dump
                          && this->dump_buf == nullptr )
        {
            const size_t stride = size_t( this->nn.n_in )
                                  + this->nn.n_out + 4;
            this->dump_buf = mod.p_dev->template
                malloc_device< float >
                ( size_t( this->dump_cap ) * stride );
            this->dump_cnt = mod.p_dev->template
                malloc_device< int >( 1 );
            mod.p_dev->f_mset( this->dump_cnt, 0,
                               sizeof( int ), 0 );
            if( this->dnn_print )
                std::printf
                    ( "[deeponet] dnn_dump on: cap %d "
                      "pairs, row %zu floats\n",
                      this->dump_cap, stride );
        };
        // Event counters: one global allocation for the whole
        // run ( accumulate across cycles; zeroed ONCE here );
        // host-read in free( ).
        if( this->use_dnn && this->cnt == nullptr )
        {
            this->cnt = mod.p_dev->template
                malloc_device< unsigned long long >( 16 );
            mod.p_dev->f_mset( this->cnt, 0,
                               16 * sizeof( unsigned long long ),
                               0 );
        };
        // Anti-grind per-cell failed-cleanup counters ( rc-40 ):
        // one global allocation for the whole run.
        if( this->use_dnn && this->grind == nullptr )
        {
            const size_t n_cell =
                size_t( bdt.geo.n_ceff[ 0 ] )
                * bdt.geo.n_ceff[ 1 ] * bdt.geo.n_ceff[ 2 ];
            this->grind = mod.p_dev->template
                malloc_device< int >( n_cell );
            mod.p_dev->f_mset( this->grind, 0,
                               n_cell * sizeof( int ), 0 );
        };
        // Reset the candidate count on the device
        // ( the probe kernel refills it this step ).
        mod.p_dev->f_mset( bat.cand, 0, sizeof( int ),
                           0 );
        mod.p_dev->launch
            ( ker_di< this_t, bdt_t, rls_T, eos_T,
                      wg_t::wdt_t >,
              rsc, d.stream, * this, bdt, rls, eos,
              stp, bat, 0, work );
        if( use_dnn )
        {
            // Batched NN: persistent grid, device-side
            // candidate count ( no host sync ). Empty
            // candidate lists exit in microseconds.
            const int n_ct( bat.n_ct );
            int n_bk( ( n_ct + this->n_bat - 1 )
                      / this->n_bat );
            if( n_bk < 1 )
                n_bk = 1;
            mod.p_dev->launch
                ( ker_nn< this_t, bdt_t, eos_T >,
                  dim3( n_bk ), dim3( this->nn_th ),
                  this->nn_smem( ), d.stream, * this,
                  bdt, eos, bat );
            // Cleanup: rerun DI for guard-tripped cells.
            mod.p_dev->launch
                ( ker_di_cleanup< this_t, bdt_t, rls_T,
                                  eos_T, wg_t::wdt_t >,
                  rsc, d.stream, * this, bdt, rls, eos,
                  stp, bat, 1, work );
        };
        return;
    };

    // Shared-memory bytes for the batched-NN kernel.
    __host__ int nn_smem( ) const
    {
        if( ! this->nn.ready )
            return 0;
        const auto & nn( this->nn );
        const int nb( this->n_bat );
        size_t f( 0 );
        f += 2 * size_t( nb ) * nn.n_hdn;   // act ping-pong
        f += size_t( nb ) * nn.n_out;        // zout
        f += size_t( nb ) * nn.n_out
             * nn.n_basis;                   // Bbuf
        f += nn.n_basis;                     // Tk
        f += size_t( nb ) * nn.n_in;         // x_in
        f += size_t( nb );                   // ids
        return utils::align( int( 4 * f ), 8 );
    };

    ////////// Device-side interface //////////

    // The stock direct-integration path, verbatim from
    // extension::chemistry::solver_t::operator() ( d is
    // already loaded and prepped ), plus the stiffness
    // counter update.
    template< class bdt_T, class rls_T, class eos_T >
    __forceinline__ __device__ void di_step
    ( dat_T & d, const bdt_T & bdt, const rls_T & rls,
      const eos_T & eos, int * stp, const int & i_cell,
      const int & iter_cap = -1 )
        const
    {
        // iter_cap > 0 replaces iter_max ( anti-grind: a cell
        // whose full-budget solves keep exhausting drops to a
        // bounded budget -- the budget-exhaustion partial
        // commit below is the stock semantics, just cheaper ).
        const int it_max =
            ( iter_cap > 0 && iter_cap < this->iter_max )
                ? iter_cap : this->iter_max;
        for( d.iter = it_max; d.iter > 0; )
            if( this->solve( d, rls, bdt, eos ) )
                break;
        if( threadIdx.x == 0 )
        {
            stp[ i_cell ] = it_max - d.iter;
            const int sub = it_max - d.iter;
            if( this->cnt != nullptr )
            {
                atomicAdd( this->cnt + 0, 1ull );
                atomicAdd( this->cnt + 1,
                           ( unsigned long long ) sub );
                atomicAdd( this->cnt + 2,
                           ( unsigned long long )
                           ( d.lu_fail > 0 ? d.lu_fail : 0 ) );
            };
            // Grinder watch: any di_step that consumed a lot
            // of substeps prints regardless of the block /
            // cell sampling gates ( the slow-cycle regression
            // hunt -- probes are capped, so only di_step can
            // burn 1e4+ substeps ).  Diagnostic builds only.
#ifdef DNN_KERNEL_PRINTF
            if( this->dnn_print
               && ( sub > 1000 || blockIdx.x % 16 == 0 ) )
                printf( "[dnn] DI blk%d c%d iter_max=%d "
                        "sub=%d lu=%d t=%g h=%g Tout=%g\n",
                        blockIdx.x, i_cell, it_max, sub,
                        d.lu_fail,
                        (double)(float)d.t,
                        (double)(float)d.h,
                        (double)(float)d.T );
#endif
        }
        if( d.t > 0 )
            this->corr( d, bdt, eos, true );
        eos.temp( d );
        __syncthreads( );
        if( threadIdx.x == 0 )
            bdt.T_hyd.at( d.idx )[ 0 ] = d.T;
        return d.save( bdt, eos );
    };
    // Kernel A body: DI + probe. pass = 0 normal ( probe
    // packs candidates into the queue ), pass = 1 cleanup
    // ( walks the QUEUE: only guard-tripped slots re-run
    // DI, so no full grid scan and no stale-status hazard ).
    template< class bdt_T, class rls_T, class eos_T,
              class spc_T >
    __forceinline__ __device__ void operator()
    ( const bdt_T & bdt, const rls_T & rls,
      const eos_T & eos, int * stp,
      const bat_t & bat, const int pass,
      const spc_T & work ) const
    {
        __dyn_shared__( char, var_sh );
        const int n_c[ 3 ]
        {
            bdt. geo. n_ceff[ 0 ], bdt. geo. n_ceff[ 1 ],
            bdt. geo. n_ceff[ 2 ]
        };
        const int n_ct( n_c[ 0 ] * n_c[ 1 ] * n_c[ 2 ] );
        const int n_bs( gridDim. x * gridDim. y
                                  * gridDim. z );
        const int i_bs( blockIdx. x + gridDim. x *
            ( blockIdx. y + gridDim. y * blockIdx. z ) );

        // --- Cleanup pass: grid-stride over queue slots.
        if( pass == 1 )
        {
            const int n_q( bat. cand[ 0 ] );
            for( int s( i_bs ); s < n_q; s += n_bs )
            {
                if( bat. ok[ s ] != 2 )
                    continue;
                if( threadIdx.x == 0 )
                    bat. ok[ s ] = 0;
                __syncthreads( );
                const int c( bat. cand[ 1 + s ] );
                type :: idx_t idx_c;
                idx_c[ 0 ] =   c              %  n_c[ 0 ];
                idx_c[ 1 ] = ( c / n_c[ 0 ] ) %  n_c[ 1 ];
                idx_c[ 2 ] =   c / ( n_c[ 0 ] * n_c[ 1 ] );
                char * base( var_sh );
                if( work )
                    base = ( char * )
                        work. at( idx_c );
                dat_T d( this-> dim,  base, rls,
                         this-> cache_rate );
                d.  idx = idx_c;
                d  .load      ( bdt,          eos );
                rls.prep_dev  ( d  ,  bdt,    eos );
                // DAgger ( kind 1 ): the fallback cleanup's
                // own solve IS the label. Input captured
                // BEFORE di_step -- u is still pristine (
                // the guard-tripped slot never committed ),
                // and d.load's floor repairs have run.
                float fin[ 160 ];
                const bool dump_c( this->dnn_dump
                                   && this->dump_buf
                                      != nullptr );
                if( dump_c && threadIdx.x == 0
                          && this->nn.n_in <= 160 )
                    this->dump_in( bdt, eos, c, fin );
                this->di_step ( d, bdt, rls, eos, stp, c );
                // Anti-grind accounting ( rc-40 ): a cleanup
                // solve that exhausted its budget or hit an LU
                // failure increments this cell's grind counter.
                // Past dnn_grind_cap the kernel-B guard path
                // RN-commits instead of falling back again.
                if( this->grind != nullptr
                    && threadIdx.x == 0
                    && ( d.iter <= 0 || d.lu_fail > 0 ) )
                {
                    atomicAdd( this->grind + c, 1 );
                    if( this->cnt != nullptr )
                        atomicAdd( this->cnt + 13, 1ull );
                };
                if( dump_c )
                {
                    __syncthreads( );
                    if( threadIdx.x == 0
                     && this->nn.n_in  <= 160
                     && this->nn.n_out <= 160 )
                        this->dump_lbl
                            ( d, bdt, c, fin, 1 );
                };
                __syncthreads( );
            };
            return;
        };

        for( int c( i_bs ); c < n_ct; c += n_bs )
        {
            type :: idx_t idx_c;
            idx_c[ 0 ] =   c              %  n_c[ 0 ];
            idx_c[ 1 ] = ( c / n_c[ 0 ] ) %  n_c[ 1 ];
            idx_c[ 2 ] =   c / ( n_c[ 0 ] * n_c[ 1 ] );
            char * base( var_sh );
            if( work )
                base = ( char * ) work. at( idx_c );

            dat_T d( this-> dim,  base, rls,
                     this-> cache_rate );
            d.  idx = idx_c;
            d  .load      ( bdt,          eos );
            rls.prep_dev  ( d  ,  bdt,    eos );

            // --- Probe: in-envelope cells try a few DI
            // --- iterations. Completing -> the cheap
            // --- non-stiff commit ( exactly the stock
            // --- flow, substeps already spent ); still
            // --- needing more iterations after the probe
            // --- budget -> stiff: register the cell in
            // --- the NN queue. The failed probe leaves d
            // --- ( shared scratch ) dirty but that is
            // --- simply discarded: kernel B reads the
            // --- PRISTINE cell state from global u /
            // --- fluxes ( commit paths are the only
            // --- writers ), so no restore is needed.
            if( this->use_dnn && pass == 0
                             && this->nn. ready )
            {
                bool want_nn = false;
                if( this->dnn_force )
                    want_nn = true;
                else
                {
                    const float T = float( d.T );
                    const float dt = float( bdt.dt_0 );
                    // Hot-cell router: cap the gate's upper
                    // bound at T_route_hi so hot cells never
                    // register for the NN ( they stay on the
                    // exact-DI trajectory, where they are cheap
                    // -- 1 substep -- and where the NN's small
                    // per-step drift cannot accumulate into a
                    // DI-unintegrable state ).
                    const float T_hi_eff =
                        ( this->T_route_hi > 0.f
                       && this->T_route_hi < this->nn.T_hi )
                            ? this->T_route_hi
                            : this->nn.T_hi;
                    if( T >= this->nn.T_lo && T <= T_hi_eff
                     && dt >= this->nn.dt_min
                     && dt <= this->nn.dt_max
                     && ( this->n_stiff <= 0
                       || stp[ c ] >= this->n_stiff ) )
                    {
                        int  tried = 0;
                        bool done  = false;
                        // Probe substep budget: the probe is a
                        // stiffness DETECTOR, not a solver -- a
                        // cell needing more than probe_iter
                        // substeps is stiff by definition and
                        // routes to the NN long before the full
                        // iter_max ( which a failing probe would
                        // otherwise grind through EVERY cycle ).
                        const int p_iter = this->probe_iter > 0
                            ? ( this->probe_iter < this->iter_max
                                  ? this->probe_iter
                                  : this->iter_max )
                            : this->iter_max;
                        for( d.iter = p_iter; d.iter > 0; )
                        {
                            if( this->solve( d, rls, bdt,
                                             eos ) )
                            {
                                done = true;
                                break;
                            };
                            if( ++ tried
                                >= this->probe_try )
                                break;
                        };
                        if( done || d.t <= 0 )
                        {
                            // Completed within the probe
                            // budget: commit via the stock
                            // tail ( di_step's loop already
                            // ran as the probe ).
                            if( threadIdx.x == 0 )
                            {
                                stp[ c ] = p_iter - d.iter;
                                if( this->cnt != nullptr )
                                    atomicAdd( this->cnt + 11,
                                               1ull );
                            };
                            if( d.t > 0 )
                                this->corr( d, bdt, eos,
                                            true );
                            eos.temp( d );
                            __syncthreads( );
                            if( threadIdx.x == 0 )
                                bdt.T_hyd.at( d.idx )
                                    [ 0 ] = d.T;
                            d.save( bdt, eos );
                            __syncthreads( );
                            continue;
                        };
                        // Stiff: register for the NN.
                        want_nn = true;
                    };
                };

                if( want_nn )
                {
                    // --- DAgger shadow label ( dnn_dump ):
                    // --- full-budget DI from the PRISTINE
                    // --- state ( reload after the failed
                    // --- probe ), through the exact di_step
                    // --- pipeline ( solve loop + corrector +
                    // --- eos.temp ) but WITHOUT the commit --
                    // --- the NN still owns the transition (
                    // --- kernel B ), the pair records what
                    // --- full DI would have done from the
                    // --- same state.
                    if( this->dnn_dump
                        && this->dump_buf != nullptr )
                        this->dump_shadow
                            ( d, bdt, rls, eos, c );
                    this->nn_pack( d, bat, c, stp );
                    __syncthreads( );
                    continue;
                };
            };

            // Gate-ineligible cells ( hot-routed / dt / stiff
            // counter ) take the full-budget DI path -- the
            // grind site ( rc-40 ): a cell whose state drifted
            // while NN-eligible and then crossed the router
            // boundary arrives here off-trajectory, h collapses
            // and the solve burns the whole iter_max budget
            // EVERY cycle.  Anti-grind: after grind_cap budget
            // exhaustions the budget drops to probe_iter (
            // bounded partial commits -- the stock
            // budget-exhaustion semantics, just cheap ); a
            // converged solve resets the counter.
            int cap_i( -1 );
            if( this->grind != nullptr && this->grind_cap > 0
                && this->grind[ c ] >= this->grind_cap )
                cap_i = ( this->probe_iter > 0
                            ? this->probe_iter : 1000 );
            this->di_step( d, bdt, rls, eos, stp, c, cap_i );
            if( this->grind != nullptr && threadIdx.x == 0 )
            {
                if( d.iter <= 0 )
                {
                    atomicAdd( this->grind + c, 1 );
                    if( this->cnt != nullptr )
                        atomicAdd( this->cnt + 13, 1ull );
                }
                else if( this->grind[ c ] > 0 )
                    this->grind[ c ] = 0;
            };
            __syncthreads( );
        };
        return;
    };

    // DAgger pair dumping ( dnn_dump ): one buffer row per
    // NN-registered cell = [ in( n_in ), out( n_out ), dt,
    // cell, kind, conv ] float32s, RAW dataset convention
    // ( eps cgs, rho code units, X renormalized number
    // fractions, flx raw bands; label X' renormalized,
    // eps' cgs ). kind 0 = shadow label ( NN commit won ),
    // kind 1 = fallback cleanup ( DI commit won ). conv = 1
    // if the DI solve finished within iter_max.
    __forceinline__ __device__ void dump_pair
    ( const float * fin, const float * lbl,
      const float & dt, const int & c, const int & kind,
      const int & conv ) const
    {
        const int slot = atomicAdd( this->dump_cnt, 1 );
        if( slot >= this->dump_cap )
            return;
        const auto & nn( this->nn );
        const int n_in( nn.n_in ), n_ot( nn.n_out );
        float * row = this->dump_buf
            + size_t( slot ) * ( n_in + n_ot + 4 );
        for( int k = 0; k < n_in; ++ k )
            row[ k ] = fin[ k ];
        for( int k = 0; k < n_ot; ++ k )
            row[ n_in + k ] = lbl[ k ];
        row[ n_in + n_ot + 0 ] = dt;
        row[ n_in + n_ot + 1 ] = float( c );
        row[ n_in + n_ot + 2 ] = float( kind );
        row[ n_in + n_ot + 3 ] = float( conv );
        return;
    };

    // Raw input features from the pristine ( floor-repaired
    // ) cell state -- thread 0, identical recipe to kernel
    // B's z-scored build ( before normalization ).
    template< class bdt_T, class eos_T >
    __forceinline__ __device__ void dump_in
    ( const bdt_T & bdt, const eos_T & eos, const int & c,
      float * fin ) const
    {
        const auto & nn( this->nn );
        type :: idx_t idx;
        idx[ 0 ] =   c % bdt. geo. n_ceff[ 0 ];
        idx[ 1 ] = ( c / bdt. geo. n_ceff[ 0 ] )
                   % bdt. geo. n_ceff[ 1 ];
        idx[ 2 ] =   c / ( bdt. geo. n_ceff[ 0 ]
                          * bdt. geo. n_ceff[ 1 ] );
        auto * u = bdt.hyd.u.at( idx );
        const float e_cgs = float( eos.e_g_cgs( u ) );
        const float rho   = float( u[ idx::rho ] );
        fin[ 0 ] = e_cgs / fmaxf( rho
            * float( eos.unit.rho0 ), 1e-30f );
        fin[ 1 ] = rho;
        float sum = 0.f;
        for( int i = 0; i < nn.n_spe; ++ i )
        {
            fin[ 2 + i ] = float( u[ eos.n_hyd + i ] );
            sum += fin[ 2 + i ];
        };
        const float inv = 1.f / fmaxf( sum, 1e-30f );
        for( int i = 0; i < nn.n_spe; ++ i )
            fin[ 2 + i ] *= inv;
        const auto * flx = bdt.rad.flx.at( idx );
        for( int k = 0; k < nn.n_hv; ++ k )
            fin[ 2 + nn.n_spe + k ] = float( flx[ k ] );
        return;
    };

    // Label from the post-solve dat_t ( X' renormalized,
    // eps' cgs ) + buffer write -- thread 0.
    template< class bdt_T >
    __forceinline__ __device__ void dump_lbl
    ( const dat_T & d, const bdt_T & bdt, const int & c,
      const float * fin, const int & kind ) const
    {
        const auto & nn( this->nn );
        float lbl[ 160 ];
        float s2 = 0.f;
        for( int i = 0; i < nn.n_spe; ++ i )
        {
            lbl[ i ] = float( d.y[ i ] );
            s2 += lbl[ i ];
        };
        const float inv2 = 1.f / fmaxf( s2, 1e-30f );
        for( int i = 0; i < nn.n_spe; ++ i )
            lbl[ i ] *= inv2;
        lbl[ nn.n_spe ] = float( d.y[ d.i_en ] );
        this->dump_pair( fin, lbl, float( bdt.dt_0 ), c,
                         kind, ( d.iter > 0
                               && d.lu_fail == 0 ) ? 1 : 0 );
        return;
    };

    // Shadow solve: full-budget DI from the pristine state,
    // di_step pipeline minus the commit ( no save, no
    // T_hyd ). Runs on ALL threads ( the solve / corrector
    // machinery is block-cooperative over the shared dat_t );
    // only the dump write is thread 0. d is left dirty --
    // kernel B reads the pristine GLOBAL state, never d.
    template< class bdt_T, class rls_T, class eos_T >
    __forceinline__ __device__ void dump_shadow
    ( dat_T & d, const bdt_T & bdt, const rls_T & rls,
      const eos_T & eos, const int & c ) const
    {
        __syncthreads( );
        d.  load      ( bdt,          eos );
        rls.prep_dev  ( d  ,  bdt,    eos );
        for( d.iter = this->iter_max; d.iter > 0; )
            if( this->solve( d, rls, bdt, eos ) )
                break;
        if( d.t > 0 )
            this->corr( d, bdt, eos, true );
        eos.temp( d );
        __syncthreads( );
        if( threadIdx.x == 0
         && this->nn.n_in  <= 160
         && this->nn.n_out <= 160 )
        {
            float fin[ 160 ];
            this->dump_in( bdt, eos, c, fin );
            this->dump_lbl( d, bdt, c, fin, 0 );
#ifdef DNN_KERNEL_PRINTF
            if( this->dnn_print )
                printf( "[dnn] dump c%d kind0 sub=%d "
                        "lu=%d eps=%g\n", c,
                        this->iter_max - d.iter,
                        d.lu_fail,
                        (double)(float)d.y[ d.i_en ] );
#endif
        };
        __syncthreads( );
        return;
    };

    // Register one cell in the NN queue: atomicAdd on the
    // count assigns a compact SLOT; only the cell id, the
    // pre-step temperature, and the status are stored.
    // Kernel B reads the cell STATE ( u, fluxes ) directly
    // from global memory, which is pristine for registered
    // cells: the failed DI probe mutates only the shared
    // dat_t ( never u / T_hyd / flx ), and no commit has
    // run for this cell this step. T_old comes from d.T
    // ( post-prep_dev ): T_hyd cannot be used because it
    // holds garbage at t = 0 ( never written before the
    // first chemistry commit ).
    __forceinline__ __device__ void nn_pack
    ( dat_T & d, const bat_t & bat, const int & c,
      int * stp ) const
    {
        if( threadIdx.x == 0 )
        {
            const int slot = atomicAdd( bat.cand, 1 );
            bat.cand[ 1 + slot ] = c;
            bat.T_old[ slot ]    = float( d.T );
            bat.ok[ slot ]       = 0;
            // Mark stiff so the gate keeps this cell
            // probe-eligible next step ( the probe is the
            // real arbiter; this avoids NN <-> DI
            // oscillation on the stale counter ).
            if( stp != nullptr )
                stp[ c ] = this->iter_max;
            if( this->cnt != nullptr )
                atomicAdd( this->cnt + 3, 1ull );
#ifdef DNN_KERNEL_PRINTF
            if( this->dnn_print )
                printf( "[dnn] pack c%d -> slot %d "
                        "( T=%g )\n", c, slot,
                        (double)(float)d.T );
#endif
        };
        __syncthreads( );
        return;
    };

    // Kernel B body: batched MLP + commit. Each block owns
    // a slice of the candidate list ( persistent grid,
    // grid-stride over batches ). Shared memory holds:
    //   act  [ n_bat * n_w ]      activations ( ping-pong )
    //   zout [ n_bat * n_out ]    combined outputs
    //   Tk   [ n_basis ]          trunk output ( uniform dt )
    //   ids  [ n_bat ]            candidate cell ids
    // The trunk is identical for all cells ( uniform dt ),
    // computed once by thread 0.
    template< class bdt_T, class eos_T >
    __forceinline__ __device__ void nn_bat
    ( const bdt_T & bdt, const eos_T & eos,
      const bat_t & bat ) const
    {
        __dyn_shared__( char, var_sh );
        const auto & nn( this->nn );
        const int n_bat( this->n_bat );
        const int n_w( nn.n_hdn );
        const int n_it( nn.n_in );
        const int n_ot( nn.n_out );
        const int n_bs( nn.n_basis );

        float * act1 = ( float * ) var_sh;
        float * act2 = act1 + size_t( n_bat ) * n_w;
        float * zout = act2 + size_t( n_bat ) * n_w;
        float * Bbuf = zout + size_t( n_bat ) * n_ot;
        float * Tk   = Bbuf + size_t( n_bat ) * n_ot * n_bs;
        float * x_in = Tk + n_bs;      // [ n_bat * n_in ]
        int   * ids  = ( int * )
            ( x_in + size_t( n_bat ) * n_it );

        const int i_b( blockIdx.x + gridDim.x *
            ( blockIdx.y + gridDim.y * blockIdx.z ) );
        const int n_bk( gridDim.x * gridDim.y
                                  * gridDim.z );
        // Batch granularity: block i_b owns slots
        // [ i_b * n_bat, i_b * n_bat + n_bat ), the
        // persistent loop advances a full grid of batches
        // ( n_bk * n_bat slots ) per iteration. Striding
        // by n_bk alone would make adjacent blocks take
        // OVERLAPPING batches and commit slots twice.
        for( int b0( i_b * n_bat ); ; b0 += n_bk * n_bat )
        {
            // Device-side candidate count.
            const int n_c = bat.cand[ 0 ];
            if( b0 >= n_c )
                return;
            const int nb( n_bat < n_c - b0 ? n_bat : n_c - b0 );

            // --- Load batch cell ids; build the input rows
            // --- from the PRISTINE global cell state ( u
            // --- for eps / rho / species, flx for the
            // --- bands; index-only queue design ).
            for( int j = threadIdx.x; j < nb;
                 j += blockDim.x )
                ids[ j ] = bat.cand[ 1 + b0 + j ];
            __syncthreads( );
            {
                const int n_r( bdt. geo. n_ceff[ 0 ] );
                const int n_y( bdt. geo. n_ceff[ 1 ] );
                const float lam
                    = nn.w_dev[ nn.off_lam ];
                const float * mu_in = nn.w_dev
                    + nn.off_mu_in;
                const float * sd_in = nn.w_dev
                    + nn.off_sd_in;
                for( int j = threadIdx.x; j < nb;
                     j += blockDim.x )
                {
                    const int c( ids[ j ] );
                    type :: idx_t idx;
                    idx[ 0 ] =   c % n_r;
                    idx[ 1 ] = ( c / n_r ) % n_y;
                    idx[ 2 ] =   c / ( n_r * n_y );
                    // Non-const: e_g_cgs may floor rho/ene
                    // exactly as dat_t::load does for DI (
                    // the stock "repair" writes ), so both
                    // paths see identical post-load state.
                    auto * u = bdt.hyd.u.at( idx );
                    // e_g_cgs first: it may floor rho/ene in u (
                    // the stock repair, same as dat_t::load ), and
                    // the eps denominator must use the FLOORED rho
                    // exactly like dataset.py's _eps_cgs.
                    const float e_cgs = float
                        ( eos.e_g_cgs( u ) );
                    const float rho
                        = float( u[ idx::rho ] );
                    // Specific internal energy ( cgs erg/g )
                    // EXCLUDING kinetic, matching dat_t::load
                    // ( y[ i_en ] = e_g_cgs / rho_cgs ):
                    // the v2 model input convention.
                    const float eps = e_cgs
                        / fmaxf( rho
                            * float( eos.unit.rho0 ),
                            1e-30f );
                    // Species: X renormalized mass
                    // fractions ( training convention ).
                    float sum = 0.f;
                    for( int i = 0; i < nn.n_spe; ++ i )
                        sum += float
                            ( u[ eos.n_hyd + i ] );
                    const float inv = 1.f
                        / fmaxf( sum, 1e-30f );
                    const auto * flx
                        = bdt.rad.flx.at( idx );
                    float * xin = x_in
                        + size_t( j ) * n_it;
                    for( int k = 0; k < n_it; ++ k )
                    {
                        float v;
                        if( k == 0 )
                            v = log10f( fmaxf( eps,
                                               1e-30f ) );
                        else if( k == 1 )
                            v = log10f( fmaxf( rho,
                                               1e-30f ) );
                        else if( k < 2 + nn.n_spe )
                            v = nn_bc_fwd( float
                                ( u[ eos.n_hyd
                                  + k - 2 ] )
                                * inv, lam );
                        else
                            v = log10f( fmaxf( float
                                ( flx[ k - 2
                                      - nn.n_spe ] ),
                                1e-30f ) );
                        xin[ k ] = ( v - mu_in[ k ] )
                                   / sd_in[ k ];
                    };
                };
            };
            __syncthreads( );

            // --- Trunk ( thread 0; uniform dt -> identical
            // --- for every cell in the run ). Serial,
            // --- register-local: widths <= 64.
            const float mu_dt = nn.w_dev[ nn.off_mu_dt ];
            const float sd_dt = nn.w_dev[ nn.off_sd_dt ];
            if( threadIdx.x == 0 )
            {
                const float dt_c = float( bdt.dt_0 );
                const float zdt = ( log10f
                    ( fmaxf( dt_c, 1e-30f ) ) - mu_dt )
                    / sd_dt;
                float cur[ 64 ];
                float nxt[ 64 ];
                float src = zdt;
                // Layer 0: 1 -> h0.
                {
                    const auto & L( nn.tr_lay[ 0 ] );
                    const float * W = nn.w_dev + L.off_w;
                    const float * B = nn.w_dev + L.off_b;
                    for( int o = 0; o < L.n_out; ++ o )
                        cur[ o ] = nn_gelu( B[ o ]
                            + src * W[ o ] );
                };
                // Layers 1..: h -> h' ( last -> Tk ).
                for( int l = 1; l < nn.n_lay_tr; ++ l )
                {
                    const auto & L( nn.tr_lay[ l ] );
                    const float * W = nn.w_dev + L.off_w;
                    const float * B = nn.w_dev + L.off_b;
                    const bool last( l + 1
                                     == nn.n_lay_tr );
                    float * tgt( last ? Tk : nxt );
                    for( int o = 0; o < L.n_out; ++ o )
                    {
                        float acc = B[ o ];
                        for( int i = 0; i < L.n_in; ++ i )
                            acc += cur[ i ]
                                * W[ i * L.n_out + o ];
                        tgt[ o ] = last ? acc
                            : nn_gelu( acc );
                    };
                    if( ! last )
                        for( int o = 0; o < L.n_out; ++ o )
                            cur[ o ] = nxt[ o ];
                };
            };
            __syncthreads( );

            // --- Branch: batched layers into Bbuf.
            {
                const float * src = x_in;
                for( int l = 0; l < nn.n_lay_br; ++ l )
                {
                    const auto & L( nn.br_lay[ l ] );
                    const bool last( l + 1 == nn.n_lay_br );
                    float * dst = last ? Bbuf
                        : ( ( l % 2 ) ? act2 : act1 );
                    nn_layer_b( src, dst, nb, L.n_in,
                                L.n_out, nn.w_dev,
                                L.off_w, L.off_b, ! last );
                    src = dst;
                };
            };

            // --- Combine: zout[ j, z ] = bias + sum_k B T.
            {
                const float * bias_out
                    = nn.w_dev + nn.off_bias;
                const float * mu_out = nn.w_dev
                    + nn.off_mu_out;
                const float * sd_out = nn.w_dev
                    + nn.off_sd_out;
                const int n_j( nb * n_ot );
                for( int j( threadIdx.x ); j < n_j;
                     j += blockDim.x )
                {
                    const int cidx( j / n_ot );
                    const int z( j % n_ot );
                    float acc = bias_out[ z ];
                    const float * B = Bbuf + size_t( cidx )
                        * n_ot * n_bs + z * n_bs;
                    for( int k = 0; k < n_bs; ++ k )
                        acc += B[ k ] * Tk[ k ];
                    zout[ size_t( cidx ) * n_ot + z ]
                        = acc * sd_out[ z ] + mu_out[ z ];
                };
                __syncthreads( );
            };

            // --- Commit phase: decode + guards + write.
            this->nn_commit_b( bdt, eos, bat, zout, x_in,
                               ids, b0, nb );
            __syncthreads( );
        };
    };

    // T from ( eps, X ): the eos.temp fixed-point replicated
    // with a LOCAL y in dat convention — y_i = X_i ( number
    // fraction, i.e. n_i / sum(n) ; dat.h builds y_l =
    // u_src[ l ] / rho which is proportional to n_i for every
    // species since u_src = n_i * m_a / rho_0 shares the
    // common m_a / rho_0. mu_cgs_xn / basic_cv weight by
    // number fraction and normalize internally, so any
    // common scale is fine -- but the SHAPE must be n_i,
    // NOT n_i / m_i: dividing by mcgs turns mu_cgs_xn into
    // a harmonic mean of the particle mass ( dominated by
    // electrons in ionized cells, ~5% low even for H2/He )
    // -- root-cause 34. y[ i_en ] slot is ignored by the
    // eos loops. Both cv paths are supported: cv_poly
    // ( xn_d fixed-point ) when a cv_file exists, basic_cv
    // ( gm1i ) otherwise — the fid network has no cv_file,
    // so the basic path is the live one.
    // Clamps to [ Tmin, Tmax ] exactly as eos.temp.
    template< class eos_T >
    __forceinline__ __device__ float nn_eos_T
    ( const eos_T & eos, const float * y,
      const float & eps ) const
    {
        const float ee = eps
            * float( eos. mu_cgs_xn( y, nullptr ) )
            / float( phys::cgs::kb );
        float T( 0.f );
        if( eos. cv_dat )
        {
            T = ee / float( eos. cv_dat
                            . xn_d_simple( y ) );
            for( int it( 0 ); it < 20; ++ it )
            {
                const float Tn_( ee / float
                    ( eos. cv_dat. xn_d( y, T ) ) );
                if( fabsf( Tn_ - T )
                    / ( Tn_ + 1e-30f ) < 1e-3f )
                { T = Tn_; break; };
                T = Tn_;
            };
        }
        else
            T = ee / float( eos. basic_cv
                            ( y, eos. gm1i_d ) );
        if( T <= 0.f || T != T )
            T = ee * float( eos. gamma - 1.f );
        return fminf( fmaxf( T, float( eos.Tmin ) ),
                      float( eos.Tmax ) );
    };

    // eps from T ( the inverse of nn_eos_T ):
    // eps = T * cv * kb / mu_cgs_xn.
    template< class eos_T >
    __forceinline__ __device__ float nn_eos_eps
    ( const eos_T & eos, const float * y,
      const float & T ) const
    {
        float cv( 0.f );
        if( eos. cv_dat )
            cv = float( eos. cv_dat. xn_d( y, T ) );
        else
            cv = float( eos. basic_cv( y,
                                       eos. gm1i_d ) );
        return T * cv * float( phys::cgs::kb )
               / float( eos. mu_cgs_xn( y, nullptr ) );
    };

    // Commit one batch: decode ( delta or absolute ),
    // guards ( nan/sum/T-bounds/per-step dlgT cap ), write
    // species / energy / T_hyd directly. Energy follows the
    // per-cell path exactly: e' = e_g_cgs * ( T'/T_old ),
    // u[ ene ] += de with de in code units ( dat_t::save ).
    // Reads ( zout / x_in / T_old / ok ) are SLOT-indexed;
    // writes ( u / T_hyd ) go to the cell from ids[ j ].
    template< class bdt_T, class eos_T >
    __forceinline__ __device__ void nn_commit_b
    ( const bdt_T & bdt, const eos_T & eos,
      const bat_t & bat, const float * zout,
      const float * x_in, const int * ids,
      const int & b0, const int & nb ) const
    {
        const auto & nn( this->nn );
        const float lam = nn.w_dev[ nn.off_lam ];
        const int n_ot( nn.n_out );
        const int n_spe( nn.n_spe );

        for( int j = threadIdx.x; j < nb; j += blockDim.x )
        {
            const int c( ids[ j ] );
            type::idx_t idx;
            idx[ 0 ] =   c % bdt.geo.n_ceff[ 0 ];
            idx[ 1 ] = ( c / bdt.geo.n_ceff[ 0 ] )
                       % bdt.geo.n_ceff[ 1 ];
            idx[ 2 ] =   c / ( bdt.geo.n_ceff[ 0 ]
                              * bdt.geo.n_ceff[ 1 ] );

            // --- Decode.
            float X[ 128 ];
            if( n_spe > 128 )
                continue;   // guarded at load, defensive
            float eps_n = 0.f;
            {
                const float * z = zout + size_t( j )
                    * n_ot;
                const float * xin = x_in + size_t( j )
                    * nn.n_in;
                const float * sd_in = nn.w_dev
                    + nn.off_sd_in;
                const float * mu_in = nn.w_dev
                    + nn.off_mu_in;
                for( int i = 0; i < n_spe; ++ i )
                {
                    const float v = z[ i ];
                    if( nn.inc )
                    {
                        const int k = 2 + i;
                        const float bc_in = xin[ k ]
                            * sd_in[ k ] + mu_in[ k ];
                        X[ i ] = nn_bc_inv( bc_in + v,
                                            lam );
                    }
                    else
                        X[ i ] = nn_bc_inv( v, lam );
                };
                {
                    // v2: slot n_spe is the specific
                    // internal energy increment ( cgs ):
                    // eps' = eps_in * 10^dz.
                    const float v = z[ n_spe ];
                    if( nn.inc )
                    {
                        const float lge_in = xin[ 0 ]
                            * sd_in[ 0 ] + mu_in[ 0 ];
                        eps_n = powf( 10.f,
                                      fmaxf( lge_in
                                             + v, -300.f ) );
                    }
                    else
                        eps_n = powf( 10.f, v );
                };
            };

            // --- Guards. eps-based ( v2 ): the model output
            // --- is the specific internal energy; T bounds
            // --- are checked AFTER the eos inversion below
            // --- ( T from eps_hat, X_hat ), together with
            // --- the per-step dlgT cap.
            int bad = 0;
            float sum = 0.f;
            for( int i = 0; i < n_spe; ++ i )
            {
                const float x = X[ i ];
                if( x != x || x < 0.f || x > 1.5f )
                { bad = 1; break; };
                sum += x;
            };
            if( bad == 0
                 && fabsf( sum - 1.f ) > this->sum_tol )
            {
                // Kind 2 ( sumX deviation ): tiered response.
                // Below sum_fb the commit RENORMALIZES X-hat by
                // its own sum ( composition rescaled, eps / T
                // untouched ) -- zero cost, and it keeps the
                // ignition-transient speedup: one 3.2% sum
                // crossing during cycle 1 cost ~20 s of
                // cleanup grind when it fell back ( root-cause
                // 31 ). Above sum_fb the deviation signals a
                // degraded / OOD prediction ( the amplifying
                // cells' outputs rot as their eps drifts ) and
                // the exact re-solve is the re-anchor that
                // bounds the rollout drift -- so it falls
                // back. dnn_sum_fb defaults to sum_tol, which
                // reproduces the v1 always-fallback semantics.
                if( fabsf( sum - 1.f )
                    > this->sum_fb )
                    bad = 2;
                else
                {
                    if( this->cnt != nullptr )
                        atomicAdd( this->cnt + 5, 1ull );
#ifdef DNN_KERNEL_PRINTF
                    if( this->dnn_print && c % 16 == 0 )
                        printf( "[dnn] RN c%d: sumX=%g "
                                "( renormalized commit )\n",
                                c, (double)sum );
#endif
                    bad = 0;
                };
            };
            const float T_old = bat.T_old[ b0 + j ];
            if( bad == 0 && ( eps_n != eps_n
                 || eps_n <= 0.f
                 || eps_n > 1e30f ) )
                bad = 1;

            // --- Eos inversion: T from ( eps_hat, X_hat )
            // --- via nn_eos_T ( the eos.temp replication ).
            // --- Runs for any commit-eligible cell ( kind 1
            // --- outputs are nan ). T-bounds / dlgT guards
            // --- escalate from bad 0 ( first-hit-wins, v1
            // --- semantics ).
            float Tn = T_old;
            if( bad == 0 )
            {
                float y[ 128 ];
                for( int i = 0; i < n_spe; ++ i )
                    y[ i ] = float( X[ i ] );
                y[ n_spe ] = eps_n;
                Tn = this->nn_eos_T( eos, y, eps_n );
                if( bad == 0 )
                {
                    // Router upper edge: with T_route_hi set,
                    // a predicted T past the route bound is a
                    // kind-3 fallback ( commit-time guard so a
                    // single NN step cannot push a cell past
                    // the boundary the gate will enforce next
                    // cycle ).
                    const float T_hi_g =
                        ( this->T_route_hi > 0.f
                       && this->T_route_hi
                            < 2.f * nn.T_hi )
                            ? this->T_route_hi
                            : 2.f * nn.T_hi;
                    if( Tn < 0.5f * nn.T_lo
                         || Tn > T_hi_g )
                        bad = 3;
                    if( bad == 0 && T_old > 0.f
                         && fabsf( log10f( Tn / T_old ) )
                           > this->dTlg_cap )
                        bad = 4;
                };
            };

            if( bad != 0 )
            {
                // Anti-grind ( rc-40 ): this cell's cleanup-DI
                // has already failed dnn_grind_cap times ( each
                // ~iter_max substeps burned on an off-trajectory
                // state -- h collapses and the solve never
                // converges ).  Commit the clamped / renormalized
                // NN state instead: the cell stays on the NN
                // trajectory and the per-cycle grind breaks.
                // Semantics identical to the no_fb path.
                const bool grind_rn =
                    ( this->grind_cap > 0
                      && this->grind != nullptr
                      && this->grind[ c ] >= this->grind_cap );
                if( grind_rn && this->cnt != nullptr )
                    atomicAdd( this->cnt + 12, 1ull );
                if( this->no_fb || grind_rn )
                {
                    // Guarded, never DI: nan/absurd -> no-op
                    // ( keep state ); T bounds -> clamp into
                    // the envelope; dlgT -> clamp eps so the
                    // recomputed T hits the cap. ( sumX never
                    // reaches here -- kind 2 is always a
                    // renormalized commit, both modes. )
                    if( bad == 1 )
                    {
                        bat.ok[ b0 + j ] = 1;
                        if( this->cnt != nullptr )
                            atomicAdd( this->cnt + 6, 1ull );
#ifdef DNN_KERNEL_PRINTF
                        if( this->dnn_print && c % 16 == 0 )
                            printf( "[dnn] GUARD c%d kind=1"
                                    " ( no-op )\n", c );
#endif
                        continue;
                    };
                    if( bad == 3 )
                        Tn = fminf( fmaxf( Tn, nn.T_lo ),
                                    nn.T_hi );
                    if( bad == 4 )
                    {
                        const float r = powf
                            ( 10.f, this->dTlg_cap );
                        Tn = fminf( fmaxf( Tn, T_old / r ),
                                    T_old * r );
                    };
                    if( bad == 3 || bad == 4 )
                    {
                        // Re-invert eps from the clamped T
                        // with the SAME eos so the commit
                        // stays self-consistent.
                        float y[ 128 ];
                        for( int i = 0; i < n_spe; ++ i )
                            y[ i ] = float( X[ i ] );
                        y[ n_spe ] = 1.f;
                        eps_n = this->nn_eos_eps
                            ( eos, y, Tn );
                    };
                    bad = 0;
                }
                else
                {
                    bat.ok[ b0 + j ] = 2;  // cleanup -> DI
                    if( this->cnt != nullptr )
                    {
                        atomicAdd( this->cnt + 10, 1ull );
                        if( bad >= 1 && bad <= 4 )
                            atomicAdd( this->cnt + 5 + bad,
                                       1ull );
                    };
#ifdef DNN_KERNEL_PRINTF
                    if( this->dnn_print )
                        printf( "[dnn] FB c%d: guard kind=%d "
                                "( Tin=%g sumX=%g )\n",
                                c, bad, (double)T_old,
                                (double)sum );
#endif
                    continue;
                };
            };

            // --- Commit. Xhat is a mass fraction ( training
            // --- convention: X = ỹ/u_ρ renormalized ), so
            // --- u_i = X̂_i * rho ( code units ) is exact.
            // --- v2 energy: eps_hat IS the label ( specific
            // --- internal energy, erg/g, chemistry ODE
            // --- state ), so the write is exact — no
            // --- cv·μ-constant T-ratio approximation, and
            // --- chemical binding energy is carried by the
            // --- network. T_hyd = the eos-inverted T above.
            const float inv = 1.f / fmaxf( sum, 1e-30f );
            auto * u_tgt = bdt.hyd.u.at( idx );
            const auto rho( u_tgt[ idx::rho ] );
            auto * u_spe = u_tgt + eos.n_hyd;
            for( int i = 0; i < n_spe; ++ i )
                u_spe[ i ] = type::float_t( X[ i ] * inv )
                             * rho;
            {
                const float rho_cgs = float( rho )
                    * float( eos.unit.rho0 );
                const float e_cgs = float
                    ( eos.e_g_cgs( u_tgt ) );
                const float e_new_cgs = fmaxf
                    ( eps_n * rho_cgs, 0.f );
                const float de = ( e_new_cgs - e_cgs )
                    / float( eos.unit.ene0 );
                u_tgt[ idx::ene ] += type::float_t( de );
                bdt.T_hyd.at( idx )[ 0 ]
                    = type::float_t( Tn );
            };
            bat.ok[ b0 + j ] = 1;
            if( this->cnt != nullptr )
                atomicAdd( this->cnt + 4, 1ull );
#ifdef DNN_KERNEL_PRINTF
            if( this->dnn_print && c % 16 == 0 )
                printf( "[dnn] nn c%d: Tin=%g That=%g "
                        "sumX=%g\n", c, (double)T_old,
                        (double)Tn, (double)sum );
#endif
        };
        return;
    };
};
};                       // namespace extension::deeponet
