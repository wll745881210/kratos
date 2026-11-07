#pragma once

#include "../../../../src/modules/chemistry/solver/solver.h"
#include "../../../../src/modules/chemistry/corrector/corrector.h"
#include "dat.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
//

//  The saturation - region skip ( usr / intermediate /
//  disk_chem_sat.h ):  block data carrying a chem_skip
//  dat_3d member get their flagged cells skipped by the
//  solver's cell loop.   SFINAE - guarded:  block data
//  without the member ( e.g. the planet pgens' ) are
//  unaffected.
template< class bdt_T, class = void >
struct has_chem_skip : std::false_type
{
};
template< class bdt_T >
struct has_chem_skip< bdt_T, std::void_t
    < decltype( std::declval< bdt_T >(  ).chem_skip ) > >
    : std::true_type
{
};

template
< class cor_T = ::chemistry::solver::corrector  :: base_t,
  class dat_T = dat_t< type::float_t >,
  class derived_T = crtp::dummy_t >
struct solver_t : public ::chemistry::solver::default_t
      < cor_T, dat_T, solver_t< cor_T, dat_T, derived_T > >
{
    ////////// Types //////////
    __crtp_def_self__( solver_t, cor_T, dat_T, derived_T );
    using super_t = ::chemistry::solver::default_t
    < cor_T, dat_T, solver_t< cor_T, dat_T, derived_T > >;

    int n_hv;

    ////////// Functions //////////
    __host__ virtual int size_sh(  ) const override
    {
        auto res = super_t::size_sh(  );
        res     += n_hv * sizeof( typename dat_T::f_t );
        return utils::align( res, 8 );
    };
    
    template < class prx_T , class mod_T ,
               class rls_T , class eos_T >  __host__ void
    integrate( mod_T & mod , mesh::block::dual_t & d ,
               const rls_T & rls,  const eos_T & eos )
    {
        n_hv = rls.n_hv;
        auto & super = dynamic_cast< super_t & >( * this );
        return super.template integrate< prx_T >
             ( mod, d, rls, eos );
    };
    
    template< class bdt_T, class rls_T, class eos_T,
              class spc_T >
    __forceinline__  __device__   void operator(  )
    ( const bdt_T & bdt ,  const rls_T  & rls,
      const eos_T & eos,  const spc_T  & work )  const
    {
        __dyn_shared__( char, var_sh      );
        // Grid-stride over the cells: with the full grid
        // ( one block per cell ) the loop runs once; with
        // work_glob batching each block walks many cells.
        const int n_c[ 3 ]
        {
            bdt. geo. n_ceff[ 0 ], bdt. geo. n_ceff[ 1 ],
            bdt. geo. n_ceff[ 2 ]
        };
        const int n_ct( n_c[ 0 ] * n_c[ 1 ] * n_c[ 2 ] );
        const int n_bs( gridDim.x * gridDim. y
                                  * gridDim. z );
        const int i_bs( blockIdx. x + gridDim. x *
            ( blockIdx. y + gridDim. y * blockIdx. z ) );
        for( int c( i_bs ); c < n_ct; c += n_bs )
        {
            type :: idx_t idx_c;
            idx_c[ 0 ] =   c              %  n_c[ 0 ];
            idx_c[ 1 ] = ( c / n_c[ 0 ] ) %  n_c[ 1 ];
            idx_c[ 2 ] =   c / ( n_c[ 0 ] * n_c[ 1 ] );
            //  The saturation - region skip ( uniform per
            //  block:  the cell loop is over BLOCKS,  so all
            //  threads take the same branch ).
            if constexpr( has_chem_skip< bdt_T >::value )
                if( bdt.chem_skip.at( idx_c )[ 0 ] > 0 )
                    continue;
            char * base( var_sh );
            if( work )
                base = ( char * ) work. at( idx_c );
            dat_T d( this-> dim,  base, rls,
                     this-> cache_rate );
            d.  idx = idx_c;
            d  .load      ( bdt,          eos );
            rls.prep_dev  ( d  ,  bdt,    eos );
            //  Pre - solve NaN skip:  a cell whose INPUT
            //  state is already NaN / Inf can never
            //  converge --  skipping it here saves the full
            //  iter_max burn ( the v12b post - mortem:  60 %
            //  of all cells NaN'd species exhausting 1000
            //  iterations each ).
            {
                const int i_th( threadIdx.x );
                __shared__ int bad_in;
                if( i_th == 0 )
                    bad_in = 0;
                __syncthreads(  );
                for( int i = i_th; i < d.dim; i += blockDim.x )
                    if( __isnan( d.y[ i ] )
                        || __isinf( d.y[ i ] ) )
                        bad_in = 1;
                __syncthreads(  );
                if( bad_in )
                    continue;
            }
            for( d.iter = this->iter_max; d.iter > 0; )
                if( this->solve( d, rls, bdt, eos ) )
                    break;
            //  The stiff - cell forensics print ( [chemistry]
            //  dbg_iter:  cells using >= dbg_iter internal
            //  steps report count + cell + T;  0 = off ).
            if( this->dbg_iter > 0 && threadIdx.x == 0
                && this->iter_max - d.iter >= this->dbg_iter )
                printf( "CHEMITER %d at %d %d %d  T %e\n",
                        this->iter_max - d.iter,
                        idx_c[ 0 ], idx_c[ 1 ], idx_c[ 2 ],
                        double( d.T ) );
            //  Last - resort skip:  a cell whose solve
            //  failed irrecoverably ( iter_max exhausted,
            //  or the fallback itself NaN'd ) is simply
            //  NOT written back ( d.load only READS bdt,
            //  so skipping fix / T_hyd / save keeps the
            //  pre - solve state untouched;  ledger cont.
            //  50:  inner - rim wind - base cells NaN -
            //  bombed the whole state ).
            {
                const int i_th( threadIdx.x );
                __shared__ int bad_cell;
                if( i_th == 0 )
                    bad_cell = 0;
                __syncthreads(  );
                for( int i = i_th; i < d.dim; i += blockDim.x )
                    if( __isnan( d.y[ i ] )
                        || __isinf( d.y[ i ] ) )
                        bad_cell = 1;
                __syncthreads(  );
                if( bad_cell )
                    continue;
            }
            this->corr.template fix
                < type::float2_t > ( d, bdt, eos, true );
            eos.temp( d );
            __syncthreads(   );
            if( threadIdx.x == 0 )
                bdt.T_hyd.at( d.idx )[ 0 ] = d.T;
            d.save( bdt, eos );
            __syncthreads(   );
        }
        return;
    };

    // Diagnostic: evaluate every reaction's rate at the
    // current cell state and store into bdt.reac_rate. No
    // ODE integration; side effects on d.f are discarded.
    template< class bdt_T, class rls_T, class eos_T,
              class spc_T >
    __forceinline__ __device__ void compute_rates
    ( const bdt_T & bdt ,  const rls_T  & rls,
      const eos_T & eos,  const spc_T  & work )  const
    {
        __dyn_shared__( char, var_sh      );
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
            auto * rr = bdt.reac_rate.at( d.idx );
            for( int i = 0; i < rls.size(); ++i )
                rr[ i ] = rls.rate_val_single( d, i, bdt, eos );
            __syncthreads( );
        }
        return;
    };
};

//////////////////////////////////////////////////////////
// Diagnostic kernel: capture per-reaction rates into the
// block's reac_rate storage. Only launched when the
// output_reac_rates flag is on (output cycles only).

template
< class sol_T, class bdt_T, class rls_T, class eos_T,
  class spc_T >
__global__ void ker_rates
( const sol_T sol, const bdt_T bdt ,
  const rls_T rls, const eos_T eos,
  const spc_T work )
{
    return sol.compute_rates( bdt, rls, eos, work );
};
};                      // namespace extension::chemistry
