#pragma once

#include "../../../../../../src/types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

////////////////////////////////////////////////////////////
//  Minimal host linear algebra for the gates:  a CSR matrix
//  built from ( row, col, value ) triplets, its matvec, and
//  a Jacobi-preconditioned conjugate gradient.
//
//  Why CG and not an LU factorisation:  the A-gates run at
//  PRECISION=2, where a converged CG IS a reference
//  solution
//  ( rtol 1e-13 ), and it scales -- dense LU does not
//  survive
//  10^4 DOFs and this tree has no sparse direct solver.  A5
//  later compares the MG preconditioner against THIS, so
//  the
//  gate always prints the true final residual and it can be
//  checked.
//
//  Host-only,  no device code:  the gate TUs stay at a few
//  lines of PTX ( rule C1 of edge-mg-foundations.md ).

namespace mhd_ee::ni::edge_mg
{
using type::float_t;

#if PRECISION == 2
constexpr float_t cheb_safety = 1.05;
#else
constexpr float_t cheb_safety = 1.5;
#endif

struct csr_t
{
    int                   n;
    //  The COLUMN count. A square operator has ncol == n,
    //  but G
    //  ( edges x nodes ) and C ( edges x faces ) are
    //  rectangular, and
    //  a Galerkin product needs the COLUMN count of the
    //  transfer. Not
    //  tracking it built Sg as n_edge x n_edge instead of
    //  n_node^2 and
    //  Sa as n_edge x n_edge instead of n_face^2, i.e. both
    //  auxiliary
    //  spaces were mostly empty rows and contributed almost
    //  nothing
    //  ( found 2026-10-29 ).
    int                ncol;
    std::vector< int   >  ptr;   // n + 1
    std::vector< int   >  idx;
    std::vector< float_t > val;

    void build( const int & n_,
                std::vector< int    > & rows,
                std::vector< int    > & cols,
                std::vector< float_t > & vals )
    {
        n = n_;
        const std::size_t m = rows.size();
        std::vector< std::size_t > ord( m );
        for( std::size_t i = 0; i < m; ++ i )
        {
            ord[ i ] = i;
        }
        std::sort( ord.begin(  ), ord.end(  ),
                   [ & ]( const std::size_t & a, const std::size_t & b )
                   {
                       if( rows[ a ] != rows[ b ] )
                       {
                           return rows[ a ] < rows[ b ];
                       }
                       return cols[ a ] < cols[ b ];
                   } );
        idx.clear(  );
        val.clear(  );
        ptr.assign( n + 1, 0 );
        std::vector< int > row_of;
        row_of.reserve( m );
        int r_prev( -1 ), c_prev( -1 );
        for( std::size_t q = 0; q < m; ++ q )
        {
            const std::size_t i = ord[ q ];
            if( rows[ i ] == r_prev && cols[ i ] == c_prev )
            {
                val.back(  ) += vals[ i ];
                continue;
            }
            idx.push_back( cols[ i ] );
            val.push_back( vals[ i ] );
            row_of.push_back( rows[ i ] );
            r_prev = rows[ i ];
            c_prev = cols[ i ];
        }
        //  row pointers from the ( non-decreasing ) row_of
        std::size_t k( 0 );
        for( int r = 0; r < n; ++ r )
        {
            while( k < row_of.size(  ) && row_of[ k ] < r )
            {
                ++ k;
            }
            ptr[ r ] = int( k );
        }
        ptr[ n ] = int( idx.size(  ) );
        ncol = 0;
        for( std::size_t q = 0; q < idx.size(  ); ++ q )
        {
            if( idx[ q ] + 1 > ncol )
            {
                ncol = idx[ q ] + 1;
            }
        }
        for( int qp = 0; qp <= n; ++ qp )
        {
            ( void ) qp;
        }
        return;
    };

    //  the same, with the column count given by the caller
    //  ( a
    //  transfer whose last column happens to be empty would
    //  otherwise
    //  be under-counted )
    void build( const int & n_, const int & ncol_,
                std::vector< int > & rows,
                std::vector< int > & cols,
                std::vector< float_t > & vals )
    {
        build( n_, rows, cols, vals );
        if( ncol_ > ncol )
        {
            ncol = ncol_;
        }
        return;
    };

    void mul( const std::vector< float_t > & x,
              std::vector< float_t > & y ) const
    {
        y.assign( n, 0. );
        for( int r = 0; r < n; ++ r )
        {
            float_t s( 0 );
            for( int q = ptr[ r ]; q < ptr[ r + 1 ]; ++ q )
            {
                s += val[ q ] * x[ idx[ q ] ];
            }
            y[ r ] = s;
        }
        return;
    };

    //  y = A^T x
    void mul_t( const std::vector< float_t > & x,
                std::vector< float_t > & y ) const
    {
        y.assign( ncol > 0 ? ncol : n, 0. );
        for( int r = 0; r < n; ++ r )
        {
            const float_t xr = x[ r ];
            if( xr == 0. )
            {
                continue;
            }
            for( int q = ptr[ r ]; q < ptr[ r + 1 ]; ++ q )
            {
                y[ idx[ q ] ] += val[ q ] * xr;
            }
        }
        return;
    };

    std::vector< float_t > diag(  ) const
    {
        std::vector< float_t > d( n, 0. );
        for( int r = 0; r < n; ++ r )
        {
            for( int q = ptr[ r ]; q < ptr[ r + 1 ]; ++ q )
            {
                if( idx[ q ] == r )
                {
                    d[ r ] = val[ q ];
                }
            }
        }
        return d;
    };
};

inline float_t _dot( const std::vector< float_t > & a,
                    const std::vector< float_t > & b )
{
    float_t s( 0 );
    for( std::size_t i = 0; i < a.size(  ); ++ i )
    {
        s += a[ i ] * b[ i ];
    }
    return s;
};

//  the position of the diagonal entry of every row ( -1
//  when the pattern
//  has none ). Cached once, so a per-dt diagonal shift is
//  an in-place
//  write instead of a rebuild with a sort.
inline std::vector< int > diag_pos( const csr_t & M )
{
    std::vector< int > d( M.n, -1 );
    for( int r = 0; r < M.n; ++ r )
    {
        for( int q = M.ptr[ r ]; q < M.ptr[ r + 1 ]; ++ q )
        {
            if( M.idx[ q ] == r )
            {
                d[ r ] = q;
                break;
            }
        }
    }
    return d;
};

//  add sh to the diagonal of every row, INSERTING the entry
//  where the
//  pattern has none. The auxiliary operators come from G
//  and C, which
//  leave whole rows empty, and a missing diagonal is what
//  made the first
//  IC(0) produce infinite factors ( 2026-10-29 ).
inline void add_diag( csr_t & M, const float_t & sh )
{
    std::vector< int    > r, c;
    std::vector< float_t > v;
    for( int i = 0; i < M.n; ++ i )
    {
        for( int q = M.ptr[ i ]; q < M.ptr[ i + 1 ]; ++ q )
        {
            r.push_back( i );
            c.push_back( M.idx[ q ] );
            v.push_back( M.val[ q ] );
        }
    }
    for( int i = 0; i < M.n; ++ i )
    {
        r.push_back( i );
        c.push_back( i );
        v.push_back( sh );
    }
    M.build( M.n, M.n, r, c, v );
    return;
};

//  Jacobi-preconditioned CG.  Returns the TRUE residual
//  |b - A x| / |b|;  `it` is the iteration count.
inline float_t cg( const csr_t & A, const std::vector<
    float_t > & b,
                  std::vector< float_t > & x, const float_t
                      & tol,
                  const int & it_max, int & it )
{
    const int n = A.n;
    if( int( x.size(  ) ) != n )
    {
        x.assign( n, 0. );
    }
    const std::vector< float_t > d = A.diag(  );
    std::vector< float_t > r( n ), z( n ), p( n ), Ap( n );
    A.mul( x, Ap );
    for( int i = 0; i < n; ++ i )
    {
        r[ i ] = b[ i ] - Ap[ i ];
    }
    const float_t b2 = _dot( b, b );
    const float_t b2s = b2 > 0 ? b2 : 1.;
    float_t rz( 0 );
    for( int i = 0; i < n; ++ i )
    {
        const float_t di = d[ i ] != 0. ? d[ i ] : 1.;
        z[ i ] = r[ i ] / di;
        p[ i ] = z[ i ];
        rz += r[ i ] * z[ i ];
    }
    it = 0;
    float_t res( std::sqrt( _dot( r, r ) / b2s ) );
    while( res > tol && it < it_max )
    {
        A.mul( p, Ap );
        const float_t pAp = _dot( p, Ap );
        if( pAp <= 0. )
        {
            break;
        }
        const float_t al = rz / pAp;
        for( int i = 0; i < n; ++ i )
        {
            x[ i ] += al * p[ i ];
            r[ i ] -= al * Ap[ i ];
        }
        float_t rz_new( 0 );
        for( int i = 0; i < n; ++ i )
        {
            const float_t di = d[ i ] != 0. ? d[ i ] : 1.;
            z[ i ] = r[ i ] / di;
            rz_new += r[ i ] * z[ i ];
        }
        const float_t be = rz_new / rz;
        rz = rz_new;
        for( int i = 0; i < n; ++ i )
        {
            p[ i ] = z[ i ] + be * p[ i ];
        }
        ++ it;
        res = std::sqrt( _dot( r, r ) / b2s );
    }
    return res;
};

};  // namespace edge_mg
