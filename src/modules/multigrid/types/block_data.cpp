#include "../../../utilities/mapping/loop.h"
#include "block_data.h"
#include "dual_data.h"

#include <algorithm>

namespace multigrid::data
{
////////////////////////////////////////////////////////////
// Block data holder for multigrid

__host__ void block_data_t::set_lvl
( dual_t & d, const int & dla ) // abs( d_l )
{
    auto & f_dev   ( * d.p_dev );
    auto & f_mgn   ( * d.b_w.get(  ). p_mesh->p_mgn );
    auto & geo_base(   d.b_w.get(  ).           geo );
    const auto f_new_h = [ & ]      ( const auto & c )
    {   return f_dev.malloc_host    < char > ( c ) ; };
    const auto f_new_d = [ & ]      ( const auto & c )
    {   return f_dev.malloc_device  < char > ( c ) ; };

    auto & b_h( dynamic_cast< block_data_t & >( d.h(  ) ) );
    mgd.order(  ) =      b_h.mgd.order(  ) ;
    mgd.n_fld(  ) =  2 * b_h.mgd.n_fld(  ) ;
    mgd.d_lvl     = -dla;

    geo_t  geo_h;
    geo_base.cp_prim( geo_h );
    for( int a = 0; a < geo_h.n_ceff.size(  ); ++ a )
        geo_h.n_ceff[ a ] = std::max
            ( 1, geo_base.n_ceff[ a ] >> dla );

    geo_h.setup( geo_h.n_ceff, mgd.order(  ), f_new_h );

    const int n_base( 1 <<    dla ) ;
    geo_h.vol0  *=  ( 1 <<  ( dla * 3 ) ) ;
    for( int a = 0; a < geo_h.n_dim; ++ a )
    {
        geo_h.dx0 [ a ] *= ( 1 <<   dla       );
        geo_h.sfc0[ a ] *= ( 1 << ( dla * 2 ) );
        if( geo_h.dx0[ a ] > 0 )
            continue;

        //  Coarse face  i  IS fine face  i * n_base.   The
        //  old  i / n_base  reached only the first
        //  1 / n_base  of the fine table and handed
        //  consecutive coarse faces the SAME fine face,  so
        //  every coarse cell had  dx = 0  on one side  ->
        //  w_c = -inf  ->  defect = inf  ->  the V-cycle was
        //  rolled back and the solve never iterated.
        const int n_g( geo_h.n_gh      );
        const int n_c( geo_h.n_ceff[ a ] );
        for( int i = 0; i <= n_c; ++ i )
            geo_h.xf[ a ][ i + n_g ]
                = geo_base.x_fc( a, i * n_base );
        //  The tables arrive uninitialized:  mirror the two
        //  end faces so the edge cells keep  dx > 0.
        for( int k = 1; k <= n_g + 1; ++ k )
            geo_h.xf[ a ][ n_g + n_c + k ]
                = 2 * geo_h.xf[ a ][ n_g + n_c ]
                - geo_h.xf[ a ][ n_g + n_c - k ];
        for( int k = 1; k <= n_g; ++ k )
            geo_h.xf[ a ][ n_g - k ]
                = 2 * geo_h.xf[ a ][ n_g ]
                - geo_h.xf[ a ][ n_g + k ];
        //  The coarse centre is the midpoint of the two
        //  coarse faces bounding its cell.
        for( int i = - n_g; i <= n_c + n_g; ++ i )
            geo_h.xv[ a ][ i + n_g ]
                = ( geo_h.xf[ a ][ i + n_g     ]
                  + geo_h.xf[ a ][ i + n_g + 1 ] ) / 2;
    }
    if( geo_h.vol0 <= 0 )
    {
        //  Geometry tables are ghost-padded: loop logical
        //  indices ( box index - n_gh ).
        idx_t i_min( - geo_h.n_gh ), i_max;
        for( int m = 0; m < 3; ++ m )
            i_max[ m ] = geo_h.vol.n_cell[ m ] - geo_h.n_gh;
        for( auto i : utils::loop( i_min, i_max ) )
            geo_h.vol( i ) = 0;
        idx_t j_min( - geo_base.n_gh ), j_max;
        for( int m = 0; m < 3; ++ m )
            j_max[ m ] = geo_base.vol.n_cell[ m ] - geo_base.n_gh;
        for( auto j : utils::loop( j_min, j_max ) )
        {
            auto  i = j ;
            for( auto & i_s : i.x )
                i_s >>= dla ;
            geo_h.vol( i ) += geo_base.vol( j );
        }
    }
    for ( int a = 0;  a <  geo_h.n_dim; ++ a )
    {
        if( geo_h.sfc0[ a ] > 0 )
            continue;
        const auto & sfc_base( geo_base.sfc [ a ] );
        if( ! sfc_base )
            continue;
    //  The coarse  a - face  I  IS the fine  a - face
    //  I * s_a  ( the same convention the  xf / xv  patch
    //  above uses ),  and its transverse extent is the union
    //  of the fine cells  I_b * s_b ..  I_b * s_b + s_b - 1.
    //  A THIN axis has ONE slot  ( s = 1 ):  the old code
    //  summed its periodic image as if it were a second cell
    //  and made the  n_phi = 1  area table twice too large
    //  ( the F2 telescoping then failed by exactly 1/2
    //  and 1/3 ).
    idx_t s;
    for( int m = 0; m < 3; ++ m )
        s[ m ] = ( geo_h.n_ceff[ m ] > 1 ? n_base : 1 );
    const int g_f( geo_base.n_gh );
    const auto & nf( sfc_base.n_cell  );
    idx_t i_min( - geo_h.n_gh ), i_max;
    for( int m = 0; m < 3; ++ m )
        i_max[ m ] = geo_h.sfc[ a ].n_cell[ m ] - geo_h.n_gh;
    for( auto i : utils::loop( i_min, i_max ) )
    {
        type::float_t acc( 0 );
        const int n_l[ 3 ]
        = { a == 0 ? 1 : s[ 0 ], a == 1 ? 1 : s[ 1 ],
            a == 2 ? 1 : s[ 2 ] };
        for( int l0 = 0; l0 < n_l[ 0 ]; ++ l0 )
        for( int l1 = 0; l1 < n_l[ 1 ]; ++ l1 )
        for( int l2 = 0; l2 < n_l[ 2 ]; ++ l2 )
        {
            const int o[ 3 ] = { l0, l1, l2 };
            idx_t j;
            for( int m = 0; m < 3; ++ m )
            {
                //  The fine ghost band is not ours to index
                //  ( s * g  exceeds it ):  clamp,  so the
                //  coarse ghost band is a sensible one - sided
                //  copy instead of uninitialized memory.
                const int lim( nf[ m ] - g_f - 1 );
                j[ m ] = std::min
                    ( std::max( i[ m ] * s[ m ] + o[ m ], - g_f ),
                      lim );
            }
            acc += sfc_base( j );
        }
        geo_h.sfc[ a ]( i ) = acc;
    }
    }
#ifdef  DBG_LVL_GEO
    //  Temporary self - check:  the coarse centres must be
    //  strictly increasing and the coarse volumes must sum
    //  to the fine ones.
    {
        double v_c( 0 ), v_f( 0 );
        idx_t i_min( - geo_h.n_gh ), i_max;
        for( int m = 0; m < 3; ++ m )
            i_max[ m ] = geo_h.vol.n_cell[ m ] - geo_h.n_gh;
        for( auto i : utils::loop( i_min, i_max ) )
            v_c += geo_h.vol( i );
        idx_t j_min( - geo_base.n_gh ), j_max;
        for( int m = 0; m < 3; ++ m )
            j_max[ m ] = geo_base.vol.n_cell[ m ] - geo_base.n_gh;
        for( auto j : utils::loop( j_min, j_max ) )
            v_f += geo_base.vol( j );
        double d_min( 1e30 );
        for( int a = 0; a < geo_h.n_dim; ++ a )
        {
            if( geo_h.dx0[ a ] > 0 )
                continue;
            for( int i = 0; i < geo_h.n_ceff[ a ]; ++ i )
                d_min = std::min
                    ( d_min, double( geo_h.x_cc
                                     ( a, i + 1 )
                                   - geo_h.x_cc( a, i ) ) );
        }
        printf( "DBGGEO dla=%d nce=%d,%d,%d  volc/volf=%.12f"
                "  min_dx_cc=%.6e\n",
                dla, geo_h.n_ceff[ 0 ], geo_h.n_ceff[ 1 ],
                geo_h.n_ceff[ 2 ],
                v_f > 0 ? v_c / v_f : - 1, d_min );
    }
#endif
    mgd  .setup   ( f_new_d,  geo_h );
    mgd  .null    ( [ & ]   ( void * p, const size_t & s )
                  { f_dev.f_mset( p, 0, s, 0 ); } );
    geo_h.cp_prim ( geo             );
    geo  .setup   ( geo_h , f_new_d );
    geo_h.cp_deep ( geo, f_dev.f_cp );
    geo_h.free( [ & ]( void * p ){ f_dev.free_host( p );} );
    return;
}
};                              // namespace multigrid::data
