#include "edge_mg_wb_dev.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

////////////////////////////////////////////////////////////
//  The host half of the device write-back.
//
//  The tables -- per-face slots,  areas,  d1 rows,  the
//  CSR cell lists -- are built once per block,  so a step
//  is one launch ( mode 1 ) or three ( mode 0 ) with no
//  host traffic beyond `dt` and `bc_edge`.  Nothing here
//  is device code:  the `__global__` half is in
//  kernels/ni_ie_kernels.h,  and `wb0_build` stays in
//  the header because it is a template on the geometry.
////////////////////////////////////////////////////////////

namespace mhd_ee::ni::edge_mg
{

__host__ wb_dev_t::wb_dev_t(  )
{
    n        = idx_t::null(  );
    n_face   = 0;
    n_cell   = 0;
    n_deg    = 0;
    n_ar0    = 0;
    n_written= 0;
    n_dsup   = 0;
    p_face   = nullptr;
};

__host__ bool wb_dev_t::ok(  ) const
{
    return p_face != nullptr;
};

__host__ int wb_dev_fill_far
( const cplx_t & c, const float2_t * p_a, float_t * far )
{
    for( int f = 0; f < c.n_face; ++ f )
    {
        int   a( 0 );
        idx_t nd( 0 );
        c.face_key( f, a, nd );
        int    s( 0 );
        idx_t  i( nd );
        const int na = int( c.n[ a ] );
        if( int( nd[ a ] ) >= na )
        {
            i[ a ] = na - 1;
            s      = 1;
        }
        const int ic = ( int( i[ 2 ] ) * int( c.n[ 1 ] )
                       + int( i[ 1 ] ) ) * int( c.n[ 0 ] )
                       + int( i[ 0 ] );
        far[ std::size_t( ic ) * 6 + 2 * a + s ] = float_t( p_a[ f ] );
    }
    return c.n_face;
};

__host__ void wb_build
( wb_dev_t & w, const mesh::geo_t & geo, const cplx_t & c,
  const float_t * far_host, const dat_3d_t< float2_t > & bf )
{
    if( bf.n_int != 3 )
    {
        throw std::runtime_error
        ( "wb_build: bf must carry three components" );
    }
    if( int( c.n[ 0 ] ) != int( geo.n_ceff[ 0 ] )
     || int( c.n[ 1 ] ) != int( geo.n_ceff[ 1 ] )
     || int( c.n[ 2 ] ) != int( geo.n_ceff[ 2 ] ) )
    {
        throw std::runtime_error
        ( "wb_build: cplx_t does not belong to this geo" );
    }
    w.n      = c.n;
    w.n_face = c.n_face;
    w.n_cell = c.n_cell;
    w.n_deg  = 0;
    w.n_ar0  = 0;
    w.n_written = 0;
    w.n_dsup = 0;
    w.face_h.assign( std::size_t( c.n_face ), wb_face_t(  ) );
    std::vector< int > slot( std::size_t( c.n_face ), - 1 );
    int n_seam( 0 );
    for( int f = 0; f < c.n_face; ++ f )
    {
        int   a( 0 );
        idx_t nd( 0 );
        c.face_key( f, a, nd );
        //  ---- the canonical owner,  the SAME decomposition
        //  the assembly uses
        int   s( 0 );
        idx_t i( nd );
        const int na = int( c.n[ a ] );
        if( int( nd[ a ] ) >= na )
        {
            i[ a ] = na - 1;
            s      = 1;
        }
        //  a periodic axis has no high wall -- no face of it
        //  may be owned from the s = 1 side
        if( s == 1 && c.per[ a ] != 0 )
        {
            ++ n_seam;
        }
        const int ic = ( int( i[ 2 ] ) * int( c.n[ 1 ] )
                       + int( i[ 1 ] ) ) * int( c.n[ 0 ] )
                       + int( i[ 0 ] );
        const int deg = c.degenerate_f( f ) ? 1 : 0;
        const float_t ar = far_host[ std::size_t( ic ) * 6
                                   + std::size_t( 2 * a + s ) ];
        //  ---- the bf slot:  `bf.idx` carries the ghost
        //  offsets and the component stride,  so the kernel
        //  only ever adds
        const int j = bf.idx( nd );
        if( j < 0 || j + 2 >= bf.n_ctot )
        {
            throw std::runtime_error
            ( "wb_build: bf slot outside the array" );
        }
        wb_face_t & r = w.face_h[ std::size_t( f ) ];
        r.slot = j + a;
        r.ar   = ar;
        r.a    = a;
        r.s    = s;
        r.ic   = ic;
        r.deg  = deg;
        if( deg != 0 )
        {
            ++ w.n_deg;
        }
        if( ! ( ar > float_t( 0 ) ) )
        {
            ++ w.n_ar0;
        }
        if( deg == 0 && ar > float_t( 0 ) )
        {
            ++ w.n_written;
        }
        slot[ std::size_t( f ) ] = r.slot;
    }
    //  ---- no atomics:  the slot map must be injective.  A
    //  collision would be a silent race ( last writer wins )
    if( n_seam != 0 )
    {
        throw std::runtime_error
        ( "wb_build: a periodic-axis face is owned from "
          "the high side" );
    }
    std::sort( slot.begin(  ), slot.end(  ) );
    for( std::size_t q = 1; q < slot.size(  ); ++ q )
    {
        if( slot[ q ] == slot[ q - 1 ] )
        {
            throw std::runtime_error
            ( "wb_build: two faces share a bf slot" );
        }
    }
    if( ! slot.empty( ) )
    {
        w.n_dsup = int( std::unique
                        ( slot.begin( ), slot.end( ) )
                        - slot.begin( ) );
    }
    return;
};

__host__ void wb_upload( wb_dev_t & w, device::base_t & dev )
{
    if( w.p_face != nullptr )
    {
        dev.f_free( w.p_face );
        w.p_face = nullptr;
    }
    if( w.face_h.empty( ) )
    {
        return;
    }
    w.p_face = ( wb_face_t * ) dev.f_malloc
             ( w.face_h.size(  ) * sizeof( wb_face_t ) );
    dev.cp( w.p_face, w.face_h.data(  ), w.face_h.size(  ) );
    return;
};

__host__ void wb_dev_setup
( wb_dev_t & w, device::base_t & dev, const mesh::geo_t & geo,
  const cplx_t & c, const float_t * far_host,
  const dat_3d_t< float2_t > & bf )
{
    wb_build( w, geo, c, far_host, bf );
    wb_upload( w, dev );
    return;
};

__host__ void wb_dev_free( device::base_t & dev, wb_dev_t & w )
{
    if( w.p_face != nullptr )
    {
        dev.f_free( w.p_face );
        w.p_face = nullptr;
    }
    w.face_h.clear(  );
    w.face_h.shrink_to_fit(  );
    w.n_face = w.n_cell = 0;
    return;
};

__host__ void ni_ie_writeback_dev
( device::base_t & dev, const wb_dev_t & w,
  dat_3d_t< float2_t > & bf, const float_t * p_rep_old,
  const float_t * p_rep_new, const int & n_th_lim )
{
    if( w.p_face == nullptr || w.n_face <= 0 )
    {
        return;
    }
    int t = n_th_lim;
    if( t > 1024 )
    {
        t = 1024;
    }
    if( t < 1 )
    {
        t = 1;
    }
    if( t > w.n_face )
    {
        t = w.n_face;
    }
    const int nb = ( w.n_face + t - 1 ) / t;
    dev.launch
    ( wb_dev_k, dim3( nb ), dim3( t ), 0, nullptr,
      ( const wb_face_t * ) w.p_face, w.n_face, bf.dat,
      p_rep_old, p_rep_new );
    dev.sync_all_streams(  );
    return;
};

__host__ wb0_dev_t::wb0_dev_t(  )
{
    n         = idx_t::null(  );
    n_cell    = 0;
    n_face    = 0;
    n_edge    = 0;
    n_ctr     = 0;
    n_deg     = 0;
    n_wall    = 0;
    n_cnt0    = 0;
    a_n_ctot  = 0;
    bf_n_ctot = 0;
    p_cf      = nullptr;
    p_ar6     = nullptr;
    p_fed     = nullptr;
    p_fsg     = nullptr;
    p_fnz     = nullptr;
    p_fdg     = nullptr;
    p_eptr    = nullptr;
    p_ecell   = nullptr;
    p_edl     = nullptr;
    p_ewall   = nullptr;
    p_aslot   = nullptr;
    p_ej      = nullptr;
    p_ee      = nullptr;
};

__host__ bool wb0_dev_t::ok(  ) const
{
    return p_cf != nullptr && p_fed != nullptr
        && p_eptr != nullptr && n_face > 0 && n_edge > 0;
};

__host__ idx_t wb0_idx( const cplx_t & c, const int & ic )
{
    const idx_t n( c.n );
    return idx_t( ic % int( n[ 0 ] ),
                  ( ic / int( n[ 0 ] ) ) % int( n[ 1 ] ),
                  ic / int( n[ 0 ] * n[ 1 ] ) );
};

__host__ void wb0_upload( wb0_dev_t & w, device::base_t & dev )
{
    wb0_dev_free( dev, w );
    auto alloc = [ & ]( const std::size_t & n_b ) -> void *
    {
        return dev.f_malloc( n_b == 0 ? 1 : n_b );
    };
    w.p_cf    = ( int *         ) alloc
        ( w.cf.size(  ) * sizeof( int ) );
    w.p_ar6   = ( float_t *     ) alloc
        ( w.ar6.size(  ) * sizeof( float_t ) );
    w.p_fed   = ( int *         ) alloc
        ( w.fed.size(  ) * sizeof( int ) );
    w.p_fsg   = ( signed char * ) alloc( w.fsg.size(  ) );
    w.p_fnz   = ( signed char * ) alloc( w.fnz.size(  ) );
    w.p_fdg   = ( signed char * ) alloc( w.fdg.size(  ) );
    w.p_eptr  = ( int *         ) alloc
        ( w.eptr.size(  ) * sizeof( int ) );
    w.p_ecell = ( int *         ) alloc
        ( w.ecell.size(  ) * sizeof( int ) );
    w.p_edl   = ( float_t *     ) alloc
        ( w.edl.size(  ) * sizeof( float_t ) );
    w.p_ewall = ( signed char * ) alloc( w.ewall.size(  ) );
    w.p_aslot = ( int *         ) alloc
        ( w.aslot.size(  ) * sizeof( int ) );
    w.p_ej    = ( float_t *     ) alloc
        ( std::size_t( w.n_cell ) * 3 * sizeof( float_t ) );
    w.p_ee    = ( float_t *     ) alloc
        ( std::size_t( w.n_edge ) * sizeof( float_t ) );
    dev.cp( w.p_cf,    w.cf.data(  ),    w.cf.size(  ) );
    dev.cp( w.p_ar6,   w.ar6.data(  ),   w.ar6.size(  ) );
    dev.cp( w.p_fed,   w.fed.data(  ),   w.fed.size(  ) );
    dev.cp( w.p_fsg,   w.fsg.data(  ),   w.fsg.size(  ) );
    dev.cp( w.p_fnz,   w.fnz.data(  ),   w.fnz.size(  ) );
    dev.cp( w.p_fdg,   w.fdg.data(  ),   w.fdg.size(  ) );
    dev.cp( w.p_eptr,  w.eptr.data(  ),  w.eptr.size(  ) );
    dev.cp( w.p_ecell, w.ecell.data(  ), w.ecell.size(  ) );
    dev.cp( w.p_edl,   w.edl.data(  ),   w.edl.size(  ) );
    dev.cp( w.p_ewall, w.ewall.data(  ), w.ewall.size(  ) );
    dev.cp( w.p_aslot, w.aslot.data(  ), w.aslot.size(  ) );
    dev.sync_all_streams(  );
    return;
};

__host__ void wb0_dev_free( device::base_t & dev, wb0_dev_t & w )
{
    for( auto * p : { ( void ** ) & w.p_cf, ( void ** ) & w.p_ar6,
                      ( void ** ) & w.p_fed, ( void ** ) & w.p_fsg,
                      ( void ** ) & w.p_fnz, ( void ** ) & w.p_fdg,
                      ( void ** ) & w.p_eptr, ( void ** ) & w.p_ecell,
                      ( void ** ) & w.p_edl, ( void ** ) & w.p_ewall,
                      ( void ** ) & w.p_aslot, ( void ** ) & w.p_ej,
                      ( void ** ) & w.p_ee } )
    {
        if( * p != nullptr )
        {
            dev.f_free( * p );
            * p = nullptr;
        }
    }
    return;
};

__host__ void wb0_dev_setup
( wb0_dev_t & w, device::base_t & dev,
  const mesh::geo_t & geo, const cplx_t & c,
  const float_t * far_host, const dat_3d_t< float_t > & a,
  const dat_3d_t< float2_t > & bf )
{
    if( c.n != geo.n_ceff )
    {
        throw std::runtime_error
            ( "wb0_dev_setup:  the complex was not built for"
              " this block." );
    }
    if( geo.geo_type == mesh::geo_t::cartesian )
    {
        geo_cart_t g;
        if( ! geo_cart_from( geo, g ) )
        {
            throw std::runtime_error
                ( "wb0_dev_setup:  cartesian policy." );
        }
        wb0_build( w, g, c, far_host, a, bf );
    }
    else if( geo.geo_type == mesh::geo_t::spherical )
    {
        geo_sph_t g;
        if( ! geo_sph_from( geo, g ) )
        {
            throw std::runtime_error
                ( "wb0_dev_setup:  spherical policy." );
        }
        wb0_build( w, g, c, far_host, a, bf );
    }
    else
    {
        throw std::runtime_error
            ( "wb0_dev_setup:  no policy for this geometry." );
    }
    wb0_upload( w, dev );
    return;
};

__host__ void ni_ie_writeback0_dev
( device::base_t & dev, const wb_dev_t & w,
  const wb0_dev_t & w0, dat_3d_t< float2_t > & bf,
  const dat_3d_t< float_t > & a, const float2_t * p_lat,
  const float_t & dt, const int & bc_edge,
  const int & n_th_lim )
{
    if( ! w.ok(  ) || ! w0.ok(  ) )
    {
        return;
    }
    if( p_lat == nullptr )
    {
        throw std::runtime_error
            ( "ni_ie_writeback0_dev:  p_lat is null." );
    }
    if( int( bf.n_ctot ) != w0.bf_n_ctot
     || int( a.n_ctot ) != w0.a_n_ctot )
    {
        throw std::runtime_error
            ( "ni_ie_writeback0_dev:  the bf / a lattice changed;"
              " rebuild the table." );
    }
    int t = n_th_lim;
    if( t > 1024 )
    {
        t = 1024;
    }
    if( t < 1 )
    {
        t = 1;
    }
    //  ---- pass 1
    {
        int tt = t;
        if( tt > w0.n_cell )
        {
            tt = w0.n_cell;
        }
        const int nb = ( w0.n_cell + tt - 1 ) / tt;
        dev.launch
        ( wb0_cell_k, dim3( nb ), dim3( tt ), 0, nullptr,
          w0.n_cell, ( const int * ) w0.p_cf,
          ( const float_t * ) w0.p_ar6,
          ( const signed char * ) w0.p_fdg,
          ( const signed char * ) w0.p_fsg,
          ( const signed char * ) w0.p_fnz,
          ( const int * ) w0.p_fed,
          ( const int * ) w0.p_aslot,
          ( const float_t * ) a.dat,
          ( const float2_t * ) p_lat,
          ( float_t * ) w0.p_ej );
    }
    //  ---- pass 2
    {
        int tt = t;
        if( tt > w0.n_edge )
        {
            tt = w0.n_edge;
        }
        const int nb = ( w0.n_edge + tt - 1 ) / tt;
        dev.launch
        ( wb0_edge_k, dim3( nb ), dim3( tt ), 0, nullptr,
          w0.n_edge, ( const int * ) w0.p_eptr,
          ( const int * ) w0.p_ecell,
          ( const float_t * ) w0.p_edl,
          ( const signed char * ) w0.p_ewall,
          ( const float_t * ) w0.p_ej, bc_edge,
          ( float_t * ) w0.p_ee );
    }
    //  ---- pass 3
    {
        int tt = t;
        if( tt > w0.n_face )
        {
            tt = w0.n_face;
        }
        const int nb = ( w0.n_face + tt - 1 ) / tt;
        dev.launch
        ( wb0_face_k, dim3( nb ), dim3( tt ), 0, nullptr,
          w0.n_face, ( const wb_face_t * ) w.p_face,
          ( const int * ) w0.p_fed,
          ( const signed char * ) w0.p_fsg,
          ( const signed char * ) w0.p_fnz,
          ( const float_t * ) w0.p_ee, dt, bf.dat );
    }
    dev.sync_all_streams(  );
    return;
};

};    // namespace mhd_ee::ni::edge_mg