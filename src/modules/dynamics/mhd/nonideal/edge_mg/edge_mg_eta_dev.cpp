#include "edge_mg_eta_dev.h"

#include <algorithm>
#include <tuple>

////////////////////////////////////////////////////////////
//  The host half of the eta lattice.
//
//  Both entries are one launch and one barrier:  the device
//  packs eta where eta already is,  and the sub-cell stretch
//  reuses `eta_centre` from the header.
//  Nothing here is device code -- the `__global__` layer
//  is in kernels/ni_ie_kernels.h and the two `apply`s
//  stay in the header.
////////////////////////////////////////////////////////////

namespace mhd_ee::ni::edge_mg
{

__host__ std::tuple< dim3, dim3, int >
eta_lat_t::resource( const int & n_th_lim ) const
{
    const int n = n_cell > 0 ? n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ),
                            dim3( t ), 0 );
};

__host__ void ni_ie_eta_lat_dev
( device::base_t & dev, const mesh::geo_t & geo,
  const mesh::dat_3d_t< float_t > & eta,
  float2_t * p_lat, const int & n_th_lim )
{
    eta_lat_t op;
    op.eta    = eta;
    op.n_ceff = geo.n_ceff;
    op.p_lat  = p_lat;
    op.n_cell = int( geo.n_ceff[ 0 ] ) * int( geo.n_ceff[ 1 ] )
              * int( geo.n_ceff[ 2 ] );
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
    dev.launch( eta_lat_k< eta_lat_t >, n_bl, n_th, s_sh, nullptr,
                op, op.n_cell );
    dev.sync_all_streams(  );
    return;
};

__host__ std::tuple< dim3, dim3, int >
eta_xc_t::resource( const int & n_th_lim ) const
{
    const int n = n_cell > 0 ? n_cell : 1;
    const int t = std::min( n_th_lim, n );
    return std::make_tuple( dim3( ( n + t - 1 ) / t ),
                            dim3( t ), 0 );
};

__host__ void ni_ie_eta_xc_dev
( device::base_t & dev, const geo_sph_t & g,
  const idx_t & n_ceff, float2_t * p_xc, const int & n_th_lim )
{
    eta_xc_t op;
    op.g      = g;
    op.n_ceff = n_ceff;
    op.p_xc   = p_xc;
    op.n_cell = int( n_ceff[ 0 ] ) * int( n_ceff[ 1 ] )
              * int( n_ceff[ 2 ] );
    const auto [ n_bl, n_th, s_sh ] = op.resource( n_th_lim );
    dev.launch( eta_xc_k< eta_xc_t >, n_bl, n_th, s_sh, nullptr,
                op, op.n_cell );
    dev.sync_all_streams(  );
    return;
};

};    // namespace mhd_ee::ni::edge_mg