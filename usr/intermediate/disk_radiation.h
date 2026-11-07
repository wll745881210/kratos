#pragma once

#include <map>

#include "../../src/modules/particle/radiation/radiation.h"
#include "../../src/modules/particle/radiation/photon_sph.h"
#include "../../usr/extension/chem_therm/reaction/photoreac_prep.h"
#include "../../usr/extension/radiation/radiation.h"
#include "../../usr/extension/radiation/particle/photon.h"
#include "../../usr/extension/radiation/particle/particle_pool.h"
#include "../../usr/extension/radiation/integrate/generate.h"
#include "../../usr/extension/radiation/integrate/integrator.h"
#include "../../usr/extension/radiation/comm/comm.h"
#include "../../usr/extension/radiation/comm/comm_functions.h"
#include "disk_chem_bdt.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  Central - star photon - packet MCRT for the disk.
//  Physics recipes follow WBG19 ( 4 bands:  7 / 12 / 25 eV
//  and 3 keV );  the TRANSPORT is the photon - packet
//  machinery of usr / extension / radiation ( the planet _
//  atm _ sph pattern ),  NOT the WBG19 / Athena++ per - cell
//  ray records --  those do not suit the GPU.
//
//  Data flow:  the radiation module's blocks carry NO rad
//  storage of their own --  bdt_radalias_t ALIASES the MHD
//  module's bdt_chem_t::rad ( allocated and sized there by
//  the ppd block_yield wrapper ),  so the chemistry solver
//  keeps reading its own block data,  unchanged.   The
//  transport proxy prx_rad_t sources hyd.u from the MHD
//  module the same way ( planet's prx_t assumes a
//  chemistry::hydro::base_t host,  which our MHD module is
//  not ).

using disk_phot_t = extension::radiation::photon_t
    < particle::radiation::photon::sph_t >;
using disk_pol_t  = extension::radiation::pool_t
    < disk_phot_t >;

////////////////////////////////////////////////////////////
//  Pre - absorption shared state ( App A.3,  the LOCAL
//  form ):  the inner -
//  shell column is estimated from the r_in - adjacent cells'
//  CURRENT state ( no radial reduction,  no cross - step
//  memory ):  tau_pre,band = mfpa_band( r_in ) * r_in /
//  1.25 ( the 1.25 from the rho ~ r^-2.25 interior
//  integral ),  and the photons' per - species columns are
//  seeded with n_sp( r_in ) * r_in / 1.25 so the shielding
//  near r_in counts the inner material.   disk_radiation_t
//  d2h - copies the two r_in rows per step into this host -
//  side struct,  and the generator applies the attenuation
//  at injection ( regenerating the packet pool every step
//  when active ).

struct disk_preab_t
{
    struct row_t
    {
        int nth, nphi;
        std::vector< type::float_t  > mfpa;  // ( nth, nphi, n_hv  )
        std::vector< type::float2_t > u;     // ( nth, nphi, n_int )
    };
    std::map< int, row_t > rows;      // by block id_l
    int            on;
    type::float2_t len;               // r_in / 1.25 ( code )
    int            n_hv, n_int, n_hva;

    __host__ disk_preab_t(  )
    {
        on = 0;  len = 0;  n_hv = 0;  n_int = 0;  n_hva = 0;
        return;
    };
};

////////////////////////////////////////////////////////////
//  Link to the MHD module ( set from usr.cpp before
//  mesh.init ).

struct disk_rad_link_t
{
    inline static std::weak_ptr< mesh::mod_base_t > p_mhd;

    static bdt_chem_t & b_mhd
    ( const mesh::region_logic_t & reg, const bool dev )
    {
        auto & dd = p_mhd.lock(  )->data( reg );
        return dynamic_cast< bdt_chem_t & >
               ( dev ? dd.d(  ) : dd.h(  ) );
    };
};

////////////////////////////////////////////////////////////
//  The radiation module's block data:  rad arrays aliased
//  from the MHD module ( setup is called by dual_t::setup
//  AFTER is_on_device is set;  the MHD module's init runs
//  first,  so its rad arrays are already allocated ).

struct bdt_radalias_t : extension::radiation::block_data_t
{
    __host__ virtual void setup
    ( const mesh::f_new_t & ) override
    {
        rad = disk_rad_link_t::b_mhd( reg, is_on_device ).rad;
        return;
    };
    __host__ virtual void free
    ( const mesh::f_free_t & ) override
    {
        return; //  aliased:  dat_3d_t proxies own nothing
    };
    //  copy_input / copy_output / read / write:  inherited
    //  --  they move the CONTENT of the aliased arrays,
    //  which is exactly the intended behavior.
};

////////////////////////////////////////////////////////////
//  The transport proxy:  rad from the ( aliased ) block
//  data,  hyd.u from the MHD module.

struct prx_rad_t
    : particle::radiation::proxy_t< bdt_radalias_t >
{
    ////////// Data //////////
    extension::radiation::rad_t rad;
    struct
    {
        int                              n_hva;
        mesh::dat_3d_t< type::float2_t >     u;
    }   hyd;

    ////////// Functions //////////
    template< class bdt_T > __forceinline__
    static auto d( const bdt_T & bdt )
    {
        const auto & b_d = dynamic_cast
            < const bdt_radalias_t & > ( bdt );
        prx_rad_t res;
        res.rad = b_d.rad;
        return    res;
    };

    __host__ virtual void setup
    ( const mesh::block::dual_t   &      d ,
      const std ::map< int, int > & m_rank ,
      particle  ::base_t          &    mod ) override
    {
        auto & b_d( this->ref( d.d(  ) ) );
        rad = b_d.rad;
        hyd.u    = disk_rad_link_t::b_mhd( b_d.reg, true )
                   .hyd.u;
        //  MHD conserved layout:  rho, ene, mom x3, bcc x3.
        hyd.n_hva = idx::bcc + 3;
        return particle::proxy_base_t::setup
               ( d, m_rank, mod );
    };
};

////////////////////////////////////////////////////////////
//  The generator:  photon packets launched from the r_in
//  face,  one radial ray per ( theta, phi ) sub - division
//  of each face cell.   The base generate_once hardcodes
//  injection on the OUTERMOST blocks along ax_inj;  here
//  the filter honors rhs ( x_in -> innermost ).   The
//  per - ray direction is radial ( dir = x^ ),  so the
//  incidence cosine is exactly 1 and the proper weight is
//  the sin - theta - weighted face - area share.

struct disk_gen_t
    : extension::radiation::gen_t< disk_gen_t >
{
    ////////// Types //////////
    using super_t = extension::radiation::gen_t
        < disk_gen_t >;

    ////////// Data //////////
    int n_ray_2d[ 2 ];  //  sub - rays per face cell ( th, ph )
    //  Pre - absorption ( App A.3,  the LOCAL form --  see
    //  disk_preab_t above ):  per - step attenuation of the
    //  injected packets + column seeding,  from the r_in -
    //  adjacent cells' current state.
    const disk_preab_t * preab_p;

    ////////// Functions //////////
    __host__ disk_gen_t(  )
    {
        n_ray_2d[ 0 ] = 4;
        n_ray_2d[ 1 ] = 4;
        preab_p = nullptr;
        return;
    };

    __host__ virtual void init
    ( const input & args, particle::base_t & mod ) override
    {
        super_t::init( args, mod );
        args( "radiation", "n_ray_2d", n_ray_2d );
        //  preab_p is wired by disk_radiation_t::init ( the
        //  module owns the buffers;  the generator is held
        //  by the driver's p_gen ).
        //  [radiation] lum_ph_cgs:  the photon luminosity
        //  per band [ ph / s ].   When given it overrides
        //  flux_cgs:  the r_in - face photon flux is
        //  Phi / ( 4 pi r_in^2 )  ( the polar caps outside
        //  the domain simply never inject --  their share
        //  of 4 pi is absent from the budget ).
        std::vector< type::float2_t > lum;
        args( "radiation", "lum_ph_cgs", lum );
        if( lum.size(  ) > 0 )
        {
            type::coord2_t x_min;
            args( "mesh", "x_min", x_min );
            const auto r_in  = x_min[ 0 ] * unit.l0;
            const auto r_fac = args.get< type::float2_t >
                ( "radiation", "flux_scale", 1 );
            flux.resize( lum.size(  ) );
            for( size_t n = 0; n < lum.size(  ); ++ n )
                flux[ n ] = lum[ n ]
                    / ( 4 * 3.141592653589793
                        * r_in * r_in )
                    * unit.derive( 0, 2, 1 ) / r_fac;
        }
        return;
    };

    //  set_par:  the base fills lumin = flux x proper and
    //  zeroes the columns;  we then apply the pre -
    //  absorption ( per - band exp( -tau ) and the column
    //  seeds ) from the r_in rows.
    template< class par_T > __host__
    void set_par( par_T & par )
    {
        super_t::set_par( par );
        if( preab_p == nullptr || ! preab_p->on )
            return;
        const auto it = preab_p->rows.find( par.ib_l );
        if( it == preab_p->rows.end(  ) )
            return;
        const auto & row( it->second );
        const int    j = par.i[ 1 ], k = par.i[ 2 ];
        const int    i = j * row.nphi + k;
        for( int n = 0; n < par.n_hv; ++ n )
        {
            const auto tau = row.mfpa[ i * preab_p->n_hv + n ]
                           * preab_p->len;
            par.lumin( n ) *= exp( - tau );
        }
        for( int n = 0; n < par.n_sp; ++ n )
            par.column( n ) = ( type::float_t )
                ( row.u[ i * preab_p->n_int
                + n + preab_p->n_hva ] ) * preab_p->len;
        return;
    };

    template< class par_T, class geo_T, class fun_T >
    __host__ void pre_proc_par
    ( par_T & par, const geo_T & geo,
      particle::base_t & mod, const fun_T & f_cache )
    {
        const int ax_p1 ( ( ax_inj + 1 ) % 3 );
        const int ax_p2 ( ( ax_inj + 2 ) % 3 );

        auto & i( par.i );
        type::coord_t xf,  dx;
        for( int a = 0; a < 3; ++ a )
        {
            xf[ a ] = geo.x_fc( a, i[ a ] );
            dx[ a ] = geo.dx_f( a, i      );
        }
        const auto ds = geo.surface( i, ax_inj );

        //  sin - theta weights across the theta sub - rays
        //  ( the face area element is sin th dth dph ).
        type::float_t s_norm( 0 );
        for( int j = 0; j < n_ray_2d[ 0 ]; ++ j )
        {
            const auto th = xf[ ax_p1 ] + dx[ ax_p1 ]
                          * ( j + 0.5 ) / n_ray_2d[ 0 ];
            s_norm += sin( th );
        }
        for( int j = 0; j < n_ray_2d[ 0 ]; ++ j )
            for( int k = 0; k < n_ray_2d[ 1 ]; ++ k )
            {
                par.x[ ax_p1 ] = xf[ ax_p1 ] + dx[ ax_p1 ]
                    * ( j + 0.5 ) / n_ray_2d[ 0 ];
                par.x[ ax_p2 ] = xf[ ax_p2 ] + dx[ ax_p2 ]
                    * ( k + 0.5 ) / n_ray_2d[ 1 ];
                par.proper = sin( par.x[ ax_p1 ] ) / s_norm
                           * ds / n_ray_2d[ 1 ];
                //  Radial ray:  dir = x^ ( spherical ).
                const auto st = sin( par.x[ 1 ] );
                par.dir[ 0 ] = st * cos( par.x[ 2 ] );
                par.dir[ 1 ] = st * sin( par.x[ 2 ] );
                par.dir[ 2 ] = cos( par.x[ 1 ] );
                par.to_car(  );
                f_cache( par );
            }
        return;
    };

    //  The base generate_once hardcodes the OUTER block
    //  filter ( generate.h:188 );  this copy honors rhs (
    //  x_in -> reg == 0 ).   Everything else is identical
    //  --  keep pace with upstream if generate_once
    //  changes.
    template< class pol_T > __host__ void
    generate_once ( pol_T & pool, particle::base_t & mod )
    {
        if( mod.p_mesh->p_com->is_root(  ) )
            std::cout << "Generating/copying photons once"
                      << " ... "<< std::flush;

        auto & rad  = dynamic_cast
             < ::extension::radiation_t & > ( mod );
        auto & chem = ( * rad .p_che );
        const auto n_hv = rad .hv_eV  .size(  );
        const auto n_sp = chem.species.size(  );

        using par_t = typename pol_T::par_t;
        std::vector< par_t > par_cache;
        const int ax      ( ax_inj ) ;
        const int ax_p1 ( ( ax_inj + 1 ) % 3 );
        const int ax_p2 ( ( ax_inj + 2 ) % 3 );
        for( auto & d : mod )
        {
            const auto & geo( d.h(  ).geo );
            const auto & reg( d.h(  ).reg );
            const auto & reg_max = mod.p_mesh->tree
                       (  ).region_max( reg.level ) ;
            const int reg_edge
                = rhs ? reg_max[ ax_inj ] - 1 : 0;
            if( reg[ ax_inj ] != reg_edge )
                continue;

            type::idx_t i_s( { 0, 0, 0 } );
            type::idx_t i_e( { 0, 0, 0 } );

            i_s[ ax    ] = rhs ? geo.n_ceff[ ax ] - 1 : 0;
            i_e[ ax    ] = i_s[ ax ] + 1;
            i_e[ ax_p1 ] = geo.n_ceff[ ax_p1 ];
            i_e[ ax_p2 ] = geo.n_ceff[ ax_p2 ];

            par_t       par;
            par.n_hv = n_hv;
            par.n_sp = n_sp;
            par.dest.todo  = particle::to_keep;
            for( int a = 0; a < 3; ++ a )
                par.dir[ a ]  = dir0[ a ];

            par.ib_l = d.b_w.get(  ).id_l;
            if( rhs )
                par.x[ ax ] = geo.x_fc( ax, i_e[ ax ] )
                            - geo.dx_f( ax, i_e ) * 1e-2;
            else
                par.x[ ax ] = geo.x_fc( ax, i_s[ ax ] )
                            + geo.dx_f( ax, i_s ) * 1e-2;
            for( auto i : utils::loop_t( i_s, i_e ) )
            {
                for( int a = 0; a < 3; ++ a )
                    par.i[ a ] = i[ a ];
                this->pre_proc_par
                ( par, geo, mod, [ & ] ( auto & p )
                {
                    p.id = par_cache.size (   );
                    par_cache . push_back ( p );
                }   );
            }
        }
        n_par = par_cache.size(  );
        pol_T pool_h;
        pool  .set_n_int( n_hv, n_sp );
        pool_h.set_n_int( n_hv, n_sp );
        pool_h.reserve_host( mod,  n_par + 1 );
        auto & par_tmp     ( pool_h[ n_par ] ); // tmp var
        par_tmp.n_hv =    n_hv;
        par_tmp.n_sp =    n_sp;
        par_tmp.set_offset(  );

        auto & f_com( * mod.p_mesh->p_com );
        const auto id_offset
            = this->set_id_offset( f_com );

        for( size_t i = 0; i < n_par; ++ i )
        {
            par_cache.at( i ).cp_shallow( par_tmp );
            this->set_par   ( par_tmp );
            par_tmp.id += id_offset;
            par_tmp.save( pool_h [ i ] );
        }

        pool.set_mem( mod, n_par );
        s_data = pool.size   (   );
        data_h = &  pool_h   [ 0 ];
        auto n_par_global( n_par );
        f_com.reduce_all ( n_par_global, comm::sum );
        if( mod.p_mesh->p_com->is_root(  ) )
            std::cout << " Done with " << n_par_global
                      << " photons. "<< std::endl;
        return;
    };

    //  The base generate() statically binds the BASE
    //  generate_once;  re - issue it here so ours is used.
    //  With preabs on,  regenerate every step ( the pre -
    //  absorption tracks the current r_in state ).
    template< class pol_T, class map_T > __host__
    void generate ( pol_T & pool, const map_T & bmap,
                    particle::base_t  & mod )
    {
        if( data_h == nullptr
            || ( preab_p != nullptr && preab_p->on ) )
            generate_once( pool, mod );
        pool.n_par = n_par;
        return mod.p_dev->a_cp
             ( & pool[ 0 ], data_h, s_data, mod.stream );
    };
};

////////////////////////////////////////////////////////////
//  The module.

class disk_radiation_t : public extension::radiation_t
{
public:                         // Data
    //  Pre - absorption buffers ( see disk_preab_t above );
    //  public for the generator's access through p_gen.
    disk_preab_t preab;
public:
    virtual void read( const input & args ) override
    {
        is_first_run = true;
        particle::radiation::base_t::read( args );
        args( "radiation", "hv_eV", hv_eV );
        n_hv = hv_eV.size(  );
        enroll_extension< prx_rad_t, disk_pol_t, disk_gen_t,
            extension::radiation::itg_t >(  );
        preab.on = args.get< int >( "radiation", "preabs", 1 );
        return;
    };

    virtual void init
    ( const input & args, mesh::mesh_t & mesh ) override
    {
        extension::radiation_t::init( args, mesh );
        //  Wire the generator to our pre - absorption
        //  buffers ( the driver holds it via p_gen ).
        auto & g = dynamic_cast< disk_gen_t & >( * p_gen );
        g.preab_p = & preab;
        if( ! preab.on )
            return;
        type::coord2_t x_min;
        args( "mesh", "x_min", x_min );
        preab.len   = x_min[ 0 ] / 1.25;   //  code units
        preab.n_hv  = n_hv;
        preab.n_hva = idx::bcc + 3;        //  the MHD layout
        //  the conserved - u slot count comes from the MHD
        //  module's block data at the first step ( rows are
        //  ( re )sized there ).
        return;
    };

    //  d2h the r_in rows ( mfpa_i per band,  u per slot )
    //  of every inner - edge block into the pre - absorption
    //  buffers.   One - step stale by construction ( the
    //  WBG19 App - A.3 timing:  the previous step's state
    //  attenuates the entering photons ) --  EXCEPT at the
    //  first cycle:  mfpa_i is zeroed by init_cond and only
    //  filled by chem_t::prep_rad inside the base step,  so
    //  without an explicit prep_rad here the very first
    //  injection would be completely unattenuated ( the
    //  step - 0 flash ).
    virtual void step( mesh::mesh_t & mesh ) override
    {
        if( preab.on )
        {
            if( is_first_run )
                p_che->prep_rad( mesh );
            for( auto & d : ( * this ) )
            {
                const auto & reg( d.d(  ).reg );
                const auto & reg_max = mesh.tree(  )
                    .region_max( reg.level );
                if( reg[ 0 ] != 0 )
                    continue;
                //  The MHD block's device data ( our own
                //  rad arrays alias it ).
                auto & b_m = disk_rad_link_t::b_mhd
                             ( reg, true );
                const auto & geo( b_m.geo );
                const int nt = geo.n_ceff[ 1 ];
                const int np = geo.n_ceff[ 2 ];
                auto & row( preab.rows[ d.b_w.get(  )
                            .id_l ] );
                if( row.mfpa.size(  ) == 0 )
                {
                    row.nth  = nt;
                    row.nphi = np;
                    row.mfpa.resize( nt * np * preab.n_hv );
                    row.u   .resize( nt * np
                                     * b_m.hyd.u.n_int );
                    preab.n_int = b_m.hyd.u.n_int;
                }
                //  The i0 = 0 row,  ghost - aware offsets.
                const auto & mf( b_m.rad.mfpa_i );
                const auto & uu( b_m.hyd.u );
                const int  no0 = mf.n_gh[ 0 ];
                const int  no1 = mf.n_gh[ 1 ];
                const int  no2 = mf.n_gh[ 2 ];
                const int  uo0 = uu.n_gh[ 0 ];
                const int  uo1 = uu.n_gh[ 1 ];
                const int  uo2 = uu.n_gh[ 2 ];
                //  Per - ( j, k ) row d2h copies ( a row
                //  is contiguous only along n_int;  1024
                //  tiny copies per step --  negligible next
                //  to the transport;  a gather kernel is
                //  the noted optimization if this ever
                //  shows ).
                for( int j = 0; j < nt; ++ j )
                    for( int k = 0; k < np; ++ k )
                    {
                        const int im = ( ( 0 + no0 )
                            + mf.n_cell[ 0 ] * ( ( j + no1 )
                            + mf.n_cell[ 1 ] * ( k + no2 ) ) )
                            * preab.n_hv;
                        p_dev->f_cp( & row.mfpa[ ( j * np + k )
                            * preab.n_hv ], mf.dat + im,
                            preab.n_hv * sizeof
                            ( type::float_t ) );
                        const int iu = ( ( 0 + uo0 )
                            + uu.n_cell[ 0 ] * ( ( j + uo1 )
                            + uu.n_cell[ 1 ] * ( k + uo2 ) ) )
                            * uu.n_int;
                        p_dev->f_cp( & row.u[ ( j * np + k )
                            * uu.n_int ], uu.dat + iu,
                            uu.n_int * sizeof
                            ( type::float2_t ) );
                    }
            }
        }
        return extension::radiation_t::step( mesh );
    };
};

};                              // namespace interm
