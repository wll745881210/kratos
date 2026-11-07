#pragma once

#include "../../src/modules/chemistry/chemistry.h"

namespace interm
{
////////////////////////////////////////////////////////////
//  The saturation region:  cells with
//  rho < kappa * rho_sat( r ),  rho_sat( r ) = rho_sat0 *
//  ( r / r0 )^q,  are the strongly - irradiated,  low -
//  opacity,  floor - adjacent region --  physically the
//  protostellar jet,  where the full chemistry is both
//  meaningless and the dominant solver cost.   There we
//  ( a ) pin the composition to the saturated photo - end
//  state ( every molecule dissociated,  every atom singly
//  ionized,  Gr -> Gr+ ) and ( b ) SKIP the chemistry
//  solver entirely ( the bdt_chem_t::chem_skip flag,  read
//  by the extension solver's cell loop ).   No mass is
//  injected ( unlike a density floor ):  rho is untouched,
//  only the composition is redistributed element -
//  conservingly,  and the region's temperature is left to
//  the hydro.

struct disk_sat_t
{
    ////////// Constants //////////
    //  Tracked elements and their saturated target species:
    //  H -> H+,  He -> He+,  C -> C+,  O -> O+,  S -> S+,
    //  Si -> Si+,  Gr -> Gr+,  and e- carries the charge.
    static constexpr int n_trk = 7;   //  excl. e-
    static constexpr int n_max = 48;  //  species slots cap

    ////////// Data //////////
    int         on;
    type::float_t rho_sat0;   //  code units,  at r0
    type::float_t      q;
    type::float_t  kappa;
    type::float_t     r0;
    int            n_sp;
    //  The suppression rim ( [suppress] ):  cells within
    //  supp_nr of r_in or supp_nt of EITHER theta edge are
    //  chem - skipped WITHOUT the composition rewrite (
    //  keeping their Gr content keeps mfpa_i -- and thus
    //  the pre - absorption -- intact;  the hydro side
    //  freezes them in disk_intg.h ).
    int         supp_nr;
    int         supp_nt;
    //  The relaxation ring ( [ring] r_out ):  cells with
    //  r < ring_r are chem - skipped WITHOUT the rewrite,
    //  too --  the ring relaxation owns their state,  and
    //  the direct - FUV photoionization of the inner cells is
    //  cut at the source.
    type::float_t ring_r;
    //  Theta - edge chem - skip band ( [chem] skip_nt ),
    //  independent of [suppress] / [ring]:  the boundary -
    //  adjacent radiation baking lives there.
    int         skip_nt;
    //  [chemistry] skip_T_hi / skip_T_lo ( K ):  chem - skip
    //  cells OUTSIDE the thermal window --  the forensics
    //  probes for the stiff - cell localization:  run one
    //  probe with skip_T_hi = 2e3 ( skip the cold cells )
    //  and one with skip_T_lo = 2e3 ( skip the hot ones );
    //  whichever recovers the speed owns the cost.
    //  The mask reads the THERMAL part of ene.
    type::float_t skip_cs2_hi;
    type::float_t skip_cs2_lo;
    int    i_tgt[ n_trk + 1 ];        //  slot of each target
    //  el_cnt[ i ][ k ]:  atoms of tracked element k in
    //  species i  ( k = n_trk is the CHARGE content )
    int    el_cnt[ n_max ][ n_trk + 1 ];
    //  mcgs[ i ]:  species masses in grams ( the mass
    //  normalization below ).   The u slots carry
    //  n_i * m_A / rho0,  so sum_i u_i * mcgs_i / m_A
    //  must equal rho  ( the species mass content of the
    //  cell );  the polar - funnel chem - skip cells have
    //  no solver to enforce that,  and the boundary fluxes
    //  accumulate a net e- / H+ mismatch instead --  an
    //  exponential drift that overflows float32 and gives
    //  Inf - Inf = NaN in the PLM reconstruction.   Every
    //  chem - skip cell is therefore renormalized back onto
    //  its rho here.
    type::float_t mcgs[ n_max ];

    ////////// Functions //////////
    __host__ disk_sat_t(  )
    {
        on = 0;   rho_sat0 = 0;   q = 0;   kappa = 0;
        r0 = 1;   n_sp = 0;  supp_nr = 0;   supp_nt = 0;
        ring_r = 0;   skip_nt = 0;
        skip_cs2_hi = 0;   skip_cs2_lo = 0;
        return;
    };

    //  Build the element -> target mapping from the
    //  chemistry module's species list.   Throws if a
    //  species carries an element outside the tracked set.
    __host__ void read
    ( const input & args, const ::chemistry::base_t & chem )
    {
        phys::unit_t< type::float2_t > unit;
        unit.init( args );
        on = args.get< int >( "sat", "on", 1 );
        //  rho_sat0 is given in H nuclei per cc ( WBG19's
        //  floor was 5e7 r^-3 );  convert to code density:
        //  rho_code = n_H * 1.4 * m_A / rho0,  rho0 = m_A.
        rho_sat0 = args.get< type::float2_t >
                   ( "sat", "rho_sat0_cgs", 5e7 ) * 1.4;
        q        = args.get< type::float2_t >
                   ( "sat", "rho_sat_pow", -3.0 );
        kappa    = args.get< type::float2_t >
                   ( "sat", "rho_sat_fac",  1.0 );
        r0       = args.get< type::float2_t >
                   ( "profile", "r0",       1.0 );
        supp_nr  = args.get< int >( "suppress", "n_r_in", 0 );
        supp_nt  = args.get< int >( "suppress", "n_th",   0 );
        ring_r   = args.get< type::float_t >
                         ( "ring", "r_out", 0 );
        skip_nt  = args.get< int >
                         ( "chemistry", "skip_nt", 0 );
        //  Thermal - window skips,  in K --  converted with
        //  the profile 's mean molecular weight.
        const auto mu_ma = args.get< type::float_t >
            ( "profile", "mu_ma", 2.34 );
        const auto cs2_per_K = phys::cgs::kb
            / ( mu_ma * phys::cgs::ma )
            / ( unit.vel0 * unit.vel0 );
        skip_cs2_hi = args.get< type::float_t >
            ( "chemistry", "skip_T_hi", 0 ) * cs2_per_K;
        skip_cs2_lo = args.get< type::float_t >
            ( "chemistry", "skip_T_lo", 0 ) * cs2_per_K;
        //  The ring chem - skip only makes sense when the
        //  ring owns the full state ( mode 0 );  with
        //  mode = 1 ( velocity - only relaxation )  the
        //  chemistry must run.
        if( args.get< int >( "ring", "mode", 0 ) != 0 )
            ring_r = 0;
        if( ! on )
            return;
        return build( chem );
    };

    //  The species - table part,  separated so the pgen can
    //  set the pars directly and build lazily at the first
    //  init_cond ( module init_cond runs BEFORE init,  but
    //  after every module's read --  the species table is
    //  only guaranteed to exist then ).
    __host__ void build( const ::chemistry::base_t & chem )
    {
        const auto & sps( chem.species );
        n_sp = sps.size(  );
        if( n_sp > n_max )
            throw std::runtime_error
                ( "disk_sat_t:  too many species" );

        //  Tracked element z codes:  H He C O S Si Gr( -1 )
        //  and the charge element ( z = 0 ).
        const int z_trk[ n_trk + 1 ]
            = { 1, 2, 6, 8, 16, 14, -1, 0 };
        const char * tgt[ n_trk + 1 ]
            = { "H+","He+","C+","O+","S+","Si+","Gr+","e-" };
        for( int k = 0; k <= n_trk; ++ k )
        {
            i_tgt[ k ] = sps.exist( tgt[ k ] )
                       ? sps.at( tgt[ k ] ) : -1;
            if( i_tgt[ k ] < 0 && k < n_trk )
                throw std::runtime_error
                    ( std::string( "disk_sat_t:  species " )
                      + tgt[ k ] + " missing" );
        }
        for( int i = 0; i < n_sp; ++ i )
            for( int k = 0; k <= n_trk; ++ k )
            {
                const auto & el( sps[ i ].elem );
                el_cnt[ i ][ k ] = el.count( z_trk[ k ] )
                    ? el.at( z_trk[ k ] ) : 0;
                //  Guard:  no untracked element allowed
                //  ( e.g. a future N - bearing species ).
            }
        for( int i = 0; i < n_sp; ++ i )
            for( const auto & [ z, cnt ] : sps[ i ].elem )
            {
                bool known( false );
                for( int k = 0; k <= n_trk; ++ k )
                    known = known || z == z_trk[ k ];
                if( ! known )
                    throw std::runtime_error
                        ( "disk_sat_t:  species "
                          + sps[ i ].name
                          + " carries an untracked element" );
            }
        for( int i = 0; i < n_sp; ++ i )
            mcgs[ i ] = sps[ i ].mass_cgs;
        return;
    };

    //  The criterion ( shared by the device kernel and the
    //  IC - time corona ionization in the pgen ).
    template< class f_T, class g_T > __host__ __device__
    __forceinline__ bool masks
    ( const f_T rho, const g_T r ) const
    {
        return on
            && rho < kappa * rho_sat0
                          * pow( r / r0, q );
    };

    //  The saturated composition:  element totals ( in  y~
    //  units )  redistributed onto the targets;  everything
    //  else to zero.   Shared by the runtime kernel and the
    //  pgen's IC ( the ionized - corona option ).
    template< class f_T > __host__ __device__
    __forceinline__ void set_sat_comp( f_T * sp ) const
    {
        type::float_t tot[ n_trk + 1 ];
        for( int k = 0; k <= n_trk; ++ k )
            tot[ k ] = 0;
        for( int s = 0; s < n_sp; ++ s )
            for( int k = 0; k <= n_trk; ++ k )
                tot[ k ] += sp[ s ] * el_cnt[ s ][ k ];
        for( int s = 0; s < n_sp; ++ s )
            sp[ s ] = 0;
        for( int k = 0; k < n_trk; ++ k )
            if( i_tgt[ k ] >= 0 )
                sp[ i_tgt[ k ] ] = tot[ k ];
        //  e- carries the total positive charge.
        if( i_tgt[ n_trk ] >= 0 )
        {
            type::float_t ne( 0 );
            for( int k = 0; k < n_trk; ++ k )
                ne += tot[ k ];
            sp[ i_tgt[ n_trk ] ] = ne;
        }
        return;
    };

    //  The device - side per - cell application.
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ void apply
    ( const bdt_T & b_d, const idx_T & i ) const
    {
        auto * sk  = b_d.chem_skip.at( i );
        //  The suppression rim:  chem - skip only,  the
        //  composition stays ( Gr must keep absorbing for
        //  the pre - absorption ).
        if( i[ 0 ] < supp_nr || i[ 1 ] < supp_nt
            || i[ 1 ] >= b_d.geo.n_ceff[ 1 ] - supp_nt )
        {
            sk[ 0 ] = 1;
            return norm_sp( b_d, i );
        }
        //  The theta - edge band ( [chem] skip_nt ):
        //  chem - skip only,  INDEPENDENT of the hydro
        //  freeze --  the boundary - adjacent FUV baking
        //  otherwise drives the fast hot funnel flow that
        //  caps dt.
        if( i[ 1 ] < skip_nt
            || i[ 1 ] >= b_d.geo.n_ceff[ 1 ] - skip_nt )
        {
            sk[ 0 ] = 1;
            return norm_sp( b_d, i );
        }
        const auto r = b_d.geo.x_cc( 0, i[ 0 ] );
        //  The relaxation ring:  chem - skip only ( the ring
        //  relaxation owns the state there ).
        if( ring_r > 0 && r < ring_r )
        {
            sk[ 0 ] = 1;
            return norm_sp( b_d, i );
        }
        auto * u   = b_d.hyd.u.at( i );
        const auto rho = u[ idx::rho ];
        if( skip_cs2_hi > 0 || skip_cs2_lo > 0 )
        {
            //  The thermal - window forensics skip:  cs2
            //  from the THERMAL part of the conserved
            //  energy ( ke and magnetic removed ).
            const auto m2 = u[ idx::mom ] * u[ idx::mom ]
                + u[ idx::mom + 1 ] * u[ idx::mom + 1 ]
                + u[ idx::mom + 2 ] * u[ idx::mom + 2 ];
            const auto b2 = u[ idx::bcc ] * u[ idx::bcc ]
                + u[ idx::bcc + 1 ] * u[ idx::bcc + 1 ]
                + u[ idx::bcc + 2 ] * u[ idx::bcc + 2 ];
            const auto eth = u[ idx::ene ]
                - 0.5 * m2 / __max1( rho, 1e-30f )
                - 0.5 * b2;
            const auto cs2 = 0.6667f * eth
                / __max1( rho, 1e-30f );
            if( ( skip_cs2_hi > 0 && cs2 > skip_cs2_hi )
                || ( skip_cs2_lo > 0 && cs2 < skip_cs2_lo ) )
            {
                sk[ 0 ] = 1;
                return norm_sp( b_d, i );
            }
        }
        const bool m = masks( rho, r );
        sk[ 0 ] = m ? 1 : 0;
        if( ! m )
            return;
        //  rho and the energy untouched.   The saturated
        //  composition is MASS - normalized onto the cell's
        //  rho:  the bare set_sat_comp redistribution leaves
        //  the species mass content inconsistent with rho by a
        //  fixed ratio each step,  and the load - clamped
        //  renormalization cycle in the solver then acts as a
        //  per - step amplifier ( which overflows float32 ).
        auto * sp = u + ( idx::bcc + 3 );
        set_sat_comp( sp );
        return norm_mass( sp, rho );
    };

    //  Rescale the species onto the cell's mass content:
    //  sum_i sp_i * mcgs_i = rho * m_A  ( the u slots carry
    //  n_i * m_A / rho0 with rho0 = m_A,  so the species
    //  mass sum in code units is sum_i sp_i * mcgs_i / m_A
    //  = rho ).   Called right after every composition
    //  rewrite ( set_sat_comp / the skip - cell fallback ).
    template< class f_T >
    __device__ __forceinline__ void norm_mass
    ( f_T * sp, const type::float_t & rho ) const
    {
        type::float_t g( 0 );
        for( int s = 0; s < n_sp; ++ s )
            g += sp[ s ] * mcgs[ s ];
        if( !( g > 0 ) )
            return;
        const auto rat = rho * type::float_t
                             ( phys::cgs::ma ) / g;
        for( int s = 0; s < n_sp; ++ s )
            sp[ s ] *= rat;
        return;
    };

    //  The chem - skip cells' mass renormalization ( see
    //  mcgs above ).   The species mass content is rescaled
    //  onto the cell's rho;  a non - finite input ( the
    //  drift's overflow stage ) falls back to the saturated
    //  composition.   Element ratios are preserved by the
    //  pure rescale;  only the overall magnitude is pinned.
    template< class bdt_T, class idx_T >
    __device__ __forceinline__ void norm_sp
    ( const bdt_T & b_d, const idx_T & i ) const
    {
        if( n_sp < 1 )
            return;
        auto * u  = b_d.hyd.u.at( i );
        auto * sp = u + ( idx::bcc + 3 );
        type::float_t g( 0 );
        bool bad( false );
        for( int s = 0; s < n_sp; ++ s )
        {
            const auto v( sp[ s ] );
            if( __isnan( v ) || __isinf( v ) )
                bad = true;
            g += v * mcgs[ s ];
        }
        const auto rho = __max1( u[ idx::rho ], 0 );
        if( bad || !( g > 0 )
            || g * type::float_t( phys::cgs::ma )
               > rho * type::float_t( 1e30 ) )
        {
            set_sat_comp( sp );
            return norm_mass( sp, rho );
        }
        //  u slots:  n_i * m_A / rho0;  rho in the same
        //  code units with rho0 = m_A  ->  target g = rho.
        const auto rat = rho / g;
        for( int s = 0; s < n_sp; ++ s )
            sp[ s ] *= rat;
        return;
    };
};

////////////////////////////////////////////////////////////
//  Thin per - cell kernel.

template< class bdt_T >
__global__ void disk_sat_k
( const bdt_T b_d, const disk_sat_t sat )
{
    const auto & geo = b_d.geo;
    const int    n0  = geo.n_ceff[ 0 ];
    const int    n1  = geo.n_ceff[ 1 ];
    const int    t   = blockIdx.x * blockDim.x
                     + threadIdx.x;
    if( t >= n0 * n1 * geo.n_ceff[ 2 ] )
        return;
    type::idx_t i;
    i[ 2 ] =   t / ( n0 * n1 );
    i[ 1 ] = ( t / n0 ) % n1;
    i[ 0 ] =   t % n0;
    return sat.apply( b_d, i );
};

};                              // namespace interm
