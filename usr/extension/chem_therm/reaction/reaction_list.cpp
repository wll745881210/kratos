#include "../chemistry.h"
#include "reaction_list.h"
#include <iomanip>

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
//

__host__ void list_t::init_mol_cool
( const input & args, ::chemistry::base_t & chem )
{
    input argm( args.get< std::string >
              ( "chemistry", "mol_cooling_file", "" ) );
    argm.set_comm( chem.p_mesh->p_com );
    if( ! argm.read(  ) )
    {
        if( chem.p_mesh->p_com->is_root (  ) )
            std::cout << "Skipping molecular cooling\n";
        return;
    }
    std::vector < mol_cool_t > mol_cool_h;
    for( const auto & r : argm.get_prefixes(  ) )
        if( ! mol_cool_h.emplace_back(  ).parse
            ( r, argm, chem ) )
            mol_cool_h.pop_back(  );
    for( auto & mc : mol_cool_h )
        mc.i_cool = ( n_cool ++ );
    mol_cool.init( mol_cool_h.size (  ) );
    for( size_t i = 0; i < mol_cool_h.size(  ); ++ i )
        mol_cool[ i ] = mol_cool_h.at( i );
    return;
}

__host__ void list_t:: init_atom_cool
( const  input & args, ::chemistry::base_t & chem )
{
    input arga( args.get< std::string >
              ( "chemistry", "atom_cooling_file", "" ) );
    arga.set_comm( chem.p_mesh->p_com );
    if( ! arga.read(  ) )
    {
        if( chem.p_mesh->p_com->is_root (  ) )
            std::cout << "Skipping atomic cooling\n";
        return;
    }
    std::vector < atom_cool_t > atom_cool_h;
    for( const auto & r : arga.get_prefixes(  ) )
        if( ! atom_cool_h.emplace_back(  ).parse
            ( r, arga, chem ) )
            atom_cool_h.pop_back(  );
    for( auto & ac : atom_cool_h )
         ac.i_cool = ( n_cool ++ );
    atom_cool.init( atom_cool_h.size (   ) );
    for( size_t i = 0; i < atom_cool_h.size(  ); ++ i )
        atom_cool[ i ] = atom_cool_h.at( i );
    return;
}

__host__ void list_t:: init_phot_reac
( const  input & args, ::chemistry::base_t & chem )
{
    const type::float2_t l0 ( chem.p_eos->unit.l0 );
    const type::float2_t t0 ( chem.p_eos->unit.t0 );
    flx_conv = args.get< type ::float2_t >
             ( "radiation", "flux_scale", 1 )
             / ( pow( l0, 2 ) * t0 );
    if( std::isinf( flx_conv ) || std::isnan( flx_conv ) ||
        fabs( flx_conv ) < type::float_min )
        throw std::runtime_error
            ( "Use a different [radiation] flux_scale" );
    
    l_ph_shield = 0;
    input parg( args.get< std::string >
              ( "chemistry", "photo_reac_file", "" ) );
    parg.set_comm( chem.p_mesh->p_com );
    if( ! parg.read(  ) )
    {
        if( chem.p_mesh->p_com->is_root (  ) )
            std::cout << "Skipping photon cooling\n";
        return;
    }
    std::vector < phot_reac_t > phot_h;
    for( const auto & r : parg.get_prefixes(  ) )
        if( ! phot_h.emplace_back(  ).parse_ph
            ( * this, r, parg, chem ) )
            phot_h.pop_back(  );
    int i( 0 );
    for( auto & pr : phot_h )
         pr.i_phot = ( i ++ );
    phot_reac.init( phot_h.size (  ) );
    for( size_t i = 0; i < phot_h.size(  ); ++ i )
        phot_reac[ i ] = phot_h.at ( i );

    std::vector< type::float_t > hv_erg;
    args( "radiation", "hv_eV",  hv_erg );
    for( auto & hv : hv_erg )
        hv *= phys::cgs::erg_eV;
    n_hv    = hv_erg.size(  );
    this->hv_erg = chem.p_dev->malloc_const
            < type::float_t >( n_hv );
    chem.p_dev->f_cc( this->hv_erg, hv_erg.data(  ),
                      sizeof( type::float_t ) * n_hv );
    return;
}

__host__ void list_t:: init_dust_reac
( const  input & args, ::chemistry::base_t & chem )
{   // Gr(+/-/0) + X(?) -> Gr(?) + X(?) : in standard reac
    // Gr(0/-) + hv -> Gr(+/0) + e- : in photon reac
    // Here are the very dust-grain specific reactions.
    
    std::vector< dust_reac_t > dust_h;

    // Here only cares about thermal accomodation...
    if( args.get< bool >( "chemistry", "dust_thermal", 1 ) )
        for( const auto & sp : chem.species )
            if( ! dust_h.emplace_back(  ) .
                parse_thermal( * this, sp, chem ) )
                dust_h.pop_back(  );

    // ... and catalyses and sublimation
    input darg( args.get< std::string >
              ( "chemistry", "dust_reac_file", "" ) );
    if ( darg.read(  ) )
        for ( const auto & r : darg.get_prefixes(  ) )
        {
            const auto rate_type = darg.get< std::string >
                ( r, "rate_type", "surface" );
            if ( rate_type == "sublimation" )
            {
                if ( ! dust_h.emplace_back(  ).
                     parse_sublimation
                    ( * this, r, darg, chem ) )
                    dust_h.pop_back(  );
            }
            else
            {
                if ( ! dust_h.emplace_back(  ).parse_gr
                    ( * this, r, darg, chem ) )
                    dust_h.pop_back(  );
            }
        }

    dust_reac.init( dust_h.size (  ) );
    for( size_t i = 0; i < dust_h.size(  ); ++ i )
        dust_reac[ i ] = dust_h.at ( i );    
    return;
}

__host__ void list_t::init_extra
( const input & args, ::chemistry::base_t & chem )
{
    auto & chem_ext
         = dynamic_cast< const chem_t & >( chem );
    // Compute geometric cross-section from chem_t member.
    // For backward compatibility:
    // sig_dust_cgs = pi * r_dust^2  (whole-grain).
    sig_dust_cgs = 3.1415927f
        * chem_ext.r_dust_cgs * chem_ext.r_dust_cgs;
    h0_col_cgs = args.get< type::float_t >
        ( "chemistry", "h0_col_cgs",  -1 );
    col_ratio  = args.get< type::float_t >
        ( "chemistry", "col_ratio",    1 );
    T_eq       = args.get< type::float_t >
        ( "chemistry", "T_eq",         1 );

    fac_dust_stiff = args.get< type::float_t >
       ( "chemistry", "fac_dust_stiff", -1e3 );

    n_cool = 0;
    i_elec = chem.species.exist( "e-" ) 
           ? chem.species.at   ( "e-" ) : -1;
    return;
}

__host__ void list_t::init
( const input & args, ::chemistry::base_t & chem )
{
    std::cout << std::setprecision( 5 );
    super_t::init ( args, chem );
    init_mol_cool ( args, chem );
    init_atom_cool( args, chem );
    init_phot_reac( args, chem );
    init_dust_reac( args, chem );
    if( args.get< bool >( "chemistry", "print", true ) )
        if( chem.p_mesh->p_com->is_root(  ) )
            print_extra (   chem );

    const auto on_const = args.get< bool >
           ( "chemistry", "reac_on_const",  true );    
    mol_cool .regularize( * chem.p_dev, on_const );
    atom_cool.regularize( * chem.p_dev, on_const );
    phot_reac.regularize( * chem.p_dev, on_const );
    dust_reac.regularize( * chem.p_dev, on_const );    
 
    l_phot_reac =   reac.size + phot_reac.size;
    l_dust_reac = l_phot_reac + dust_reac.size;
    l_atom_cool = l_dust_reac + atom_cool.size;
    l_mole_cool = l_atom_cool +  mol_cool.size;
    return;
}

////////////////////////////////////////////////////////////
// Print the lists

__host__ void list_t::print_extra
( const ::chemistry::base_t & chem )
{
    const auto & sps( chem.species );
    std::cout << "=======================\n# of photon "
              << "reac: " << phot_reac.size << '\n';
    for( size_t i = 0; i < phot_reac.size; ++ i )
        phot_reac[ i ].print( sps );
    std::cout << "=====================\n# molecular"
              << " cooling: " << mol_cool.size << '\n';
    for( size_t i = 0; i < mol_cool.size; ++ i )
        std::cout << sps.at( mol_cool[ i ].i_chem[ 0 ] )
            .name << ( i < mol_cool.size - 1 ? ", " : "" ) ;
    std::cout << "\n-=-=-=-=-=-=-=-=-=-=-\n# atomic    "
              << "cooling: "<< atom_cool.size << "\npars = "
              << "[ A, gu/gl, E/kb, n_crit_n, n_crit_e ]\n";
    for( size_t i = 0; i < atom_cool.size; ++ i )
        atom_cool[ i ].print( sps );
    std::cout << "=======================\n# of dust "
              << "reac: " << dust_reac.size << '\n';
    for( size_t i = 0; i < dust_reac.size; ++ i )
        dust_reac[ i ].print( sps );    
    std::cout << "=====================\n" << std::endl;
    return;
}

};                    // namespace extension::chemistry
