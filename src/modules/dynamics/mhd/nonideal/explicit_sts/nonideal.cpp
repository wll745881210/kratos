////////////////////////////////////////////////////////////
//  Default - mode instantiation of the non - ideal module
//  ( the dynamics/mhd/mhd.cpp and dynamics/hydro/hydro.cpp
//  convention: every module ships one .cpp , and the
//  DEFAULT
//  configuration --  the one a pgen gets when it names the
//  module without overriding anything --  is compiled ONCE
//  here instead of inside every pgen 's single giant TU ).
//
//  What is instantiated and why:
//
//    ni_t<>            = ni_t< dynamics::mhd::integrator::rk2_t<>,
//                              const_eta_t >
//      the DEFAULT explicit / split non - ideal integrator,
//      exactly the type the header 's enroll recipe names
//      ( unified with the EOS ' own default,  ni_eos_t<> ).
//  Its non - template members ( read / enroll / diffuse /
//  the split - mode dt cap ) are what the instantiation
//      compiles;  the members that ARE templates
//      ( calc_flx / intg_flx ,  keyed on the block data )
//      stay in the pgen,  as they must.
//
//    ni_eos_t<>        = ni_eos_t< const_eta_t >
//      the matching EOS wrapper  ( the parabolic dt cap ).
//
//  ------------------------------------------------------------
//  The implicit curl - curl leg is not instantiated: it has no
//  driver,  no proxy statics and no supporting headers in this
//  build.  [nonideal] mg_cc > 0 throws in ni_t::init rather
//  than being silently ignored -- ignoring it would change the
//  physics without saying so.
//  ------------------------------------------------------------
//
//  NOTE the include ORDER.   ni_t 's  init / diffuse  take
//  dynamics::base_t  BY REFERENCE and touch its members
//  ( mod.p_dev / mod.p_bdk ), so the base must be a COMPLETE
//  type here --  i.e. the dynamics module header,  not just
//  the non - ideal one. Without ../mhd.h the instantiation
//  dies with a wall of " incomplete type dynamics::base_t "
//  errors that then CASCADE into bogus  " no member read "
//  reports from the EOS chain.
#include "../../../../../../src/modules/dynamics/mhd/mhd.h"
#include "nonideal.h"

namespace mhd_ee::ni
{
using type::float_t;


////////////////////////////////////////////////////////////
//  The eta-policy readers.
//
//  Each policy struct names its knobs in its own `read`
//  ( the module's `[nonideal]` section ),  and the one that
//  a pgen enrolls decides which are live.  `eval` stays in
//  the header:  it is `__host__ __device__` and the kernels
//  evaluate it.
//
__host__ void const_eta_t::read( const input & args )
{
    eta_o = args.get< type::float_t >
                ( "nonideal", "eta_o", 0 );
    eta_h = args.get< type::float_t >
                ( "nonideal", "eta_h", 0 );
    eta_a = args.get< type::float_t >
                ( "nonideal", "eta_a", 0 );
    on    = ( eta_o > 0 ) || ( eta_h > 0 )
         || ( eta_a > 0 );
    return;
};

__host__ void hall_len_eta_t::read( const input & args )
{
    eta_o  = args.get< type::float_t >
                 ( "nonideal", "eta_o",   0 );
    eta_h  = args.get< type::float_t >
                 ( "nonideal", "eta_h",   0 );
    eta_a  = args.get< type::float_t >
                 ( "nonideal", "eta_a",   0 );
    hall_l = args.get< type::float_t >
                 ( "nonideal", "hall_l",  0 );
    on     = ( eta_o > 0 ) || ( eta_h > 0 )
          || ( eta_a > 0 ) || ( hall_l > 0 );
    return;
};

__host__ void prof_eta_t::read( const input & args )
{
    eta_o   = args.get< type::float_t >
                  ( "nonideal", "prof_eta_o",   0 );
    eta_a   = args.get< type::float_t >
                  ( "nonideal", "prof_eta_a",   0 );
    rho_ref = args.get< type::float_t >
                  ( "nonideal", "prof_rho_ref", 1 );
    alpha_o = args.get< type::float_t >
                  ( "nonideal", "prof_alpha_o", 0 );
    alpha_a = args.get< type::float_t >
                  ( "nonideal", "prof_alpha_a", 0 );
    out_kill = args.get< int >
                  ( "profile", "eta_out_kill", 0 );
    on      = ( eta_o > 0 ) || ( eta_a > 0 );
    return;
};

__host__ void chem_eta_t::read( const input & args )
{
    throw std::runtime_error
        ( "nonideal::chem_eta_t is not implemented"
          " yet;  use const_eta_t or hall_len_eta_t." );
    return;
};

////////////////////////////////////////////////////////////
//  Default - mode explicit instantiations.
template struct ni_t< >;
template struct ni_eos_t< >;

};                // namespace mhd_ee::ni
