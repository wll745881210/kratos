#pragma once

#include "../../../../src/modules/chemistry/chemistry.h"

namespace extension::chemistry
{
////////////////////////////////////////////////////////////
// General reactions

struct reac_t : ::chemistry::reaction::std_t< reac_t >
{
    ////////// Type //////////
    using super_t = ::chemistry::reaction::std_t< reac_t >;
    using     f_t = super_t::f_t;
    enum   kind_t { standard, interp, troe };

    ////////// Data //////////
    kind_t kind;
    int  n_intp;                // Interpolation parameters

    //////////Functions //////////
    __host__ reac_t(  ) : kind( standard ) {  };

    __host__ virtual bool parse
    ( const std::string & l , const input & rarg,
      const ::chemistry::base_t & chem );

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t rate_coef( dat_T & d,
                   const bdt_T & bdt, const eos_T & eos,
                   const rls_T & rls ) const
    {
        if( kind != troe )
            return super_t::rate_coef( d, bdt, eos, rls );

        const f_t A0   = this->pars[0];
        const f_t b0   = this->pars[1];
        const f_t C0   = this->pars[2];
        const f_t Ainf = this->pars[3];
        const f_t binf = this->pars[4];
        const f_t Cinf = this->pars[5];
        const f_t  mu  = this->pars[10];

        const f_t n_tot
            = d.rho_cgs / ( mu * phys::cgs::mp );
        const f_t k0
            = A0 * pow_( d.T / 300, b0 )
            * exp_( -C0 / d.T );

        if( Ainf <= type::float_min * 100 )
            return k0 * n_tot;

        const f_t kinf
            = Ainf * pow_( d.T / 300, binf )
            * exp_( -Cinf / d.T );

        if( __abs1( k0 )   < type::float_min ||
            __abs1( kinf ) < type::float_min )
            return k0 * n_tot;

        const f_t Pr = k0 * n_tot / kinf;

        if( Pr <= 0 )
            return 0;

        f_t F = 1.0f;
        const f_t alpha = this->pars[6];
        const f_t    T1 = this->pars[7];
        const f_t    T2 = this->pars[8];
        const f_t    T3 = this->pars[9];

        const bool has_broad =
            __abs1( alpha )          > type::float_min ||
            __abs1( 1.0f / T1 ) > type::float_min ||
            __abs1( 1.0f / T2 ) > type::float_min ||
            __abs1( 1.0f / T3 ) > type::float_min;

        if( has_broad )
        {
            const f_t F_cent
                = exp_( -T2 / d.T )
                + alpha * exp_( -d.T / T1 )
                + ( 1.0f - alpha )
                * exp_( -d.T / T3 );
            if( F_cent <= 0 )
                return 0;

            const f_t lg_Fc
                = logf( F_cent ) / logf( 10.0f );
            const f_t lg_Pr
                = logf( Pr     ) / logf( 10.0f );
            const f_t C = -0.4f  - 0.67f * lg_Fc;
            const f_t N =  0.75f - 1.27f * lg_Fc;
            const f_t D =  0.14f;

            const f_t lpC  = lg_Pr + C;
            const f_t den  = N - D * lpC;
            const f_t den2 = __abs1( den )
                           > type::float_min
                           ? den * den
                           : type::float_min;

            const f_t lg_F
                = lg_Fc
                / ( 1.0f + lpC * lpC / den2 );
            F = exp_( lg_F * logf( 10.0f ) );
        }

        return kinf * ( Pr / ( 1.0f + Pr ) ) * F;
    }

    template < class dat_T, class bdt_T,
               class eos_T, class rls_T > __device__
    f_t dlnk_dlnT( dat_T & d,
                   const bdt_T & bdt, const eos_T & eos,
                   const rls_T & rls ) const
    {
        if( kind != troe )
            return super_t::dlnk_dlnT( d, bdt, eos, rls );

        const f_t b0    = this->pars[1];
        const f_t C0    = this->pars[2];
        const f_t Ainf  = this->pars[3];
        const f_t binf  = this->pars[4];
        const f_t Cinf  = this->pars[5];
        const f_t  mu   = this->pars[10];

        const f_t dlk0 = b0 + C0 / d.T;
        const f_t k0n
            = this->pars[0]
            * pow_( d.T / 300, b0 )
            * exp_( -C0 / d.T )
            * d.rho_cgs / ( mu * phys::cgs::mp );

        if( Ainf <= type::float_min * 100 )
            return dlk0;

        const f_t dlkinf = binf + Cinf / d.T;
        const f_t kinf
            = Ainf * pow_( d.T / 300, binf )
            * exp_( -Cinf / d.T );
        const f_t sum = k0n + kinf;

        if( __abs1( sum ) < type::float_min )
            return dlkinf;

        const f_t dlk
            = dlk0 * ( kinf / sum )
            + dlkinf * ( k0n / sum );
        return __min1( dlk, 30.0f );
    }
};
};                     // namespace extension::chemistry
