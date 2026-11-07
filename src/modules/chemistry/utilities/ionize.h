#pragma once

namespace chemistry::ionize
{
////////////////////////////////////////////////////////////
// Ionization energy

struct photoionize_t
{
    ////////// Data //////////
    int               n_e;  // Electron        number
    int                 z;  // Nucleus charge  number
    int                 n;  // Main    quantum number
    int                 l;  // Angular quantum number
    type::float_t       P;
    type::float_t     y_a;
    type::float_t     y_w;
    type::float_t     e_0;
    type::float_t    e_th;
    type::float_t sigma_0;

    ////////// Functions //////////
    type::float_t sigma( const type::float_t & hv,
                         const type::float_t ion_state = 0 )
    {
        if( ion_state > z )
            throw std::runtime_error( "ionize_state > z" );

        type::float_t sigma( 0 );
        const int n_e = z - ion_state;
        for( const auto & e : elements_dat[ z - 1 ].ionize )
        {
            if( this->n_e != n_e || hv < this->e_th )
                continue;

            const auto y = ( hv / this->e_0 );
            const auto Q = 5.5 + this->l - 0.5 * this->P;
            const auto fy
                =  ( pow( y - 1, 2 ) + pow( this->y_w, 2 ) )
                * pow( y, -Q ) * pow
                ( 1. + sqrt( y / this->y_a ), -this->P );
            sigma += fy * this->sigma_0;
        }
        return sigma * 1e-18f
    };
};
};                          // namespace chemistry::ionize
