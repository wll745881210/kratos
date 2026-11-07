#pragma once

#include <map>
#include <vector>
#include <string>

namespace chemistry
{
////////////////////////////////////////////////////////////
// Chemicals

struct species_t
{
    int                    idx;
    int                 n_atom;  // Total # of atoms
    std ::string          name;
    type::float_t     mass_cgs;  // Mass in CGS
    std ::map< int, int > elem;  // Elemental comp
};

struct sp_vec_t : public std::vector< species_t >
{
    std::map< std::string, int >   indices;
    using     std::vector< species_t >::at;

    void setup_idx(  )
    {
        for( size_t i = 0; i < this->size(  ); ++ i )
        {
            auto & sp( at( i ) );
            if( indices.find( sp.name ) != indices.end(  ) )
                throw std::runtime_error
                    ( "Duplicated species: " + sp.name );
            sp.idx = ( indices [ sp.name ] = i );
        }
        return;
    };
    const int & at( const std::string & sp ) const
    {
        return indices.at( sp );
    };
    bool exist( const std::string & sp ) const
    {
        return ( indices.find( sp ) != indices.end(  ) );
    };
};
};                              // namespace chemistry
