#pragma once

#include "../../../types.h"
#include "../types.h"
#include <string>
#include <map>

namespace chemistry
{
////////////////////////////////////////////////////////////
//
struct elem_t
{
    std :: string name;
    int              z;   // Nucleus charge number
    int              n;   // Main quantum number
    int              l;   // Angular quantum number
    type::float_t    m;   // Mass in AMU
    type::float_t  I_e;   // Ionization energy in eV
};

struct elem_dat_t
{
    ////////// Data //////////
    std::map< int,      elem_t > elem_idx;
    std::map< std::string, int > elem_map;
    
    ////////// Functions //////////
    elem_dat_t(  );
    
    const elem_t & elem( const         int &  z );
    const elem_t & elem( const std::string &  e );
    bool  contain      ( const std::string &  e );
    void  enroll_elem  ( const elem_t & e_equiv );
};                         // struct   elem_dat_t
};                         // namespace chemistry
