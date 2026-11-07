#include "periodic.h"
#include <vector>
#include <map>

namespace chemistry
{
////////////////////////////////////////////////////////////
// Data, hard-coded. You want to challenge basic chemistry?

std::vector< elem_t > elem_orig =
{
    { "e",    0, 0, 0, 5.49e-4,  0.0000 },
    { "H",    1, 1, 0, 1.00794, 13.5984 },
    { "He",   2, 1, 0, 4.0026,  24.5874 },
    { "Li",   3, 2, 0, 6.941,    5.3917 },
    { "Be",   4, 2, 0, 9.01218,  9.3227 },
    { "B",    5, 2, 1, 10.811,   8.2980 },
    { "C",    6, 2, 1, 12.0107, 11.2603 },
    { "N",    7, 2, 1, 14.0067, 14.5341 },
    { "O",    8, 2, 1, 15.9994, 13.6181 },
    { "F",    9, 2, 1, 18.9994, 17.4228 },
    { "Ne",  10, 2, 1, 20.1797, 21.5645 },
    { "Na",  11, 3, 0, 22.9898,  5.1391 },
    { "Mg",  12, 3, 0, 24.305,   7.6462 },
    { "Al",  13, 3, 1, 26.9815,  5.9858 },
    { "Si",  14, 3, 1, 28.0855,  8.1517 },
    { "P",   15, 3, 1, 30.9738, 10.4867 },
    { "S",   16, 3, 1, 32.065,  10.3600 },
    { "Cl",  17, 3, 1, 35.453,  12.9676 },
    { "Ar",  18, 3, 1, 39.948,  15.7596 },
    { "K",   19, 4, 0, 39.0983,  4.3407 },
    { "Ca",  20, 4, 0, 40.078,   6.1132 },
    { "Sc",  21, 4, 2, 44.9559,  6.5615 },
    { "Ti",  22, 4, 2, 47.867,   6.8281 },
    { "V",   23, 4, 2, 50.9415,  6.7462 },
    { "Cr",  24, 4, 2, 51.9961,  6.7665 },
    { "Mn",  25, 4, 2, 54.938,   7.4340 },
    { "Fe",  26, 4, 2, 55.845,   7.9024 },
    { "Co",  27, 4, 2, 58.9332,  7.8810 },
    { "Ni",  28, 4, 2, 58.6934,  7.6398 },
    { "Cu",  29, 4, 2, 63.546,   7.7264 },
    { "Zn",  30, 4, 2, 65.38,    9.3492 },
    // Abstract "Metal"
    { "Z",   -1, 0, 0, 16.,      13.6   }    
};

////////////////////////////////////////////////////////////
//

elem_dat_t::elem_dat_t(  )
{
    for( const auto & e : elem_orig )
        enroll_elem( e );
    return;
}

const elem_t & elem_dat_t::elem( const int & z )
{
    return  elem_idx.at ( z ) ;
}

const elem_t & elem_dat_t::elem( const std::string & name )
{
    return  elem_idx.at( elem_map.at( name ) );
}

bool elem_dat_t::contain( const std::string & e )
{
    return elem_map.find( e ) != elem_map.end(  );
}

void elem_dat_t::enroll_elem( const elem_t & e_equiv )
{
    elem_idx[ e_equiv.   z ] = e_equiv;
    elem_map[ e_equiv.name ] = e_equiv.z;
    return;
}

};                         // namespace chemistry
