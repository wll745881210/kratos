#include "corrector.h"
#include "../chemistry.h"

namespace chemistry::nuclear::corrector
{
////////////////////////////////////////////////////////////
// 
__host__ void  base_t::init
( const  input & args, ::chemistry::base_t & mod )
{
    std::vector< int > a_h;
    for( const auto  & sp : mod.species )
        a_h.push_back( sp.elem.at( 0  ) );
    n_sp = mod.species.size      (      );

    const int size_b = sizeof( int ) * a_h . size(  );
    a = ( int * ) mod.p_dev->f_malloc_const( size_b );
    mod.p_dev->f_cc ( a,  a_h .data(  ),     size_b );
    return;
}

};             // namespace chemistry::nuclear::corrector
