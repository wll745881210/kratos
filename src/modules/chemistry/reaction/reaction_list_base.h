#pragma once

#include "reaction.h"

namespace chemistry
{
class base_t;
};

namespace chemistry::reaction::list
{
////////////////////////////////////////////////////////////
// Lists for reactions

struct base_t
{
    ////////// Host-side Interfaces //////////
    __host__ virtual void init
    ( const  input & args, ::chemistry::base_t & mod ) =  0;
    __host__ virtual void init_extra
    ( const  input & args, ::chemistry::base_t & mod ) {  };
    __host__ virtual void free( device::base_t & f_d ) {  };
    __host__ virtual int  size_sh (  ) const { return  0; };
};
};                   // namespace chemistry::reaction::list
