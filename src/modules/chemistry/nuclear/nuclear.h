#pragma once

#include "../chemistry.h"
#include "reaction.h"

namespace chemistry::nuclear
{
////////////////////////////////////////////////////////////
// Chemistry module

class base_t : public ::chemistry::base_t
{
protected:                      // Function
    virtual void parse_species( const input & args );
public:
    virtual void read         ( const input & args );
};
};                         // namespace chemistry
