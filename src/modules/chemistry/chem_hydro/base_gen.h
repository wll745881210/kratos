#pragma once

#include "../../../device/device.h"
#include "../types.h"

namespace chemistry::hydro
{
////////////////////////////////////////////////////////////
// General base for chemically enabled hydro modules

struct base_gen_t
{
    ////////// Data //////////
    int   n_sps;
    int   n_hva;

    ////////// Function //////////
    __host__ virtual int  size_sh(  ) const { return 0; };
    __host__ virtual void set_chem
    ( const input & args, const sp_vec_t & species,
      device::base_t &  f_dev )
    {
        this->n_sps = species.size(  );
    };
};
};                         // namespace chemistry::hydro
