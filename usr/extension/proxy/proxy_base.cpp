#include "proxy_base.h"

namespace extension
{
////////////////////////////////////////////////////////////
// Base of all proxies

std::weak_ptr< mod_base_t > prx_base_t::p_mgd;
std::weak_ptr< mod_base_t > prx_base_t::p_hyd;
std::weak_ptr< mod_base_t > prx_base_t::p_par;

};                              // namespace extension
