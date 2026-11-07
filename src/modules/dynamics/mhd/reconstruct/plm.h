#pragma once

#include "../../hydro/reconstruct/plm.h"

namespace dynamics::mhd::reconstruct
{
////////////////////////////////////////////////////////////
// PLM reconstruction for MHD

template< class derived_T = crtp::dummy_t >
struct plm_t : public crtp::helper_t
< plm_t, derived_T, hydro::reconstruct::plm_t >
{
    ////////// Types //////////
    __crtp_def_self__( plm_t, derived_T );

    ////////// Device-side interface //////////
    template< class dat_T, class bdt_T > __device__
    void operator(  ) ( dat_T & d, const bdt_T & b_d ) const
    {
        __syncthreads(  );
        const int n_int = d.q.n_int;
        const int nt    = d.n_feff * n_int;
        for(  int j = threadIdx.x; j < nt; j += blockDim.x )
        {
            int i, n;
            utils::divmod( i, n, j, n_int );
            auto  *   ql = d.ql.at    ( i );
            auto  *   qr = d.qr.at    ( i );
            if( n ==  idx::bcc   + d.axis )
            {
                ql[ n ] = d.q( i, n );
                qr[ n ] = ql [    n ];
                continue;
            }
            this->q_ll = d.q( i - 2, n );
            this->q_l  = d.q( i - 1, n );
            this->q_r  = d.q( i,     n );
            this->q_rr = d.q( i + 1, n );
            if( d.is_uniform )
                get_self(  ).uniform( d, i, n, b_d );
            else
            {
                get_self(  ).load_geo
                           ( d.xf.at( i ), d.xv.at( i ) );
                get_self(  ).general( d, i, n, b_d );
            }
        }
        return;
    };
};

};               // namespace dynamics::mhd::reconstruct
