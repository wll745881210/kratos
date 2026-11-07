#pragma once

namespace multigrid::operation
{
////////////////////////////////////////////////////////////
// Initializaiton/finalization

template< class derived_T > __host__ void operation_t
< derived_T >::finalize( mesh::mod_base_t & mod )
{
    mod.p_dev       ->free_device(  defect_d );
    mod.p_dev       ->free_host  (  defect_h );
    mod.p_dev       ->free_device( src_ave_d );
    return mod.p_dev->free_host  ( src_ave_h );
}

template< class derived_T > __host__ void operation_t
< derived_T >::setup_mem( mesh::mod_base_t & mod )
{
    this->flag              = 0;
    auto & dev( * mod . p_dev );
    using  f_t  = type::float_t;
    const auto  n = 2 * n_fld + 1;    
    src_ave_h   = dev.template malloc_host  < f_t > ( n );
    src_ave_c   = dev.template malloc_const < f_t > ( n );
    src_ave_d   = dev.template malloc_device< f_t > ( n );
    dev.f_mset  ( src_ave_d, 0,  n * sizeof ( f_t ) , 0 );
    memset      ( src_ave_h, 0,  n * sizeof ( f_t )     );
    dev.f_cc ( src_ave_c, src_ave_h, sizeof ( f_t ) * n );
    const  auto n_def( 2 * n_fld ); // # of defect stats.
    defect_h = dev.template malloc_host  < f_t >( n_def );
    defect_d = dev.template malloc_device< f_t >( n_def );
    field_max_d = defect_d + n_fld;
    field_max_h = defect_h + n_fld;
    dev.f_mset  ( defect_d, 0, n_def * sizeof( f_t ), 0 );
    std::memset ( defect_h, 0, n_def * sizeof( f_t )    );
    return;
}

template< class derived_T > __host__ void operation_t
< derived_T >::stat_src_ave( mesh::mod_base_t & mod )
{
    auto & dev( * mod.p_dev );
    for( auto & d : mod )
        dev.sync_stream( d.stream );
    const auto & strm( mod.begin(  )->second->stream );
    const auto n_tot = 2 * n_fld + 1;
    const auto s_tot = type::size ( src_ave_d ) * n_tot;
    dev.cp_a  ( src_ave_h, src_ave_d, n_tot, strm );
    dev.f_mset( src_ave_d,         0, s_tot, strm );
    dev.sync_stream( strm );
    mod.p_mesh->p_com->reduce_all
        ( src_ave_h, comm::operation_t::sum, n_tot );
    for( int n = 0; n < n_tot; ++ n ) 
        src_ave_h[ n ] /= src_ave_h [ n_tot - 1 ];
    dev.f_cc( src_ave_c,  src_ave_h , s_tot );
    return dev.sync_stream( strm );
}

template< class derived_T > __host__
type::float_t operation_t< derived_T >::stat_defect
( mesh::mod_base_t & mod )
{
    auto & dev( * mod.p_dev );
    for( auto & d : mod )
        dev.sync_stream( d.stream );
    const auto & strm( mod.begin(  )->second->stream );    
    dev.cp_a( defect_h, defect_d,  2 * n_fld,   strm );
    dev.f_mset( defect_d, 0, type::size( defect_d )
                * n_fld * 2, strm );
    dev.sync_stream( strm );
    for( int n = 1; n < n_fld; ++ n )
    {
        defect_h   [ 0 ] = std::min( defect_h   [ 0 ] ,
                                     defect_h   [ n ] );
        field_max_h[ 0 ] = std::min( field_max_h[ 0 ] ,
                                     field_max_h[ n ] );
    }   // On device side, both recorded by negative
    field_max_h[ 0 ] = fabs( field_max_h[ 0 ] ) + 1e-32;
    auto defect = fabs( defect_h[ 0 ] / field_max_h[ 0 ] );
    // Defect for each block first makes more sense!
    mod.p_mesh->p_com->reduce_all
         ( & defect, comm::operation_t::max );
    return defect;
}

////////////////////////////////////////////////////////////
// Resource

template < class derived_T >
template < class     bdt_T > __host__ __device__
int operation_t< derived_T >::n_sh_smooth
( const bdt_T & bdt ) const
{
// #warning "WHY!!!!"
    return n_fld * ( 2 * bdt.geo.n_dim + 3 );
}

template < class derived_T >
template < class     bdt_T > __host__ __device__
int operation_t< derived_T >::n_sh_refine
( const bdt_T & bdt ) const
{
    return n_fld * 4;
}

template < class derived_T >
template < class     bdt_T > __host__ __device__
int operation_t< derived_T >::n_sh_coarse
( const bdt_T & bdt ) const
{
    return n_sh_smooth( bdt ) + n_fld * 2;
}

template< class derived_T >
template< class     bdt_T > __host__
auto    operation_t< derived_T >::rsc_refine
( const bdt_T & b_c, int  n_th ) const
{
    using  f_t = typename decltype( b_c.mgd )::f_t;
    const  int n_cell = b_c.geo.n_ceff.prod(  );
    n_th = std::min   ( n_cell,  n_th  ) ;
    const dim3 n_bl ( ( n_cell + n_th  - 1 )    / n_th );
    int s_sh = n_th * sizeof( f_t ) * n_sh_refine( b_c );
    return std::make_tuple  ( n_bl, dim3( n_th ), s_sh );
}

template< class derived_T >
template< class     bdt_T > __host__
auto    operation_t< derived_T >::rsc_coarse
( const bdt_T & b_c, int  n_th ) const
{
    using  f_t = typename decltype( b_c.mgd )::f_t;
    const  int n_cell = b_c.geo.n_ceff.prod(  );
    n_th = std::min   ( n_cell,  n_th );
    const dim3 n_bl ( ( n_cell + n_th  - 1 )    / n_th );
    int s_sh = n_th * sizeof( f_t ) * n_sh_coarse( b_c );
    return std::make_tuple  ( n_bl, dim3( n_th ), s_sh );
}

template< class derived_T >
template< class     bdt_T > __host__
auto    operation_t< derived_T >::rsc_smooth
( const bdt_T & bdt, int  n_th ) const
{
    using  f_t = typename decltype( bdt.mgd )::f_t;
    const  int n_cell = ( bdt.geo.n_ceff.prod(  ) + 1 ) / 2;
    n_th = std::min   ( n_cell,  n_th );
    const dim3 n_bl ( ( n_cell + n_th  - 1 )    / n_th );
    int s_sh = n_th * ( sizeof( f_t ) * n_sh_smooth( bdt )
                      + sizeof( int ) ) ;
    return std::make_tuple  ( n_bl, dim3( n_th ), s_sh );
}

};                         // namespace multigrid::operation
