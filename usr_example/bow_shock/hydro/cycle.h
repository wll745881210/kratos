#pragma once

namespace friction
{
////////////////////////////////////////////////////////////
//

struct cycle_t : public mesh::cycle::base_t
{
protected:                      // Types
    using float_t = type::float_t;
protected:                      // Data
    bool   enable_stat;
    float_t   t_gf_out;
    float_t  dt_gf_out;
    std::ofstream fout;
public:
    std::function< void( const    bool & ) > f_pre;
    std::function< void(                 ) > f_acc;
    std::function< void( std::ofstream & ) > f_out;
protected:
    virtual void init
    ( const input & args, mesh::mesh_t & mesh ) override
    {
        mesh::cycle::base_t::init( args, mesh );
        enable_stat = args.get< bool >
                ( "cycle",  "stat_grav",    false );
        t_gf_out    = args.get< float_t >
                ( "cycle",  "t_gf_out_next", 0.00 );
        dt_gf_out   = args.get< float_t >
                ( "cycle",  "dt_gf_out",     1e32 );
        return;
    };
    virtual bool step( mesh::mesh_t & mesh ) override
    {
        const bool stat_grav = enable_stat
            & ( ( i_cycle % n_display_freq == 0 )
                | update_tout( t_gf_out, dt_gf_out ) );
        f_pre( stat_grav );
        const auto res = mesh::cycle::base_t::step( mesh );
        f_acc(  );
        if( stat_grav && mesh.p_com->is_root(  ) )
        {
            fout << t << ' ';
            f_out( fout );
        }
        return res;
    };

    virtual void evolve( mesh::mesh_t & mesh ) override
    {
        t_gf_out = 0;
        auto hist_file_name = prefix  +  "_gf_hist.dat";
        auto fout_flag      = std::ios::out;
        if( mesh.is_restart (  ) )
            fout_flag      |= std::ios::app;
        fout.open( hist_file_name.c_str(  ), fout_flag );
        if( ! mesh.is_restart (  ) )
            fout << "# t acc_x acc_y acc_z\n";
        mesh::cycle::base_t::evolve( mesh );
        return fout.close(  );
    };
    virtual void save( binary_io::base_t & bio ) override
    {
        fout << std::flush;
        return mesh::cycle::base_t::save( bio );
    };
};

};                              // namespace friction

