import os
import re
import gc

from numpy     import frombuffer, zeros, array, meshgrid,\
                      sqrt, log2, sum, minimum, maximum, \
                      copy, log10, arange, cbrt, unique,\
                      concatenate, log, pi, cos, sin, ones,\
                      exp
try:
    from numpy import trapezoid as trapz
except:
    from numpy import trapz     
#


from binary_io  import binary_io
from hydro_data import hydro_data

kb = 1.38065e-16;   # CGS Boltzmann constant
mp = 1.67262e-24;   # CGS proton mass;
eV = 1.60218e-12;   # CGS eV

############################################################
# Read cosmological simulation data

class cosmic_data( hydro_data ):
    def read_geometry( self ):
        self.x_lim = [ self.args( 'mesh', 'x_min' ) ,
                       self.args( 'mesh', 'x_max' ) ];
        self.x_lim = array( self.x_lim ).T;
        return;
    #
    def read_block_data( self, blk ):
        dat = self.data[ blk ];
        if  not 'n_cell' in dat:
            self.read_block_geometry( blk );
        #
        gam1 = self.args( 'dynamics', 'gamma' ) - 1;        
        bd   = self.get_field( blk, 'hydro_cons' );
        l0   = self.args( 'unit', 'length' );
        vel0 = l0 / self.args( 'unit', 'time' );
        rho0 = self.args( 'unit', 'density' ) ;    
        ene0 = rho0 * vel0**2        
        dat[ 'rho' ] = bd[ 0 ];
        dat[ 'ene' ] = bd[ 1 ];
        dat[ 'mom' ] = bd[ 2 : 5 ];
        dat[ 'ent' ] = bd[ 5 ];

        # buf = frombuffer( bd[ 6 ].data, dtype = '<f4' ).\
        #       reshape( ( * dat[ 'rho' ].shape, -1 ) );
        # dat[ 'neu' ] = buf[ :, :, :, 0 ];
        # dat[ 'met' ] = buf[ :, :, :, 1 ] / dat[ 'rho' ];
        # dat[ 'x_n' ] = maximum( dat[ 'neu' ] /
        #                         dat[ 'rho' ], 0 );

        isize = bd[ 6 ].itemsize // 2;
        buf  = frombuffer( bd[ 6 ].data,
                           dtype = '<u%d' % isize ).\
               reshape( ( * dat[ 'rho' ].shape, -1 ) );
        bfac = -64. / 2**32 if isize == 4 else -32. / 2**16;
        dat[ 'x_n' ] = exp( buf[ :, :, :, 0 ] * bfac );
        dat[ 'met' ] = exp( buf[ :, :, :, 1 ] * bfac );

        # dat[ 'x_n' ] = bd[ 6 ] / dat[ 'rho' ];
        # Fake data
        # dat[ 'met' ] = 0 * dat[ 'rho' ];
        # dat[ 'neu' ] = dat[ 'x_n' ] * dat[ 'rho' ];
        
        dat[ 'x_e' ] = maximum( 1   - dat[ 'x_n' ], 0 );
        dat[ 'pre' ] = bd[ 5 ] * dat[ 'rho' ]**gam1;

        a = self.globals[ 'a_cosmic' ];
        dat[ 'T' ] = dat[ 'pre' ] / dat[ 'rho' ] \
            * ene0 / rho0 \
            / a**2 / kb * mp / ( 1 + dat[ 'x_e' ] );
        dat[ 'vel' ] = dat[ 'mom' ] / dat[ 'rho' ];

        try:
            dat[ 'prim' ] = self.get_field\
               ( blk, 'prim' )#, load_raw = True );
        except:
            pass;
        #
        try:
            dat[ 'src' ] \
                = self.get_field( blk, 'mg_src' )[ 0 ]\
                       + self.globals[ 'mg_ave' ][ 0 ];
        except:
            try:
                dat[ 'src' ] \
                    = self.get_field( blk, 'prim', )[ 0 ]\
                    + self.globals[ 'mg_ave' ][ 0 ];
            except:
                pass;
            #
        #
        dat[ 'fld' ] = self.get_field( blk, 'mg' );
        return;
    #

    def read( self ):
        self.cache_used            =  False;
        self.globals[ 'a_cosmic' ] =    0.0;
        if 'a_cosmic' in self.bin_data.hmap:
            a = self.get_entry( 'a_cosmic', 'f' )[ 0 ];
            self.globals[ 'a_cosmic' ] = a;
            print( 'a_cosmic = ', a );
        if 'mg_ave' in self.bin_data.hmap:
            self.globals[ 'mg_ave' ] \
                = self.get_entry( 'mg_ave', 'f' );
        else:
            self.globals[ 'mg_ave' ] = [ 0, 0 ];
        #
        super(  ).read(  );
    #

    def get_float( self, key, flag = 'f' ):
        res = self.bin_data. as_array( key, flag );
        if flag == 'f':
            return res;
        s    = 2**( 8 * self.bin_data.hmap[ key ][ 1 ] );
        return res / ( s / ( 2 if flag == 'i' else 1 ) );
    #

    def load_particles( self, load_vel = False,
                        var_mpar = False ):
        self.bin_data.open_basic(  );
        self.map_x0 = zeros( ( len( self.data ), 3 ) );
        self.map_dx = zeros( ( len( self.data ), 3 ) );
        for b, db in self.data.items(  ):
            xf0 = [  ];
            dx0 = [  ];
            for xf in db[ 'x_f' ]:
                xf0.append( xf[ 0 ] );
                dx0.append( xf[ 1 ] - xf[ 0 ] );
            #
            ib = int( b.split( '_' )[ -1 ] )
            self.map_x0[ ib ] = array( xf0 )
            self.map_dx[ ib ] = array( dx0 );
        #
        x_arr = [  ];
        v_arr = [  ];
        m_arr = [  ];
        for k in self.bin_data.hmap:
            if  not ( 'par_rank_' in k and '_idx' in k ):
                continue;
            k0 = k.replace( 'idx', '' );
            idx = self.bin_data.as_array\
                ( k0 + 'idx',  'u' ).reshape( -1, 3 );
            dx  = self.get_float\
                ( k0 + 'd_x',  'u' ).reshape( -1, 3 );
            ibg = self.bin_data.as_array( k0 + 'iblg', 'u' )
            
            x_arr.append( self.map_x0[ ibg ] + ( idx + dx ) 
                        * self.map_dx[ ibg ] ) ;
            if var_mpar:
                m_arr.append\
                     ( self.get_float( k0 + 'prop' ) );
            if load_vel:
                vmag  = self.get_float( k0 + 'vmag', 'f' );
                mu    = self.get_float( k0 +   'mu', 'i' );
                st    = sqrt( 1 - mu**2 );
                phi   = self.get_float\
                      ( k0 +  'phi', 'i' ) * pi;
                v_arr.append\
                    ( array( [ vmag * st * cos( phi ),
                               vmag * st * sin( phi ),
                               vmag * mu ] ).T );
            #
            # gc.collect (   );
        #
        self.data[ 'particle_x' ] = concatenate( x_arr );
        if var_mpar:
            self.data[ 'particle_m' ] = concatenate( m_arr )
        else:
            self.data[ 'particle_m' ] \
                = self.get_float( k0 + 'prop' )[ 0 ];
        if load_vel:
            self.data[ 'particle_v' ] = concatenate( v_arr )
        return;
    #    
#

############################################################
# Rebinned data

class rebin_data( cosmic_data ):
    def read_geometry( self ):
        self.x_lim = [ self.args( 'mesh', 'x_min' ) ,
                       self.args( 'mesh', 'x_max' ) ];
        self.x_lim = array( self.x_lim ).T;
        return;
    #    
    def read_block_geometry( self, blk ):
        super(  ).read_block_geometry( blk );        
        dat  = self.data[ blk ];
        n_rb = self.get_entry( 'n_rebin', 'i' ) [ 0 ];
        
        for a in range( 3 ):
            xf0 = dat[ 'x_f' ][ a ];
            xf  = xf0[ : -1 ].reshape( -1, n_rb ).T[ 0 ];
            xf  = concatenate( ( xf, [ xf0[ -1 ] ] ) );
            dat[ 'x_c' ][ a ] \
                = ( xf[ 1 : ] + xf[ : -1 ] ) / 2;
            dat[ 'x_f' ][ a ] = xf;
        #
        n_c  = dat[ 'n_ceff' ] // n_rb;
        n_gh = dat[ 'n_gh' ];
        dat[ 'n_ceff' ] =        n_c;
        dat[ 'n_face' ] = 1   +  n_c;
        dat[ 'n_cell' ] = n_c + n_gh;
        return;
    #
    def read_block_data( self, blk ):
        dat = self.data[ blk ];
        if  not 'n_cell' in dat:
            self.read_block_geometry( blk );
        #
        gam1 = self.args( 'dynamics', 'gamma' ) - 1;
        bd   = self.get_field( blk, 'rebin_cons' );

        l0   = self.args( 'unit', 'length' );
        vel0 = l0 / self.args( 'unit', 'time' );
        rho0 = self.args( 'unit', 'density' ) ;    
        ene0 = rho0 * vel0**2        
        a = self.globals[ 'a_cosmic' ];        

        dat[ 'rho' ] = bd[ 0 ];
        dat[ 'ene' ] = bd[ 1 ];
        dat[ 'mom' ] = bd[ 2 : 5 ];
        # dat[ 'x_n' ] = maximum( bd[ 6 ] / bd[ 0 ], 0 );
        # dat[ 'neu' ] = bd[ 6 ] ;

        isize = bd[ 6 ].itemsize // 2;
        buf  = frombuffer( bd[ 6 ].data,
                           dtype = '<u%d' % isize ).\
               reshape( ( * dat[ 'rho' ].shape, -1 ) );
        bfac = -64. / 2**32 if isize == 4 else -32. / 2**16;
        dat[ 'x_n' ] = exp( buf[ :, :, :, 0 ] * bfac );
        dat[ 'met' ] = exp( buf[ :, :, :, 1 ] * bfac );
        dat[ 'neu' ] = dat[ 'x_n' ] * dat[ 'rho' ];

        dat[ 'x_e' ] = maximum( 1 - dat [ 'x_n' ], 0 );
        dat[ 'pre' ] = bd[ 5 ] * dat[ 'rho' ]**gam1;
        
        a = self.globals[ 'a_cosmic' ];
        dat[ 'T' ] = dat[ 'pre' ] / dat[ 'rho' ] \
            * ene0 / rho0 \
            / a**2 / kb * mp / ( 1 + dat[ 'x_e' ] );

        dat[ 'fld' ] = self.get_field( blk, 'rebin_mg_fld' )
        dat[ 'src' ] = self.get_field\
				( blk, 'rebin_mg_src' )[ 0 ]\
                + self.globals[ 'mg_ave' ][ 0 ];
        return;
    #
#

############################################################
# Emissivity data

def proj_fields( dd, fields, axis = 2 ):
    db0   = dd.data[ 'block_0' ];
    dx0   = array  ( [ x[ -1 ] - x[ 0 ] for \
                       x in db0[ 'x_f' ] ] );
    lb    = array  ( db0[ 'rho' ].shape );
    ltot  = dd.args( "mesh", "n_cell_global" )[ : : -1 ];
    lblk  = dd.args( "mesh", "n_cell_block"  );
    if  lblk is not None:
        l = lb * ltot // lblk[ : : -1 ];
    else:
        l = lb;
    #
    ax    = [ ( axis + a ) % 3 for a in range( 2 ) ];
    l     = array( [ l[ a ] for a in ax ] )
    x_min = dd.x_lim.T[ 0 ];
    res   = {  };
    for f in fields:
        shape    = dd.data[ 'block_0' ][ f ].shape;
        res[ f ] = zeros( l if len( shape ) <= 3 
                          else ( shape[ 0 ], * l ) );
    #
    x_f   = [ [  ], [  ], [  ] ];
    def f_trap( dat, xc ):
        res = trapz( dat, xc, axis = ( 2 + axis ) % 3 );
        return res.T if axis == 2 else res;
    #
    for b, db in dd.data.items(  ):
        if 'particle' in b:
            continue;
        x_min_s = [ x[ 0 ] for x in db[ 'x_f' ] ];
        for a in range( 3 ):
            x_f[ a ].append( db[ 'x_f' ][ a ] );
        dl = [ int( ( x_min_s[ a ] - x_min[ a ] \
                      + dx0[ a ] * 0.1 ) / dx0[ a ] ) \
               for a in range( 3 ) ][ : : -1 ];
        il = [ lb[ a ] *   dl[ a ]       for a in ax ];
        ir = [ lb[ a ] * ( dl[ a ] + 1 ) for a in ax ];
        for f in fields:
            shape = db[ f ].shape;
            xc    = db[ 'x_c' ][ axis ];
            if len( shape ) > 3:
                res[ f ][ a, il[ 0 ] : ir[ 0 ] ,
                             il[ 1 ] : ir[ 1 ] ] \
                             +=  f_trap( db[ f ][ a ], xc );
            else:
                res[ f ][ il[ 0 ] : ir[ 0 ] ,
                          il[ 1 ] : ir[ 1 ] ]\
                          +=  f_trap( db[ f ], xc );
            #
        #
    #
    x_f = [ unique( concatenate( x_f [ a ] ). round
                  ( decimals = 5 ) ) for a in ax ];
    return res, x_f;
#

############################################################
# Emissivity data

def emis_lya( data, a, ret_ene = True ):
    Y        = 0.1;
    mu_per_H = ( ( 1 - Y ) * mp + Y * 4 * mp ) \
               / ( 1 - Y );
    rho_g  = data[ 'rho_cgs' ];
    nH     = rho_g / mu_per_H / a**3;
    xe     = data[ 'x_e' ];
    T      = maximum( data[ 'T' ], 3 );
    x      = kb * T / ( 13.6 * eV );
    a_all  = 2e-14 * ( log( 1 + 1 / x ) + 3.499 );
    a_all /= sqrt( x ) + 0.652 * x + 0.214 * x**1.5;
    a_1    =  4.145e-14 / ( sqrt( x ) + 0.36 *
                    x**( 7 / 6 ) + 0.1471 * x**1.5 );
    a_B   = a_all - a_1
    res   = a_B * ( nH * xe )**2 * a**3
    if  ret_ene:
        res *= 10.2 * eV * a;
    return res;
#

def f_emis_ha( data, a ):
    Y        = 0.1;
    mu_per_H = ( ( 1 - Y ) * mp + Y * 4 * mp ) \
               / ( 1 - Y );
    rho_g  = data[ 'rho_cgs' ];
    nH     = rho_g / mu_per_H;
    xe     = data[ 'x_e' ];
    T      = maximum( data[ 'T' ], 3 );
    T4    = T / 1e4;
    a_3   =  4.55e-14 * T4**( -0.729 - 0.059 * log( T4 ) );
    # Table 14.1, Draine (2011)
    return a_3 * ( nH * xe )**2 * dd.a**3    
#


# data   = pd.read_csv( 'metal.csv' )
# zs     = array( [ 3, 1, 0.5, 0  ] );
# ln_a   = log( 1 / ( 1 + zs ) );
# ln_Z   = log( column_stack\
#             ( ( data[ 'x' ], data[ 'z_3' ], data[ 'z_1' ] ,
#                 data[ 'z_0.5' ], data[ 'z_0' ] ) ) );
# ln_rho = ln_Z.T[ 0 ];
# f_lnZ = RegularGridInterpolator\
#     ( ( ln_rho, ln_a ), ln_Z[ :, 1 : ],
#       bounds_error = False, fill_value = None );

# def f_emis_o( data, a, f_emis, x_O0 = 6e-4 ):
#     Y        = 0.1;
#     mu_per_H = ( ( 1 - Y ) * mp + Y * 4 * mp ) \
#                / ( 1 - Y );
#     rho_g  = data[ 'rho_cgs' ];
#     nH     = rho_g / mu_per_H;
#     xe     = data[ 'x_e' ];
#     T      = maximum( data[ 'T' ], 3 );
#     T4    = T / 1e4;
#     a_3   =  4.55e-14 * T4**( -0.729 - 0.059 * log( T4 ) );
#     # Table 14.1, Draine (2011)
#     return a_3 * ( nH * xe )**2 * dd.a**3    
# #
