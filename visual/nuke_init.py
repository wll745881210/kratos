from binary_io  import binary_io
from numpy      import array, byte, meshgrid

############################################################
# 

class gen_profiles_bin( binary_io ):

    def enroll( self, func, coords, prefix,
                field_names = None, coord_type  = 'car' ):
        n_pts  = array( [ len( coords[ i ] ) for i in
                          range( len( coords ) ) ] );
        if  not isinstance( coord_type, byte ) :
            coord_type = bytes( coord_type,
                                encoding = 'ascii' )
        #
        self.cache( prefix + '_coord', coord_type );
        self.cache( prefix + '_n_pts',      n_pts );
        for a in range( len( n_pts ) ):
            self.cache( prefix + '_x_%d' % a, coords[ a ] );
        #
        if  field_names  is  None:
            field_names = [ 'data' ];
        if  not hasattr( func, '__len__' ):
            func = [ func ];
        if  len( func ) != len( field_names ):
            raise ValueError( "Inconsistent function "
                              "list length with names" );
        #
        coords_mg = meshgrid( * coords, indexing = 'ij' );
        for n, field in enumerate( field_names ):
            self.cache( prefix + '_' + field,
                        func[ n ]( * coords_mg ) );
        #
        return;
    #
#
    
