class Freezeable( object ):

    def freeze( self ):
        self._frozen = None

    def __setattr__( self, name, value ):
        if hasattr( self, '_frozen' )and not hasattr( self, name ):
            raise AttributeError( "Error! No adding additional attribute '%s' to class '%s'!" % (name,self.__class__.__name__) )
        object.__setattr__( self, name, value )
