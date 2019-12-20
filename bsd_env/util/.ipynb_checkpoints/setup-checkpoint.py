import sys
import math
import numpy
import freezeable
Freezeable = freezeable.Freezeable

class EmptyOptionContainer( Freezeable ):
    def __init__( self ):
        pass
   
# global control panel
class Setup( Freezeable ):
    
    def __init__( self, filename=None ):
        
        self.constants = EmptyOptionContainer()
        self.constants.RAD2DEG = 180.0/math.pi
        self.constants.DEG2RAD = math.pi/180.0
        self.constants.freeze()
        
        self.world = EmptyOptionContainer()
        self.world.type        = 'box'
        self.world.dim         = numpy.array([300,200,100])
        self.world.box_dim	   = numpy.array([ 8, 8, 80])
        self.world.boxmix      = False # True denotes to apply different textures to the world's obstacles
        self.world.wallmix     = True
        self.world.wall_offset = 0.
        self.world.touch_offset = 0.
        self.world.cam_height  = 30 ### def. used to be 4, ePuck sim trials use 6 (aligned to photo view)
        self.world.obstacles   = []
        self.world.goals = []
        self.world.limits      = numpy.array([numpy.NAN,numpy.NAN,numpy.NAN,numpy.NAN])
        self.world.color_background     = numpy.array([1.0,1.0,1.0])
        self.world.color_rat_marker     = numpy.array([1.0,1.0,1.0])
        self.world.color_rat_path       = numpy.array([0.5,0.5,0.5])
        self.world.color_sketch_default = numpy.array([0.0,0.0,0.8])
        self.world.touchBoard = []
        self.world.stepBoard = []
        self.world.freeze()
        
        self.opengl = EmptyOptionContainer()
        self.opengl.clip_near = 2.0
        self.opengl.clip_far  = 512.0
        self.opengl.freeze()
        
        self.rat = EmptyOptionContainer()
        self.rat.color	   = 'RGB'
        self.rat.fov       = numpy.array([128.0,128.0])
        self.rat.arc       = 320.0
        self.rat.path_dev  = 0.125
        self.rat.path_mom  = 0.5
        self.rat.bias	   = numpy.array([0.0,0.0])
        self.rat.bias_s	   = 0.0
        self.rat.speed     = 1.0
        self.rat.path      = None
        self.rat.path_loop = False
        self.rat.freeze()
        
        self.freeze()
        
        # optional: overwrite default values
        if filename != None:
            self.fromFile( filename )
        
    def toString(self):
        # world parameters
        s  = str('world.type').ljust(20)         + str(self.world.type)                    			+'\n'
        s += str('world.dim').ljust(20)          + str(self.world.dim)         +'\n'
        s += str('world.boxmix').ljust(20)       + str(self.world.boxmix)                  +'\n'
        s += str('world.wallmix').ljust(20)      + str(self.world.wallmix)                 +'\n'
        s += str('world.wall_offset').ljust(20)  + str(self.world.wall_offset)             +'\n'
        s += str('world.touch_offset').ljust(20)  + str(self.world.touch_offset)             +'\n'
        s += str('world.cam_height').ljust(20)   + str(self.world.cam_height)              +'\n'
        s += str('world.obstacles').ljust(20)    + str(self.world.obstacles)   			   +'\n'
        s += str('world.goals').ljust(20)    + str(self.world.goals)   			   +'\n'
        # rat paramerters
        s += str('rat.fov').ljust(20)      + str(self.rat.fov) +'\n'
        s += str('rat.speed').ljust(20)    + str(self.rat.speed)            +'\n'# string
        return s
    
    def toFile( self, filename ):
        f = open(filename, 'w')
        f.write( self.toString() )
        f.close()
        
    def fromFile( self, filename ):
        # file
        f = open( filename, 'r' )
        # world type and dim
        self.world.type = f.readline().strip('world.type').strip()
        s = f.readline().strip('world.dim').split()
        if self.world.type == 'file': self.world.dim = s[0]
        if self.world.type == 'box' or self.world.type == 'circle': self.world.dim = numpy.array( [ float(s[0]), float(s[1]), float(s[2]) ] )
        if self.world.type == 'star': self.world.dim = numpy.array( [ float(s[0]), float(s[1]), float(s[2]), float(s[3]) ] )
        if self.world.type == 'T': self.world.dim = numpy.array( [ float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]) ] )
        # miscellaneous
        s = f.readline().strip('world.box_dim').split()
        self.world.box_dim = numpy.array( [ float(s[0]), float(s[1]), float(s[2]) ] )
        self.world.boxmix = True if f.readline().strip('world.boxmix').strip() == 'True' else False
        self.world.wallmix = True if f.readline().strip('world.wallmix').strip() == 'True' else False
        self.world.wall_offset = float(f.readline().strip('world.wall_offset'))
        self.world.cam_height = float(f.readline().strip('world.cam_height'))
        # obstacles
        s = f.readline().strip('world.obstacles').split()
        if self.world.type != 'file':
            s = [ float(i.strip('[,]\n')) for i in s if i.strip('[,]\n')!='' ]  # clear & convert
            self.world.obstacles = [ s[i:i+4] for i in range(0,len(s),4) ]      # group in fours
        # world limits
        s = f.readline().strip('world.limits').split()
        self.world.limits = numpy.array( [ float(s[0]), float(s[1]), float(s[2]), float(s[3]) ] )
        # rat parameters
        f.readline()
        self.rat.color = f.readline().strip('rat.color').strip()
        s = f.readline().strip('rat.fov').split()
        self.rat.fov = numpy.array([ float(s[0]), float(s[1]) ])
        self.rat.arc = float(f.readline().strip('rat.arc'))
        self.rat.path_dev = float(f.readline().strip('rat.path_dev'))
        self.rat.path_mom = float(f.readline().strip('rat.path_mom'))
        s = f.readline().strip('rat.bias').split()
        self.rat.bias = numpy.array([ float(s[0]), float(s[1]) ])
        self.rat.bias_s = float(f.readline().strip('rat.bias_s'))
        self.rat.speed = float(f.readline().strip('rat.speed'))
        path_file = f.readline().strip('rat.path').split()
        self.rat.path = None if path_file == [] else path_file[0]
        # file
        f.close()
