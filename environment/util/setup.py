import sys
import math
import numpy
from environment.util.freezeable import Freezeable

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
        self.world.color_rat_marker     = numpy.array([0.0,0.5,0.0])
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