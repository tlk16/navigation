# system
import os
import sys
import types
import string

# python image library
from PIL import Image as img

# math
import math
import numpy
# numpy.random.seed(3)
import random as rnd
# rnd.seed(3)
# OpenGL
from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *

# utilities / own
from environment.util.setup import *
from environment.util.opengl_text import *

#defines
def_MARKER_HEIGHT =   0.1  # default height of drawn debug markers
def_CUSTOM_HEIGHT =  12.0  # default height for walls in custom mazes ### adapted for epuck scenario atm

#===============================================================[ Wall Segment ]

class Wall( Freezeable ):
    
    def __init__( self, left_end, right_end, height, texture=None, offset=None ):
        self.vec_from = left_end
        self.vec_to   = right_end
        self.height   = height
        self.normal   = numpy.array( [(self.vec_to-self.vec_from)[1],-(self.vec_to-self.vec_from)[0]] )
        self.normal  /= numpy.sqrt( self.normal[0]**2 + self.normal[1]**2 )
        self.texture  = 0 if texture==None else texture
        self.offset   = 0.0 if offset==None else offset
        
    # check wether a given point lies in front of the wall
    def facingFront( self, pos ):
        if numpy.dot( pos-(self.vec_from+0.5*wall_vec), self.normal ) < 0.0: 
            return False
        else:
            return True
        
    # check wether a given point lies closer to the wall than the allowed offset
    def proximityAlert( self, pos ):
        wall_vec = self.vec_to - self.vec_from
        mu  = (pos[0]-self.vec_from[0])*(self.vec_to[0]-self.vec_from[0]) + (pos[1]-self.vec_from[1])*(self.vec_to[1]-self.vec_from[1])
        mu /= wall_vec[0]**2 + wall_vec[1]**2
        if mu < 0.0:
            return numpy.sqrt( (pos-self.vec_from)[0]**2 + (pos-self.vec_from)[1]**2 ) < self.offset
        elif mu > 1.0:
            return numpy.sqrt( (pos-self.vec_to)[0]**2 + (pos-self.vec_to)[1]**2 ) < self.offset
        else:
            proj = self.vec_from + mu*wall_vec
            dist = numpy.sqrt( (proj-pos)[0]**2 + (proj-pos)[1]**2 )
            return dist < self.offset
        
    # check wether the path between two positions crosses the wall segment
    def crossedBy( self, pos_old, pos_new ):
        # check 1: old position in front, new position behind wall segment
        wall_vec = self.vec_to - self.vec_from
        if not ( numpy.dot( pos_old-(self.vec_from+0.5*wall_vec), self.normal ) > 0.0 and numpy.dot( pos_new-(self.vec_from+0.5*wall_vec), self.normal ) < 0.0 ):
            return False
        # determine crossing point (moving parallel to wall is cought by check 1)
        xy = self.vec_to-self.vec_from
        uv = pos_new-pos_old
        l = ( pos_old[1] + (self.vec_from[0]*uv[1]-pos_old[0]*uv[1])/(uv[0]+1e-8) - self.vec_from[1] ) / ( (1.0-(xy[0]*uv[1])/(uv[0]*xy[1] + 1e-8)) * xy[1])
        m = ( self.vec_from[0] + l*xy[0] - pos_old[0] ) / uv[0]
        # check C: intersection lies between old & new position, and also within wall segment limits
        if l >= 0 and l <= 1.0 and m >= 0 and m <= 1.0:
            return True
        else:
            return False
        
#============================================================[ Texture Catalog ]

class Textures:
    
    def __init__( self ):
        # texture dictionary index
        self.index = {}
        # texture id's by category
        self.floor  = []
        self.wall   = []
        self.crate  = []
        self.skybox = None
        # find available textures
        tex_list = os.listdir( './textures' )
        tex_list.sort()
        for f in tex_list:
            s = f.split( '.' )[0]
            i = self.load( 'textures/'+f )
            # add to category
            if   'floor'  in s: self.floor.append( i )
            elif 'wall'   in s: self.wall.append ( i )
            elif 'crate'  in s: self.crate.append( i )
            elif 'skybox' in s: self.skybox = i
            else: continue
            # add to index
            self.index[s] = i
        # assign random colors to textures
        self.sketch_color = numpy.zeros([len(self.floor)+len(self.wall)+len(self.crate)+2, 3]) # +2 b/c skybox and tex 0
        for i in range(self.sketch_color.shape[0]):
            self.sketch_color[i] = numpy.array([min(numpy.random.random()+0.2,1.0),min(numpy.random.random()+0.2,1.0),min(numpy.random.random()+0.2,1.0)])
        self.sketch_color[0] = numpy.array([0.0,0.0,0.8]) # one default color
        
    def load( self, filename, ):
        mipmapping = False
        # open image
        src_img = img.open( filename )
        img_str = src_img.tobytes( 'raw', 'RGB', 0, -1 )
        # opengl image id
        img_id = glGenTextures(1)
        glBindTexture( GL_TEXTURE_2D, img_id )
        # texture parameters
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        if mipmapping:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST )
        else:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);	
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        # store texture image (use mipmapping to kill off render artifacts)
        glPixelStorei( GL_UNPACK_ALIGNMENT, 1 )
        if mipmapping:
            gluBuild2DMipmaps( GL_TEXTURE_2D, 3, src_img.size[0], src_img.size[1], GL_RGB, GL_UNSIGNED_BYTE, img_str )
        else:
            glTexImage2D( GL_TEXTURE_2D,    # target
            	          0,                # mipmap level
            	          3,                # color components (3: rgb, 4: rgba)
            	          src_img.size[0],  # texture width
            	          src_img.size[1],  # texture height
            	          0,                # border
            	          GL_RGB,           # format
        	              GL_UNSIGNED_BYTE, # data type
            	          img_str )         # texture data
        # return handle id
        return int(img_id)
    
#================================================================[ World Class ]
class World:
    
    def __init__( self, control ):
        
        # components
        self.__walls__        = []
        self.__ctrl__         = control
        self.__display_list__ = None
        self.__textures__     = Textures()
        
        self.__textures__.sketch_color[0] = self.__ctrl__.color_sketch_default
        
        # construct world
        self.__constructWorld__( control )
        
        
        # optional obstacles
        OBC = []
        if len(control.obstacles) != 0:
            for i in range(int(control.obstacles[0][0])):
                startX, startY, endX, endY = self.__addObstacleWalls__( control.obstacles[0][1:])
                OBC.append( [startX, startY, endX, endY] )

        # find world limits [x_min, y_min, x_max, y_max]
        control.limits = numpy.array([numpy.iinfo('i').max,numpy.iinfo('i').max,numpy.iinfo('i').min,numpy.iinfo('i').min])
        for w in self.__walls__:
            control.limits[0] = min( w.vec_to[0], w.vec_from[0], control.limits[0] )
            control.limits[1] = min( w.vec_to[1], w.vec_from[1], control.limits[1] )
            control.limits[2] = max( w.vec_to[0], w.vec_from[0], control.limits[2] )
            control.limits[3] = max( w.vec_to[1], w.vec_from[1], control.limits[3] )
            
        # construct display list
        self.__constructDisplayList__()
        
        # check goal
        if len(control.goals) != 0:
            for j, g in enumerate(control.goals):
                miss = self.__isGoal__( g, OBC )
                if not miss:
                    print(' Illegal Goal [%f, %f] !! ' %(g[0], g[1]))
                    sys.exit()
                    
        control.touchBoard = self.__isTouch__( OBC )
        control.stepBoard = self.__isValidStep__( OBC )
        
    
    def __isValidStep__(self, obstacles):
        length = int(self.__ctrl__.dim[0])
        width = int(self.__ctrl__.dim[1])
        offset = int(self.__ctrl__.wall_offset)
        
        world_null = numpy.zeros((length, width), dtype=int)
        world_null[:offset] = 1
        world_null[(length - offset):] = 1
        world_null[:,:offset] = 1
        world_null[:,(width - offset):] = 1
        
        
        if len(obstacles) != 0:
            for i in range(len(obstacles)):
                pos_ob = obstacles[i]
                lfrom_ob = int(pos_ob[0]-offset) if (pos_ob[0] - offset)>=0 else 0
                lto_ob = int(pos_ob[2]+offset) if (pos_ob[2] + offset)<length else int(length-1)
                wfrom_ob = int(pos_ob[1]-offset) if (pos_ob[1] - offset)>=0 else 0
                wto_ob = int(pos_ob[3]+offset) if (pos_ob[3] + offset)<width else int(width-1)
                
                world_null[lfrom_ob : lto_ob, wfrom_ob : wto_ob] = 1
       
        return world_null                
    
    def __isTouch__(self, obstacles):
        length = int(self.__ctrl__.dim[0])
        width = int(self.__ctrl__.dim[1])
        offset = int(self.__ctrl__.touch_offset)
        
        world_null = numpy.zeros((length, width), dtype=int)
        world_null[:offset] = 1
        world_null[(length - offset):] = 1
        world_null[:,:offset] = 1
        world_null[:,(width - offset):] = 1
        
        if len(obstacles) != 0:
            for i in range(len(obstacles)):
                pos_ob = obstacles[i]
                lfrom_ob = int(pos_ob[0]-offset) if (pos_ob[0] - offset)>=0 else 0
                lto_ob = int(pos_ob[2]+offset) if (pos_ob[2] + offset)<length else int(length-1)
                wfrom_ob = int(pos_ob[1]-offset) if (pos_ob[1] - offset)>=0 else 0
                wto_ob = int(pos_ob[3]+offset) if (pos_ob[3] + offset)<width else int(width-1)
                
                world_null[lfrom_ob : lto_ob, wfrom_ob : wto_ob] = 1
       
        return world_null
    
    def validTouch(self, pos):
        touchBoard = self.__ctrl__.touchBoard
        length = int(self.__ctrl__.dim[0])
        width = int(self.__ctrl__.dim[1])
        
        cond1 = numpy.sum(touchBoard[int(pos[0]-1) , int(pos[1]-1) : int(pos[1]+2)])==3
        cond2 = numpy.sum(touchBoard[int(pos[0]-1) : int(pos[0]+2) , int(pos[1]-1)])==3
        cond3 = numpy.sum(touchBoard[int(pos[0]+1) , int(pos[1]-1) : int(pos[1]+2)])==3
        cond4 = numpy.sum(touchBoard[int(pos[0]-1) : int(pos[0]+2) , int(pos[1]+1)])==3
        
        touch = 0
        if cond1 and cond2 and cond3 and cond4:
            offset = int(self.__ctrl__.touch_offset)
            c1 = pos[0] < offset
            c2 = pos[0] > (length - offset)
            c3 = pos[1] < offset
            c4 = pos[1] > (width - offset)
            if c1 and pos[1]<=(width-offset):
                touch = 1
            elif c3 and pos[0]>=offset:
                touch = 2
            elif c2 and pos[1]>=offset:
                touch = 3
            elif c4 and pos[0]<=(length-offset):
                touch = 4           
        elif cond4 and cond1:
            touch = 4  
        elif cond1:
            touch = 1 # west-touch
        elif cond2:
            touch = 2 # south-touch
        elif cond3:
            touch = 3 # east-touch
        elif cond4:
            touch = 4 # north-touch
        
        return touch
        
        
                    
    #-----------------------------------------------------------[ check is goal ]
    def __isGoal__(self, goal, obstacles):
        
        miss1 = False
        miss2 = True
        
        est1 = ( goal[0] >= self.__ctrl__.wall_offset ) and ( goal[0] < (self.__ctrl__.dim[0] - self.__ctrl__.wall_offset) )
        est2 = ( goal[1] >= self.__ctrl__.wall_offset ) and ( goal[1] < (self.__ctrl__.dim[1] - self.__ctrl__.wall_offset) )
        if est1 and est2:
            miss1 = True
            
        for i, o in enumerate(obstacles):
            est3 = ( goal[0] >= (o[0] - self.__ctrl__.wall_offset) ) and ( goal[0] <= (o[2] + self.__ctrl__.wall_offset) )
            est4 = ( goal[1] >= (o[1] - self.__ctrl__.wall_offset) ) and ( goal[1] <= (o[3] + self.__ctrl__.wall_offset) )
            if est3 and est4:
                miss2 = False
                break
            
        if miss1 and miss2:
            return True
        else:
            return False
        
    #-----------------------------------------------------------[ construction ]
    
    def __constructWorld__( self, control ):
        
        # build rectangular world ( dim := [ <world_length>, <world_width>, <world_height> ] )
        if control.type == 'box':
            self.__walls__.append( Wall( numpy.array([           0.0,           0.0]), numpy.array([           0.0,control.dim[1]]), control.dim[2], self.__textures__.wall[3 if control.wallmix else 0], control.wall_offset ) )
            self.__walls__.append( Wall( numpy.array([           0.0,control.dim[1]]), numpy.array([control.dim[0],control.dim[1]]), control.dim[2], self.__textures__.wall[0 if control.wallmix else 0], control.wall_offset ) )
            self.__walls__.append( Wall( numpy.array([control.dim[0],control.dim[1]]), numpy.array([control.dim[0],           0.0]), control.dim[2], self.__textures__.wall[1 if control.wallmix else 0], control.wall_offset ) )
            self.__walls__.append( Wall( numpy.array([control.dim[0],           0.0]), numpy.array([           0.0,           0.0]), control.dim[2], self.__textures__.wall[2 if control.wallmix else 0], control.wall_offset ) )
            
        # build star maze ( dim := [ <arms>, <arm_width>, <arm_length>, <arm_height> ] )
        elif control.type == 'star':
            # constants
            RAD2DEG = 180.0/math.pi
            DEG2RAD = math.pi/180.0
            # angles
            arm_dir  = 270.0
            dir_step = 360.0 / control.dim[0]
            inner_r  = control.dim[1] / (2*math.sin( (180.0/control.dim[0])*DEG2RAD ))
            # add arm walls
            for n in range(0,int(control.dim[0])):
                # arm vertices
                a = numpy.array([inner_r*math.cos((arm_dir-0.5*dir_step)*DEG2RAD), \
							  	 inner_r*math.sin((arm_dir-0.5*dir_step)*DEG2RAD)] )
                b = numpy.array([inner_r*math.cos((arm_dir-0.5*dir_step)*DEG2RAD) + math.cos(arm_dir*DEG2RAD)*control.dim[2], \
							  	 inner_r*math.sin((arm_dir-0.5*dir_step)*DEG2RAD) + math.sin(arm_dir*DEG2RAD)*control.dim[2]] )
                c = numpy.array([inner_r*math.cos((arm_dir+0.5*dir_step)*DEG2RAD), \
							  	 inner_r*math.sin((arm_dir+0.5*dir_step)*DEG2RAD)] )
                d = numpy.array([inner_r*math.cos((arm_dir+0.5*dir_step)*DEG2RAD) + math.cos(arm_dir*DEG2RAD)*control.dim[2], \
							  	 inner_r*math.sin((arm_dir+0.5*dir_step)*DEG2RAD) + math.sin(arm_dir*DEG2RAD)*control.dim[2]] )
                # arm walls (note: each complete arm gets a different texture in case of wallmix)
                self.__walls__.append( Wall( c, d, control.dim[3], self.__textures__.wall[0 if not control.wallmix else n%len(self.__textures__.wall)], control.wall_offset ) )
                self.__walls__.append( Wall( d, b, control.dim[3], self.__textures__.wall[0 if not control.wallmix else n%len(self.__textures__.wall)], control.wall_offset ) )
                self.__walls__.append( Wall( b, a, control.dim[3], self.__textures__.wall[0 if not control.wallmix else n%len(self.__textures__.wall)], control.wall_offset ) )
                # next
                arm_dir += dir_step
                if arm_dir >= 360: arm_dir -=360
                
        # build T maze ( dim := [ <vertical_length>, <vertical_width>, <horizontal_length>, <horizontal_width>, <wall_height> ] )
        elif control.type == 'T':
            # T coords
            a = numpy.array([-control.dim[1]/2.0, 0.0])
            b = numpy.array([-control.dim[1]/2.0, control.dim[0]])
            c = numpy.array([-control.dim[2]/2.0, control.dim[0]])
            d = numpy.array([-control.dim[2]/2.0, control.dim[0]+control.dim[3]])
            e = numpy.array([ control.dim[2]/2.0, control.dim[0]+control.dim[3]])
            f = numpy.array([ control.dim[2]/2.0, control.dim[0]])
            g = numpy.array([ control.dim[1]/2.0, control.dim[0]])
            h = numpy.array([ control.dim[1]/2.0, 0.0])
            # T walls
            self.__walls__.append( Wall( a, b, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 0], control.wall_offset ) )
            self.__walls__.append( Wall( b, c, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 1], control.wall_offset ) )
            self.__walls__.append( Wall( c, d, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 2], control.wall_offset ) )
            self.__walls__.append( Wall( d, e, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 3], control.wall_offset ) )
            self.__walls__.append( Wall( e, f, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 0], control.wall_offset ) )
            self.__walls__.append( Wall( f, g, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 1], control.wall_offset ) )
            self.__walls__.append( Wall( g, h, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 2], control.wall_offset ) )
            self.__walls__.append( Wall( h, a, control.dim[4], self.__textures__.wall[0 if not control.wallmix else 3], control.wall_offset ) )
            
        # build circle maze ( dim := [ <radius>, <segments>, <wall_height> ] )
        elif control.type == 'circle':
            # constants
            RAD2DEG = 180.0/math.pi
            DEG2RAD = math.pi/180.0
            # angles
            angle = 0.0
            step  = 360.0 / control.dim[1]
            # circle segments
            for w in range(0,int(control.dim[1])):
                a = numpy.array([control.dim[0]*math.cos(angle*DEG2RAD),control.dim[0]*math.sin(angle*DEG2RAD)])
                b = numpy.array([control.dim[0]*math.cos((angle+step)*DEG2RAD),control.dim[0]*math.sin((angle+step)*DEG2RAD)])
                self.__walls__.append( Wall( b, a, control.dim[2], self.__textures__.wall[0 if not control.wallmix else int( w/(control.dim[1]/len(self.__textures__.wall)) )], control.wall_offset ) )
                angle += step
                
        # build world from file ( dim := [ <file_name> ] )
        elif control.type == 'file':
            # open file
            print('Reading custom world from \'%s\'.' % ('./current_experiment/'+control.dim[0]))
            world_file = open( './current_experiment/' + control.dim )
            # read lines
            for l in world_file:
                if l == '\n': break
                par = l.split()
                if par[0] == 'floor': self.__textures__.floor[0] = self.__textures__.index[par[1]]
                else:
                    self.__walls__.append( Wall( numpy.array([float(par[0]),float(par[1])]), numpy.array([float(par[2]),float(par[3])]), def_CUSTOM_HEIGHT,  self.__textures__.index[par[4]], self.__ctrl__.wall_offset ) ) #self.__textures__.wall[int(par[4])], self.__ctrl__.wall_offset ) )
                    
    def __addObstacleWalls__( self, obstacle):
        choice = False
        while not choice:
            startX = float(rnd.randint(self.__ctrl__.wall_offset + 1, self.__ctrl__.dim[0] - self.__ctrl__.wall_offset - 1))
            startY = float(rnd.randint(self.__ctrl__.wall_offset + 1, self.__ctrl__.dim[1] - self.__ctrl__.wall_offset - 1))
            if len(obstacle) == 2: # Box Obstacles
                endX = startX + obstacle[0]
                endY = startY + obstacle[1]
            elif len(obstacle) == 1: # Bar Obstacles
                endX = startX + rnd.randint(self.__ctrl__.wall_offset + 10, self.__ctrl__.dim[0] - self.__ctrl__.wall_offset - 10)
                endY = startY + obstacle[0]
            choice = (endX < (self.__ctrl__.dim[0] - self.__ctrl__.wall_offset)) and (endY < (self.__ctrl__.dim[1] - self.__ctrl__.wall_offset))
        
        #print startX,endX, startY,endY
        self.__walls__.append( Wall( numpy.array([startX,endY]), numpy.array([startX,startY]), self.__ctrl__.box_dim[2], self.__textures__.crate[0], self.__ctrl__.wall_offset ) )
        self.__walls__.append( Wall( numpy.array([endX,endY]), numpy.array([startX,endY]), self.__ctrl__.box_dim[2], self.__textures__.crate[0], self.__ctrl__.wall_offset ) )
        self.__walls__.append( Wall( numpy.array([endX,startY]), numpy.array([endX,endY]), self.__ctrl__.box_dim[2], self.__textures__.crate[0], self.__ctrl__.wall_offset ) )
        self.__walls__.append( Wall( numpy.array([startX,startY]), numpy.array([endX,startY]), self.__ctrl__.box_dim[2], self.__textures__.crate[0], self.__ctrl__.wall_offset ) )
        return startX, startY, endX, endY
    
    def __constructDisplayList__( self ):
        # sort wall list by texture id
        self.__walls__ = sorted( self.__walls__, key = lambda wall: wall.texture )
        # start dispaly list
        self.__display_list__ = glGenLists(1)
        glNewList( self.__display_list__, GL_COMPILE )
        # floor quad
        l = self.__ctrl__.limits
        tex_id = self.__textures__.floor[0]
        glBindTexture( GL_TEXTURE_2D, tex_id )
        glBegin( GL_QUADS )
        glTexCoord2f( 0.0, 0.0 )
        glVertex3f( l[0], l[1], 0.0 )
        glTexCoord2f( 1.0, 0.0 )
        glVertex3f( l[2], l[1], 0.0 )
        glTexCoord2f( 1.0, 1.0 )
        glVertex3f( l[2], l[3], 0.0 )
        glTexCoord2f( 0.0, 1.0 )
        glVertex3f( l[0], l[3], 0.0 )
        # wall segments
        for w in self.__walls__:
            # change texture
            if w.texture != tex_id:
                glEnd()
                glBindTexture( GL_TEXTURE_2D, w.texture )
                glBegin( GL_QUADS )
            # wall segment quad
            glTexCoord2f( 0.0, 0.0 )
            glVertex3f( w.vec_from[0], w.vec_from[1], 0.0 )
            glTexCoord2f( 1.0, 0.0 )
            glVertex3f( w.vec_to[0], w.vec_to[1], 0.0 )
            glTexCoord2f( 1.0, 1.0 )
            glVertex3f( w.vec_to[0], w.vec_to[1], w.height )
            glTexCoord2f( 0.0, 1.0 )
            glVertex3f( w.vec_from[0], w.vec_from[1], w.height )
        # finish list
        glEnd()	
        glEndList()
        
    #--------------------------------------------------------------[ Utilities ]
    def validStep( self, pos_new ): 
        # check 1: new position still lies in the valid region
        #if self.validPosition( pos_new ) == False:
        #    return False
        # check 2: step does not cross any wall segments
        #for w in self.__walls__:
        #    if w.crossedBy( pos_old, pos_new):
        #        return False
        #return True
        length = int(self.__ctrl__.dim[0])
        width = int(self.__ctrl__.dim[1])
        stepBoard = self.__ctrl__.stepBoard
        
        if int(pos_new[0])<=0 or int(pos_new[0])>=(length-1) or int(pos_new[1])<=0 or int(pos_new[1])>=(width-1):
            return False
        else:
            cond = stepBoard[int(pos_new[0]) , int(pos_new[1])]!=0
            if cond:
                return False
        
        return True
    
    def validPosition( self, pos ):
        odd_nodes = False
        for w in self.__walls__:
            # break if too close to any wall
            if w.proximityAlert(pos): return False
            # run Point in Polygon algorithm (@ http://alienryderflex.com/polygon/)
            if (w.vec_from[1]<pos[1] and w.vec_to[1]>=pos[1]) or (w.vec_to[1]<pos[1] and w.vec_from[1]>=pos[1]):
                if (w.vec_from[0]+(pos[1]-w.vec_from[1])/(w.vec_to[1]-w.vec_from[1])*(w.vec_to[0]-w.vec_from[0]) < pos[0]):
                    odd_nodes = not odd_nodes
        return odd_nodes
    
    def randomPosition( self ):
        # init
        position = numpy.array( [0.0,0.0] )
        check    = False
        # find
        while check == False:
            position = numpy.array( [rnd.randrange(self.__ctrl__.limits[0],self.__ctrl__.limits[2]), \
		                          rnd.randrange(self.__ctrl__.limits[1],self.__ctrl__.limits[3])] )
            check = self.validStep( position )
        # return
        return position
    
    #----------------------------------------------------------------[ Drawing ]
    def drawWorld( self, focus ):
        
        # skybox (not optional atm)
        use_skybox = False
        if use_skybox:
            glDisable( GL_DEPTH_TEST )
            glBindTexture( GL_TEXTURE_2D, self.__textures__.skybox )
            glPushMatrix()
            glLoadIdentity()
            gluLookAt( 0.0, 0.0, focus[2],
					   focus[0], focus[1], focus[2],
					   0.0, 0.0, 1.0 )
            glBegin( GL_QUADS )	
            glTexCoord2f( 0.0, 0.0 )
            glVertex3f( -4.0, -4.0, 5.0 )
            glTexCoord2f( 0.5, 0.0 )
            glVertex3f(  4.0, -4.0, 5.0 )
            glTexCoord2f( 0.5, 0.5 )
            glVertex3f(  4.0, -4.0, 0.0 )
            glTexCoord2f( 0.0, 0.5 )
            glVertex3f( -4.0, -4.0, 0.0 )
            
            glTexCoord2f( 0.5, 0.0 )
            glVertex3f( -4.0, 4.0, 0.0 )
            glTexCoord2f( 1.0, 0.0 )
            glVertex3f(  4.0, 4.0, 0.0 )
            glTexCoord2f( 1.0, 0.5 )
            glVertex3f(  4.0, 4.0, 5.0 )
            glTexCoord2f( 0.5, 0.5 )
            glVertex3f( -4.0, 4.0, 5.0 )
            
            glTexCoord2f( 0.5, 0.5 )
            glVertex3f( -4.0, -4.0, 0.0 )
            glTexCoord2f( 1.0, 0.5 )
            glVertex3f( -4.0,  4.0, 0.0 )
            glTexCoord2f( 1.0, 1.0 )
            glVertex3f( -4.0,  4.0, 5.0 )
            glTexCoord2f( 0.5, 1.0 )
            glVertex3f( -4.0, -4.0, 5.0 )
            
            glTexCoord2f( 0.0, 0.5 )
            glVertex3f( 4.0, -4.0, 5.0 )
            glTexCoord2f( 0.5, 0.5 )
            glVertex3f( 4.0,  4.0, 5.0 )
            glTexCoord2f( 0.5, 1.0 )
            glVertex3f( 4.0,  4.0, 0.0 )
            glTexCoord2f( 0.0, 1.0 )
            glVertex3f( 4.0, -4.0, 0.0 )
            glEnd()
            
            glPopMatrix()
            glEnable( GL_DEPTH_TEST )
            
        glCallList( self.__display_list__ )
        
    #--------------------------------------------------------------[ Sketching ]
    
    def sketchWorld( self, sketch_uniform=False):
        # walls
        for w in self.__walls__:
            glColor3f( 0.0,0.0,0.0 )
            glBegin( GL_LINES )
            glVertex3f( w.vec_from[0], w.vec_from[1], def_MARKER_HEIGHT )
            glVertex3f( w.vec_to[0],   w.vec_to[1],   def_MARKER_HEIGHT )
            glEnd()
        # info: mark goals
        for g in self.__ctrl__.goals:
            self.sketchMarker( g[0], g[1] )
        
    def sketchPath( self, path ):
        glColor( self.__ctrl__.color_rat_path )
        glBegin( GL_POINTS )
        for p in path:
            glVertex3f( p[0], p[1], def_MARKER_HEIGHT )
        glEnd()
        
    def sketchArrow( self, pos_x, pos_y, dir_x, dir_y, color=None ):
        # color
        if color == None:
            glColor( 0.6, 0.0, 0.0 )
        elif color == 'red':
            glColor( 0.8, 0.0, 0.0 )
        elif color == 'green':
            glColor( 0.0, 0.6, 0.0 )
        elif color == 'blue':
            glColor( 0.0, 0.0, 0.8 )
        elif color == 'grey':
            glColor( 0.4, 0.4, 0.4 )
        # normalized direction x2
        dir_xn = (dir_x / (numpy.sqrt(dir_x**2+dir_y**2) + 1e-8))*2.0
        dir_yn = (dir_y / (numpy.sqrt(dir_x**2+dir_y**2) + 1e-8))*2.0
        # draw
        glBegin( GL_LINES )
        glVertex3f( pos_x+dir_xn*6.0, pos_y+dir_yn*6.0, def_MARKER_HEIGHT )
        glVertex3f( pos_x+dir_yn, pos_y-dir_xn, def_MARKER_HEIGHT )
        glVertex3f( pos_x+dir_yn, pos_y-dir_xn, def_MARKER_HEIGHT )
        glVertex3f( pos_x-dir_yn, pos_y+dir_xn, def_MARKER_HEIGHT )		
        glVertex3f( pos_x-dir_yn, pos_y+dir_xn, def_MARKER_HEIGHT )		
        glVertex3f( pos_x+dir_xn*6.0, pos_y+dir_yn*6.0, def_MARKER_HEIGHT )
        glVertex3f( pos_x, pos_y, def_MARKER_HEIGHT )
        glVertex3f( pos_x-dir_xn*4.0, pos_y-dir_yn*4.0, def_MARKER_HEIGHT )
        glEnd()
        
    def sketchMarker( self, pos_x, pos_y, size=None, color=None ):
        scale = 2.5
        if size == 'small': scale = 0.5
        # color
        glColor( self.__ctrl__.color_rat_marker )
        
        glBegin( GL_LINES )
        glVertex3f( pos_x-scale, pos_y, def_MARKER_HEIGHT )
        glVertex3f( pos_x+scale, pos_y, def_MARKER_HEIGHT )
        glVertex3f( pos_x, pos_y-scale, def_MARKER_HEIGHT )
        glVertex3f( pos_x, pos_y+scale, def_MARKER_HEIGHT )
        glVertex3f( pos_x, pos_y, def_MARKER_HEIGHT-scale )
        glVertex3f( pos_x, pos_y, def_MARKER_HEIGHT+scale )
        glEnd()

