# system
import os
import shutil
import sys
import time

# math
import math
import numpy

# python image library
from PIL import Image as img

# OpenGL
from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *

# utilities
sys.path.append( './util' )
from util.setup import *
import world
import ratbot
import opengl_text as text
# import RNN
import torch


#--------------------------------------------------------------------[ control ]

class LocalControl( Freezeable ):
    def __init__( self ):
        
        # define: set for the current version
        self.defines = EmptyOptionContainer()	# defines
        self.defines.window_title  = 'RatLab v2.4'
        self.defines.window_width  =  800
        self.defines.window_height =  600
        self.defines.aspect_ratio  =  800.0/600.0
        self.defines.freeze()
        
        # config: no change during runtime
        self.config = EmptyOptionContainer()	# config
        self.config.record        = True
        self.config.limit         = None
        self.config.run_wallcheck = False
        self.config.freeze()
        
        # option: may change during runtime
        self.options = EmptyOptionContainer()	# options
        self.options.show_overview  = True
        self.options.show_ratview   = True
        self.options.show_progress  = False
        self.options.sketch_uniform = False
        self.options.ratview_scale  = 1
        self.options.freeze()
        
        # module: separate part of the program
        self.modules = EmptyOptionContainer()	# modules
        self.modules.world    = None
        self.modules.rat      = None
        self.modules.datafile = None
        self.modules.freeze()
        
        # state: set and used only by the program
        self.state = EmptyOptionContainer() 	# state
        self.state.step           = 0
        self.state.last_view      = None
        self.state.shot_count     = 0
        self.state.initPos = None
        self.state.initAction = None
        self.state.freeze()
        
        # save: return by the program
        self.save = EmptyOptionContainer()
        self.save.reward = 0
        self.save.done = False
        self.save.touch = 0
        self.save.act = None
        self.save.nextPos = None
        self.save.freeze()
        
        # table
        self.table = EmptyOptionContainer()
        self.table.sample_dir = []
        self.table.sample_dir.append( ('e', numpy.array([ 1, 0])) ) # 0
        self.table.sample_dir.append( ('ne', numpy.array([ numpy.sqrt(0.5), numpy.sqrt(0.5)])) ) # 1
        self.table.sample_dir.append( ('n', numpy.array([ 0, 1])) ) # 2
        self.table.sample_dir.append( ('nw', numpy.array([-numpy.sqrt(0.5), numpy.sqrt(0.5)])) ) # 3
        self.table.sample_dir.append( ('w', numpy.array([-1, 0])) ) # 4
        self.table.sample_dir.append( ('sw', numpy.array([-numpy.sqrt(0.5),-numpy.sqrt(0.5)])) ) # 5
        self.table.sample_dir.append( ('s', numpy.array([ 0,-1])) ) # 6
        self.table.sample_dir.append( ('se', numpy.array([ numpy.sqrt(0.5),-numpy.sqrt(0.5)])) ) # 7
        self.table.freeze()
        
        self.setup = Setup()					# global control
        self.freeze()
        
ctrl = LocalControl()

#===================================================================[ callback ]
def __keyboardPress__( key, x, y ):
    
    # c: switch uniform sketch
    if key == 'c':
        ctrl.options.sketch_uniform = not ctrl.options.sketch_uniform
        
    # ESC: quit program
    elif ord(key) == 27:
        if ctrl.config.record: 	
            screenshot = glReadPixels( 0,0, ctrl.defines.window_width, ctrl.defines.window_height, GL_RGBA, GL_UNSIGNED_BYTE)
            im = img.frombuffer('RGBA', (ctrl.defines.window_width,ctrl.defines.window_height), screenshot, 'raw', 'RGBA', 0, 0)
            im.save('./save/exp_finish.png')
            sys.exit()
        
def __keyboardSpecialPress__( key, x, y ):
    
    # F1: show/hide map overview
    if key == GLUT_KEY_F1:
        ctrl.options.show_overview = not ctrl.options.show_overview
        
    # F2: show/hide ratview
    elif key == GLUT_KEY_F2 and ctrl.config.record == False:
        ctrl.options.show_ratview = not ctrl.options.show_ratview
        
    # F3: show/hide progress bar
    elif key == GLUT_KEY_F3 and ctrl.config.limit != None:
        ctrl.options.show_progress = not ctrl.options.show_progress
        
#===============================================================[ OpenGL Setup ]
def __setupGlut__():
    
    # create GLUT window
    glutInit( sys.argv )
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH  )
    glutInitWindowSize( ctrl.defines.window_width, ctrl.defines.window_height )
    glutCreateWindow( ctrl.defines.window_title )
    
    # display
    glutDisplayFunc( __display__ )
    glutTimerFunc( 0, __drawcall__, 1 )
    
    # callback
    glutKeyboardFunc( __keyboardPress__ )
    glutSpecialFunc( __keyboardSpecialPress__ )
    
    # set cursor
    glutSetCursor( GLUT_CURSOR_CROSSHAIR )
    
def __setupOpenGL__():
    
    # projection matrix setup w/ default viewport
    glViewport( 0, 0, ctrl.defines.window_width, ctrl.defines.window_height )
    
    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    gluPerspective( ctrl.setup.rat.fov[1], ctrl.defines.aspect_ratio, ctrl.setup.opengl.clip_near, ctrl.setup.opengl.clip_far )
    
    # viewing matrix setup
    glMatrixMode( GL_MODELVIEW )
    glLoadIdentity()
    
    # misellaneous parameters
    clear_color = ctrl.setup.world.color_background
    glClearColor( clear_color[0], clear_color[1], clear_color[2], 0.0 )
    glLineWidth( 2 )
    glPointSize(2.0)
    glLineStipple( 1, 0xAAAA )
    
    # depth buffer
    glClearDepth( 1.0 )
    glEnable( GL_DEPTH_TEST )
    glDepthFunc( GL_LEQUAL )
    
    # backface culling
    glCullFace( GL_BACK )
    glEnable( GL_CULL_FACE )
    
    # texture mapping
    glEnable( GL_TEXTURE_2D )
    
#====================================================================[ Drawing ]

def __drawcall__( i ):
    glutPostRedisplay()
    glutTimerFunc( 0, __drawcall__, 1 )
    
def __display__():
    
    # main reset
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
    glBindTexture( GL_TEXTURE_2D, 0 )
    
    # get rat's next state
    if ctrl.state.initAction is not None:
        randomChoose = False
    else:
        randomChoose = True
    rat_state = ctrl.modules.rat.nextPathStep(randomChoose, ctrl.state.initAction) #pos_next, pos_next-pos, touch, reward
    # print(rat_state)
    
    if True:
        #-----------------------------------------------[ map overview / wallcheck ]
        if ctrl.options.show_overview == True or ctrl.config.run_wallcheck == True:
            glViewport( 40, 150, 720, 450 )
            glMatrixMode( GL_PROJECTION )
            glLoadIdentity()
            gluPerspective( 40.0, 720.0/450.0, 2.0, 512.0 ) # fov, ratio, near, far
            glMatrixMode( GL_MODELVIEW )
            glLoadIdentity()
            # camera
            limits = ctrl.setup.world.limits
            gluLookAt( (limits[0]+limits[2])/2.0, -limits[3], 300, (limits[0]+limits[2])/2.0,  (limits[1]+limits[3])/2.0, 0, 0.0, 0.0, 1.0 )
            # default overview render
            if ctrl.config.run_wallcheck == False:
                ctrl.modules.world.sketchWorld( sketch_uniform = ctrl.options.sketch_uniform )
                ctrl.modules.world.sketchPath ( ctrl.modules.rat.getPath() )
                ctrl.modules.world.sketchArrow( rat_state[0][0], rat_state[0][1], rat_state[1][0], rat_state[1][1] )
            # wallcheck render
            else:
                for i, raster in enumerate([False,True]):
                    ctrl.modules.world.sketchWorld( sketch_info = True, raster=raster )
                    # read frame buffer into image
                    screenshot = glReadPixels( 0,0, ctrl.defines.window_width, ctrl.defines.window_height, GL_RGBA, GL_UNSIGNED_BYTE )
                    im = img.frombuffer('RGBA', (ctrl.defines.window_width,ctrl.defines.window_height), screenshot, 'raw', 'RGBA', 0, 0 )
                    im.save( './current_experiment/wallcheck_'+('raster' if raster==True else '')+'.png' )
                # exit
                sys.exit()
            
        #-----------------------------------------------------------[ progress bar ]
        if ctrl.options.show_progress:
            glViewport( 0, 0, 800, 600 )
            glMatrixMode( GL_PROJECTION )
            glLoadIdentity()
            gluOrtho2D( -300, 300, -300, 300 )
            glMatrixMode( GL_MODELVIEW )
            glLoadIdentity()
            text.drawProgressBar( ctrl.config.limit, ctrl.state.step+1, (-130,-250) )
            
        #-----------------------------------------------------[ rat view rendering ]
        if ctrl.options.show_ratview == True:
            glColor( 1.0, 1.0, 1.0 )
            dir_n  = numpy.array( [rat_state[1][0],rat_state[1][1]] )
            dir_n /= math.sqrt( dir_n[0]**2 + dir_n[1]**2 + 1e-8)
            dir_a  = math.asin( abs(dir_n[1]) ) * ctrl.setup.constants.RAD2DEG
                
            if   dir_n[0]<=0 and dir_n[1]>=0: dir_a =180.0-dir_a
            elif dir_n[0]<=0 and dir_n[1]<=0: dir_a =180.0+dir_a
            elif dir_n[0]>=0 and dir_n[1]<=0: dir_a =360.0-dir_a
            x = int( ctrl.defines.window_width/2 - ctrl.setup.rat.fov[0]/2*ctrl.options.ratview_scale )
            for i in range( int(dir_a-ctrl.setup.rat.fov[0]/2), int(dir_a+ctrl.setup.rat.fov[0]/2)+1 ):
                glViewport( x, 80, 1*ctrl.options.ratview_scale, int(ctrl.setup.rat.fov[1])*ctrl.options.ratview_scale )
                glMatrixMode( GL_PROJECTION )
                glLoadIdentity()
                gluPerspective( ctrl.setup.rat.fov[1], 1.0/ctrl.setup.rat.fov[1], ctrl.setup.opengl.clip_near, ctrl.setup.opengl.clip_far )
                glMatrixMode( GL_MODELVIEW )
                glLoadIdentity()
                focus = [ rat_state[0][0]+math.cos(i*ctrl.setup.constants.DEG2RAD)*100.0,
                         rat_state[0][1]+math.sin(i*ctrl.setup.constants.DEG2RAD)*100.0,
                         ctrl.setup.world.cam_height ]
                    
                gluLookAt( rat_state[0][0], rat_state[0][1], ctrl.setup.world.cam_height,
                          focus[0], focus[1], focus[2], 
                          0,0,1 )
                ctrl.modules.world.drawWorld( focus )
                x+=ctrl.options.ratview_scale
                    
        #---------------------------------------------------[ simulation recording ]
        # read out current rat view image
        opengl_buffer = glReadPixels( ctrl.defines.window_width/2 - ctrl.setup.rat.fov[0]/2*ctrl.options.ratview_scale,
                                     80, ctrl.setup.rat.fov[0], ctrl.setup.rat.fov[1], GL_RGBA, GL_UNSIGNED_BYTE )
            
        ctrl.state.last_view  = img.frombuffer( 'RGBA', (int(ctrl.setup.rat.fov[0]),int(ctrl.setup.rat.fov[1])), 
                                               opengl_buffer, 'raw', 'RGBA', 0, 0 )
        
        ctrl.save.nextPos = rat_state[0] 

        if rat_state[1][0]<1e-5 and rat_state[1][0]>0:
            rat_state[1][0] = 0.
        elif rat_state[1][1]<1e-5 and rat_state[1][1]>0:
            rat_state[1][1] = 0.
        ctrl.save.act = rat_state[1]
        ctrl.save.touch = rat_state[2]
        ctrl.save.reward = rat_state[3]
        ctrl.state.step += 1
        
        collect()
        
        # runtime limit
        if ctrl.save.done or (ctrl.config.limit != None and ctrl.state.step == ctrl.config.limit):
            screenshot = glReadPixels( 0,0, ctrl.defines.window_width, ctrl.defines.window_height, GL_RGBA, GL_UNSIGNED_BYTE)
            im = img.frombuffer('RGBA', (ctrl.defines.window_width,ctrl.defines.window_height), screenshot, 'raw', 'RGBA', 0, 0)
            im.save('./save/exp_finish.png')
            sys.exit()
        
            
    #------------------------------------------------------------[ end drawing ]
    glutSwapBuffers()
 

#=======================================================================[ Collect ]

def collect():
    # save current rat view to the image sequence
    if( os.path.isdir('./save/sequence') == False ):
        os.mkdir('./save/sequence')
    ctrl.state.last_view.save( './save/sequence/frame_'+str(ctrl.state.step).zfill(5)+'.png' )
    # collect movement data
    ctrl.modules.datafile = open( './save/exp_trajectory.txt', 'a' )
    if len(ctrl.setup.world.goals) != 0:
        ctrl.modules.datafile.write( str(ctrl.state.step) + ' ' + str(ctrl.save.touch) + ' ' + str(ctrl.save.reward) + ' ' + str(ctrl.save.act) + ' ' + str(ctrl.save.nextPos) + '\n' )
    else:
        ctrl.modules.datafile.write( str(ctrl.state.step) + ' ' + str(ctrl.save.touch) + ' ' + str(ctrl.save.act) + ' ' + str(ctrl.save.nextPos) + '\n' )
    ctrl.modules.datafile.close()

#=======================================================================[ Main ]

def main(func=None, dim=None, speed=None, box=None, bar=None, goal=None, limit=None, wall_offset=None, touch_offset=None):
    
    if( os.path.isdir('./save') == False ):
        os.mkdir('./save')
    else:
        shutil.rmtree('./save')
        os.mkdir('./save') 
    #if( os.path.exits('./save/exp_trajectory.txt') == True ):
    #    os.remove('./save/exp_trajectory.txt')
        
    if dim != None:                # [10,10,100]
        ctrl.setup.world.dim = numpy.array(dim)    
    if speed != None:              # 1.0
        ctrl.setup.rat.speed = float(speed)   
    if box != None:                # [1, 1]
        ctrl.setup.world.obstacles.append( [float(box[0]), float(box[1]), float(box[1])] ) # obstacles' number, side_length, side_width
    if bar != None:    
        ctrl.setup.world.obstacles.append( [float(bar[0]), float(bar[1])] ) # obstacles' number, width
    if goal != None:               # [4,5,2]
        center_x = float(goal[0])
        center_y = float(goal[1])
        radius = float(goal[2])
        region_x = range(int(center_x - radius), int(center_x + radius + 1))
        region_y = range(int(center_y - radius), int(center_y + radius + 1))
        for m in region_x:
            for n in region_y:
                ctrl.setup.world.goals.append( [m, n] ) # x, y   
    if limit != None:              # 1000
        ctrl.config.limit = float(limit)
        ctrl.options.show_progress = True
    if wall_offset != None:             # >1
        ctrl.setup.world.wall_offset = float(wall_offset)    # the safe distance to avoid hitting a wall
    if touch_offset != None:             # >1
        ctrl.setup.world.touch_offset = float(touch_offset)  # within the distance from a wall, providing a touch signal
        
    ctrl.setup.toFile('./save/exp_setup')
    
    __setupGlut__()
    __setupOpenGL__()
    
    # setup world and agent  
    ctrl.modules.world = world.World( ctrl.setup.world )
    ctrl.state.initPos = ctrl.modules.world.randomPosition()
    ctrl.modules.rat = ratbot.RatBot( ctrl.state.initPos, ctrl )
    
#     rnn = RNN.RNN(4, 2, 512, 8)
#     print ('size',rnn.i2h.size(), rnn.a2h.size())
#     # action loop
#     hidden = rnn.initHidden()
# #     Hiddens = []
    while True:
        #touch 1-west; 2-south; 3-east; 4-north      
        __display__()
        if func == None:
            touch = torch.eye(4)[ctrl.save.touch - 1]
            act = torch.from_numpy(ctrl.save.act).type(torch.FloatTensor)
#             print (touch.size(), touch.size(), act.size())
            out, hidden = rnn(touch, hidden, act) # ctrl.save.act [0,1] = 90 degree = North
            Hiddens.append(hidden)
#             act_ = numpy.argmax(out.cpu().data.numpy().squeeze())
#             act = ctrl.table.sample_dir[act_][1]

#             print ('output', act_, act)
            ctrl.state.initAction = [float(act[0]), float(act[1])] # vel-x, vel-y
            
        

# main(func =5, dim = [100,100,100], speed = 1., box=[1,1], goal=[3,3,0], limit = 100, wall_offset=2., touch_offset=3.)

class Ratenv:
    def __init__(self, dim = [100,100,100], speed = 1., box=[1,1], bar=None, goal=[3,3,0], limit = 100, wall_offset=2., touch_offset=3.):
        if (os.path.isdir('./save') == False):
            os.mkdir('./save')
        else:
            shutil.rmtree('./save')
            os.mkdir('./save')
            # if( os.path.exits('./save/exp_trajectory.txt') == True ):
            #    os.remove('./save/exp_trajectory.txt')

        if dim != None:  # [10,10,100]
            ctrl.setup.world.dim = numpy.array(dim)
        if speed != None:  # 1.0
            ctrl.setup.rat.speed = float(speed)
        if box != None:  # [1, 1]
            ctrl.setup.world.obstacles.append(
                [float(box[0]), float(box[1]), float(box[1])])  # obstacles' number, side_length, side_width
        if bar != None:
            ctrl.setup.world.obstacles.append([float(bar[0]), float(bar[1])])  # obstacles' number, width
        if goal != None:  # [4,5,2]
            center_x = float(goal[0])
            center_y = float(goal[1])
            radius = float(goal[2])
            region_x = range(int(center_x - radius), int(center_x + radius + 1))
            region_y = range(int(center_y - radius), int(center_y + radius + 1))
            for m in region_x:
                for n in region_y:
                    ctrl.setup.world.goals.append([m, n])  # x, y
        if limit != None:  # 1000
            ctrl.config.limit = float(limit)
            ctrl.options.show_progress = True
        if wall_offset != None:  # >1
            ctrl.setup.world.wall_offset = float(wall_offset)  # the safe distance to avoid hitting a wall
        if touch_offset != None:  # >1
            ctrl.setup.world.touch_offset = float(
                touch_offset)  # within the distance from a wall, providing a touch signal

    def reset(self):
        ctrl.setup.toFile('./save/exp_setup')

        __setupGlut__()
        __setupOpenGL__()

        # setup world and agent
        ctrl.modules.world = world.World(ctrl.setup.world)
        ctrl.state.initPos = ctrl.modules.world.randomPosition()
        ctrl.modules.rat = ratbot.RatBot(ctrl.state.initPos, ctrl)

        return [ctrl.save.nextPos, ctrl.state.last_view, ctrl.save.touch], ctrl.save.reward

    def step(self, action):
        ctrl.state.initAction = [float(action[0]), float(action[1])]  # vel-x, vel-y
        __display__()
        return [ctrl.save.nextPos, ctrl.state.last_view, ctrl.save.touch], ctrl.save.reward

if __name__ == '__main__':
    import numpy as np
    ratenv = Ratenv()
    state, reward = ratenv.reset()
    print(state, reward)
    for i in range(10):
        action = np.random.random((2,))
        state, reward = ratenv.step(action)
        print(i, action, state, reward)



