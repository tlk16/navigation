# system
import sys

# math
import math
import numpy  as np
# np.random.seed(3)
import random as rnd
# rnd.seed(3)
# OpenGL
from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *

# utilities / own
from environment.util.freezeable import Freezeable

#------------------------------------------------------------------[ Constants ]

def_RAD2DEG         = 180.0/math.pi
def_DEG2RAD         = math.pi/180.0

#------------------------------------------------------------------[ Numpy Mod ]

np.seterr(divide='ignore') # ignore 'division by zero' errors (occur on path reset)


#==============================================================[ RatBot Class ]

class RatBot( Freezeable ):
    def __init__( self, pos, control ):
        """
        Constructor. Initializes the rat bot.
        pos    : Valid 2D position within the simulation world.
        control: Simulation control panel. Defined in ratlab.py.
        """
        # simulation control panel
        self.__ctrl__ = control
        # path
        self.__path__ = []
        self.__path__.append( pos )
        self.velocity = np.array([1,0])
        
        self.sample_dir = []
        self.sample_dir.append( ('e', np.array([ 1, 0])) )
        self.sample_dir.append( ('ne', np.array([ np.sqrt(0.5), np.sqrt(0.5)])) )
        self.sample_dir.append( ('n', np.array([ 0, 1])) )
        self.sample_dir.append( ('nw', np.array([-np.sqrt(0.5), np.sqrt(0.5)])) )
        self.sample_dir.append( ('w', np.array([-1, 0])) )
        self.sample_dir.append( ('sw', np.array([-np.sqrt(0.5),-np.sqrt(0.5)])) )
        self.sample_dir.append( ('s', np.array([ 0,-1])) ) 
        self.sample_dir.append( ('se', np.array([ np.sqrt(0.5),-np.sqrt(0.5)])) )
        
        
        
        # follow path if specified via file
        if control.setup.rat.path != None:
            f = open( './current_experiment/' + control.setup.rat.path )
            control.setup.rat.path = np.zeros( [sum(1 for l in f),2] )
            f.seek(0)
            for i,l in enumerate(f):
                c = l.split()
                control.setup.rat.path[i] = np.array([c[0],c[1]])
            self.__path_index__ = 1
            # reset starting position
            self.__path__[0] = control.setup.rat.path[0]
        # lockdown
        self.freeze()
        
    #-----------------------------------------------------------[ Path Control ]
    def getPath( self ):
        return self.__path__
    
    def __gaussianWhiteNoise2D__( self, dir):
        dir_n = dir / math.sqrt( dir[0]**2+dir[1]**2 )
        dir_a = math.asin( abs(dir_n[1]) ) * def_RAD2DEG
        if   dir_n[0]<=0 and dir_n[1]>=0: dir_a =180.0-dir_a
        elif dir_n[0]<=0 and dir_n[1]<=0: dir_a =180.0+dir_a
        elif dir_n[0]>=0 and dir_n[1]<=0: dir_a =360.0-dir_a
        rat_fov = 180.0
        angle   = (dir_a - rat_fov/2.0 + np.random.random_integers(0,1) * rat_fov) * def_DEG2RAD        
        return np.array( [math.cos(angle),math.sin(angle)] )
            
    def followPathNodes( self ):
        path = self.__ctrl__.setup.rat.path
        pos  = self.__path__[ len(self.__path__)-1 ]
        dist = np.sqrt(np.vdot(pos-path[self.__path_index__],pos-path[self.__path_index__]))
        if dist < self.__ctrl__.setup.rat.speed:
            self.__path_index__ += 1
            self.__path_index__ %= len(path)
            # end of non-loop path: teleport back to starting position
            if self.__path_index__ == 0 and self.__ctrl__.setup.rat.path_loop == False:
                pos_next    = path[0]
                trajectory  = np.array( path[1]-path[0], dtype=np.float32 )
                trajectory /= np.sqrt( np.vdot(trajectory,trajectory) )
                self.__path__.append( pos_next )
                return (pos_next, trajectory)
        # new step
        step  = np.array( path[self.__path_index__]-pos, dtype=np.float32 )
        step /= np.sqrt(np.vdot(step,step))
        noise = self.__ctrl__.setup.rat.path_dev
        while True:
            if np.random.random() > 0.5:
                step += np.array( [-step[1],step[0]] )*noise
            else:
                step += np.array( [step[1],-step[0]] )*noise
            step *= self.__ctrl__.setup.rat.speed
            # check for valid step
            pos_next = pos + step
            self.__path__.append( pos_next )
            return (pos_next, step)
    # setup next position, touch, and reward  
    def nextPathStep( self, randomChoose=True, action=None ):
        if self.__ctrl__.setup.rat.path != None:
            return self.followPathNodes()
        # current position & velocity/direction
        pos      = self.__path__[len(self.__path__)-1]
        touch = self.__ctrl__.modules.world.validTouch( pos )
        
        reward = None
        if len(self.__ctrl__.setup.world.goals) != 0:
            for g in self.__ctrl__.setup.world.goals:
                reward = int(np.sqrt((int(pos[0]) - g[0])**2 + (int(pos[1]) - g[1])**2) < 1e-3)
                if reward:
                    break
        
        pos_next = np.array([np.nan,np.nan])
        
        
        if randomChoose:
            if len(self.__path__) > 1: 
                vel = pos-self.__path__[len(self.__path__)-2]
            else:   
                loss = np.zeros(len(self.sample_dir))
                for index in range(len(self.sample_dir)):
                    loss[index] = np.sum( np.square( self.velocity - self.sample_dir[index][1] ) )
                r = np.argmin(loss)			
                vel = self.sample_dir[r][1]  
            
            
            # generate next step
            while True:
                noise = self.__gaussianWhiteNoise2D__(vel)
                mom   = self.__ctrl__.setup.rat.path_mom
                step  = vel*mom + noise*(1.0-mom)
                step /= np.sqrt(np.vdot(step,step))
                step *= self.__ctrl__.setup.rat.speed
                # optional movement bias
                bias  = self.__ctrl__.setup.rat.bias
                step += bias*(np.dot(bias,step)**2)*np.sign(np.dot(bias,step))*self.__ctrl__.setup.rat.bias_s
                # check for valid step
                pos_next = pos + step
                if self.__ctrl__.modules.world.validStep( pos_next ) == False: 
                    dir_n = vel / math.sqrt( vel[0]**2+vel[1]**2 )
                    dir_a = math.asin( abs(dir_n[1]) ) * def_RAD2DEG
                    if   dir_n[0]<=0 and dir_n[1]>=0: dir_a =180.0-dir_a
                    elif dir_n[0]<=0 and dir_n[1]<=0: dir_a =180.0+dir_a
                    elif dir_n[0]>=0 and dir_n[1]<=0: dir_a =360.0-dir_a
                    dir_a -= 45
                    angle = dir_a * def_DEG2RAD
                    vel = np.array([math.cos(angle),math.sin(angle)])
                else: 
                    self.__path__.append(pos_next)
                    break
        else:
            action = np.array(action)
            while True:
                pos_next = pos + action
                if self.__ctrl__.modules.world.validStep( pos_next ) == False:
                    dir_n = action / math.sqrt( action[0]**2+action[1]**2 )
                    dir_a = math.asin( abs(dir_n[1]) ) * def_RAD2DEG
                    if   dir_n[0]<=0 and dir_n[1]>=0: dir_a =180.0-dir_a
                    elif dir_n[0]<=0 and dir_n[1]<=0: dir_a =180.0+dir_a
                    elif dir_n[0]>=0 and dir_n[1]<=0: dir_a =360.0-dir_a
                    dir_a -= 45
                    angle = dir_a * def_DEG2RAD
                    action = np.array([math.cos(angle),math.sin(angle)])
                else:
                    self.__path__.append(pos_next)
                    break
        return (pos_next, pos_next-pos, touch, reward)