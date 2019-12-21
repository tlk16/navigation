# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:29:37 2019

@author: Dongye Zhao
"""
import logging
logging.raiseExceptions=False
import numpy

import environment.ratlab

# utilities
import sys
sys.path.append( './util' )
from environment.util.setup import *
import environment.util.world as world
import environment.util.ratbot as ratbot
import RNN
import torch

#=======================================================================[ Main ]
def main(RLfunc=True, collect=False, iters=None, dim=None, speed=None, box=None, bar=None, goal=None, limit=None, wall_offset=None, touch_offset=None):  
    
    if iters == None:
        iters = 1
    
    ctrl = ratlab.parms(collect=collect, dim=dim, speed=speed, box=box, goal=goal, limit=limit, wall_offset=wall_offset, touch_offset=touch_offset)
    ratlab.__setupGlut__(ctrl)
    ratlab.__setupOpenGL__(ctrl)
    ctrl.modules.world = world.World( ctrl.setup.world )
    ctrl.state.initPos = ctrl.modules.world.randomPosition()
    ctrl.modules.rat = ratbot.RatBot( ctrl.state.initPos, ctrl )
    
    for k in range(iters):
        ctrl.state.iters = k
        
        #==================================================================[RNN]
        rnn = RNN.RNN(4, 2, 512, 8)     
        hidden = rnn.initHidden()
        #======================================================================
        
        while True:
            #==============================================================[environment]
            ratlab.__display__(ctrl)
            #return: ctrl.state.last_view; ctrl.save.nextPos; ctrl.save.act; ctrl.save.touch; ctrl.save.reward
            #return: visual image;  position;  velocity;  touch:1-west, 2-south, 3-east, 4-north ;  reward:0-no reward, 1-have reward
            
            #==================================================================[RNN]
            if RLfunc:

                touch = torch.eye(4)[ctrl.save.touch - 1]
                act = torch.from_numpy(ctrl.save.act).type(torch.FloatTensor)
                out, hidden = rnn(touch, hidden, act) # ctrl.save.act [0,1] = 90 degree = North
                act_ = numpy.argmax(out.cpu().data.numpy().squeeze())
                act = ctrl.table.sample_dir[act_][1]
                print(act)
                ctrl.state.initAction = [float(act[0]), float(act[1])] # vel-x, vel-y
                print(ctrl.save.nextPos)

            #======================================================================    
                
            if ctrl.save.done:
                ctrl.modules.rat = ratbot.RatBot( ctrl.modules.world.randomPosition(), ctrl )
                ctrl.save.done = False
                ctrl.state.step = 0
                break




# iters = 5
# collect = True
# dim = [100,100,100]  # length, width, height
# box=[1,1]          # number of box obstacles, side of box obstacles
# goal=[30,30,2]       # center_x of goals, center_y of goals, radius of goals
# limit = 100
# wall_offset=2.   # >1
# touch_offset=3.  # >1
# main(collect=collect, iters=iters, dim=dim, box=box, goal=goal, limit=limit, wall_offset=wall_offset, touch_offset=touch_offset)





class RatEnv:
    def __init__(self, RLfunc=True, collect=False, iters=None, dim=None, speed=None, box=None, bar=None, goal=None, limit=None,
             wall_offset=None, touch_offset=None):

        self.ctrl = ratlab.parms(collect=collect, dim=dim, speed=speed, box=box, goal=goal, limit=limit,
                            wall_offset=wall_offset, touch_offset=touch_offset)
        ratlab.__setupGlut__(self.ctrl)
        ratlab.__setupOpenGL__(self.ctrl)
        self.ctrl.modules.world = world.World(self.ctrl.setup.world)
        self.ctrl.state.initPos = self.ctrl.modules.world.randomPosition()
        self.ctrl.modules.rat = ratbot.RatBot(self.ctrl.state.initPos, self.ctrl)

    def step(self, act):

        self.ctrl.state.initAction = [float(act[0]), float(act[1])]  # vel-x, vel-y
        ratlab.__display__(self.ctrl)
        # return: self.ctrl.state.last_view; self.ctrl.save.nextPos; self.ctrl.save.act; self.ctrl.save.touch; self.ctrl.save.reward
        # return: visual image;  position;  velocity;  touch:1-west, 2-south, 3-east, 4-north ;  reward:0-no reward, 1-have reward

        touch = numpy.zeros((4,))
        if self.ctrl.save.touch != 0:
            touch[self.ctrl.save.touch - 1] = 1

        return [self.ctrl.save.nextPos, numpy.asarray(self.ctrl.state.last_view), touch], self.ctrl.save.reward, self.ctrl.save.done, self.ctrl.state.step

    def reset(self):
        position0 = self.ctrl.modules.world.randomPosition()
        self.ctrl.modules.rat = ratbot.RatBot(position0, self.ctrl)
        self.ctrl.save.done = False
        self.ctrl.state.step = 0

        # print('hh1', position0)
        ratlab.__display__(self.ctrl)
        # print('hh2', self.ctrl.save.nextPos)
        self.ctrl.state.step = 0

        touch = numpy.zeros((4,))
        if self.ctrl.save.touch != 0:
            touch[self.ctrl.save.touch - 1] = 1
        return [self.ctrl.save.nextPos, numpy.asarray(self.ctrl.state.last_view), touch], self.ctrl.save.reward, self.ctrl.save.done, self.ctrl.state.step


if __name__ == '__main__':
    iters = 5
    collect = False
    dim = [100, 100, 100]  # length, width, height
    box = [1, 4]  # number of box obstacles, side of box obstacles
    goal = [30, 30, 10]  # center_x of goals, center_y of goals, radius of goals
    limit = 100
    wall_offset = 1.  # >1
    touch_offset = 2.  # >1

    ratenv = RatEnv(collect=collect, iters=iters, dim=dim, box=box, goal=goal, limit=limit, wall_offset=wall_offset, touch_offset=touch_offset)
    for k in range(iters):
        state, reward, done, step = ratenv.reset()
        print(state[0], state[2], reward, done, step)
        while not done:

            act = numpy.random.random((2,))
            print('action', act)

            state, reward, done, step = ratenv.step(act)
            print(state[0], state[2], reward, done, step)


# can't make sure whether the initial position is random
# the touch signal is on even when the distance to wall is slightly larger than touch_offset
# the shape of reward area is not sure, so not sure whether the postion-reward relation is correct.
# the problem of illegal goal






