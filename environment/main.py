# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:29:37 2019

@author: Dongye Zhao
"""
import logging
logging.raiseExceptions=False
import numpy

import ratlab

# utilities
import sys
sys.path.append( './util' )
from util.setup import *
import world
import ratbot
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
                ctrl.state.initAction = [float(act[0]), float(act[1])] # vel-x, vel-y
            #======================================================================    
                
            if ctrl.save.done:
                ctrl.modules.rat = ratbot.RatBot( ctrl.modules.world.randomPosition(), ctrl )
                ctrl.save.done = False
                ctrl.state.step = 0
                break
            
        


iters = 5
collect = True
dim = [100,100,100]  # length, width, height
box=[1,1]          # number of box obstacles, side of box obstacles
goal=[30,30,0]       # center_x of goals, center_y of goals, radius of goals
limit = 1000
wall_offset=2.   # >1
touch_offset=3.  # >1  
main(collect=collect, iters=iters, dim=dim, box=box, goal=goal, limit=limit, wall_offset=wall_offset, touch_offset=touch_offset)

