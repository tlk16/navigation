# Modify the training algorithm with david silver memroy based control with recurrent neural networks   

# the training pipelines : 1,  initiailize h0  2, forward dynamics RNN + environments get data, o1, a1, r1.....  3,  store them  4,  smaple them from R  5,  reconstruct h for training 6, compute target by TD for each batch of data(cannot be the one used in forward step)  7 compute gradient 8 possibly use delay weight strategy to stabilize 

# future step, use critic to make it ddpg 

import sys
sys.path.insert(0,'/home/tie/Research/Navigation/NavigationPaper_925/Qlearn')

import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import init
from torch.nn import DataParallel

import torch
import torch.utils.data

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML

from net.Recurrent import RNN, RLS
from net.lstm import LstmNet
from envs.base import BaseGame
from envs.envs import Game

import os, psutil


# Dict_tasks = {'basic': basic, 'bar':barset, 'hole':hole}




class ReinforceGame(Game):

    def __init__(self, e = 0, holes = 0, grid_size = (15, 15), random_seed = 0, set_reward = ((0.5, 0.25), (0.5, 0.75)) , time_limit = 200, input_type = 0, lam = 0.5, discount = 0.99, alpha = 0.5, implicit = 'False', task = 'basic', gpu = 0, Net = 'RNN', train_readout = False):
        Game.__init__(self, discount=discount, grid_size=grid_size, time_limit=time_limit,
                         random_seed=random_seed, set_reward=set_reward, input_type=input_type)
        # need to have randomness
        self.e = e
        self.gpu = gpu
        if Net == 'RNN':
            self.net = RNN(9, 512, 4,).cuda(self.gpu)
        elif Net == 'lstm':
            self.net = LstmNet((9, 512), 512, (512, 4)).cuda(self.gpu)
        if Net == 'RNN':
            self.hidden = self.net.initHidden().cuda(self.gpu)
            self.action = self.net.initAction().cuda(self.gpu)
            self.policy = 0.25 * torch.ones(4).cuda(self.gpu)
        self.Loss = 0
        self.lr = 0
        self.Life = []
        self.Succeed = []
        self.trace = []
        self.hiddens = []
        self.Actions = []
        self.Visions = []
        self.Contexts = []
        self.Rewards = []
        self.Qs = []
        self.trace = []
        self.time_limit = time_limit
        # running avearage rate
        self.alpha = alpha
        # backward ratio
        self.lam = lam
        self.succeed = 0
        self.life = 0
        self.y_mid = 0
        self.x_mid = 0
        self.holes = holes
        # control the map seed, if train == true then render seed between 0 , 1
        self.implicit = implicit
        self.task = task
        self.train_readout = train_readout

    def sample(self):
        # choose between 0, 1,2,3
        np.random.seed()
        return np.random.randint(0,4)

    def placefield(self, pos):
        pos_ = (pos[0] - self.y_mid, pos[1] - self.x_mid)
#         print (pos_, self.set_reward)
#         print (self.grid_size)
        field =np.zeros((2, 19))
#         print (len(field))
        for k in range(2):
            for i in range(field.shape[1]):
            # distance generation
                pos_relative = pos[k] * 19./(self.grid_size[1] + 4)
                field[k, i] =  (i- pos_relative) ** 2
#                 print (i - pos[k])e
        # gaussian density, but before exponential to help learning identity mapping input to output
        field = - 0.1 * torch.from_numpy(field).cuda(self.gpu).resize(1, 2 * 19).float()
        return field
# get the softmax policy vector 
    def softmaxplay(self, state):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        # value for four action
        policy0 = self.policy.clone().cuda(self.gpu)
        action0 = self.action.clone().cuda(self.gpu)
        if self.implicit == True:
            self.policy, self.values, self.hidden = self.net(state.cuda(self.gpu), self.hidden.cuda(self.gpu), self.action.cuda(self.gpu), self.placefield(self.pos_reward_).cuda(self.gpu))
        else:
            # print(self.action, self.hidden, self.placefield(self.pos_reward), state)
            self.policy, self.values, self.hidden = self.net(state.cuda(self.gpu), self.hidden.cuda(self.gpu), self.action.cuda(self.gpu), self.placefield(self.pos_reward).cuda(self.gpu))
        # action to state
#         action = np.random.choice(np.arange(0, 4, 1), 1, p = policy)
#         self.action = torch.eye(4)[action].resize(1, 4).cuda(self.gpu)
        return policy0, action0
#  decode is binary
    def decode(self):
        pos = self.hidden.matmul(self.net.h2p_rls) + self.net.bp_rls
        return pos
    # test is for testing phase, decode is to train decoder
    def step(self, Policy, epsilon = 'default', train_hidden = False, decode = False, test = False):
        if epsilon != 'default':
            self.e = epsilon
        self.t += 1
        """enviroment dynamics Update state (grid and agent) based on an action"""
        # state to action
        state_t0 = self.visible_state
        pos0 = self.agent.pos
        # network dynamics and decision
        policy0, action0 = Policy(torch.FloatTensor(state_t0).cuda(self.gpu).resize(1, 9))
        # sampling action from policy   
        action = np.random.choice(4, 1, p = self.policy.cpu().data.numpy().ravel())
#         print (self.policy.cpu().data.numpy().ravel())
        self.action = torch.eye(4)[action].resize(1, 4).cuda(self.gpu)
        # action to state
        self.agent.act(action)
        # check wall clicking and reward
        pos1 = self.agent.pos
        def wall(pos1):
            y, x = pos0
            # wall detection
            if self.grid.grid[pos1] < 0:
                self.agent.pos = pos0
                pos_possible = [pos for pos in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)] if
                                self.grid.grid[pos] >= 0]
                self.agent.pos = pos_possible[np.random.randint(len(pos_possible))]
                pos1 = self.agent.pos

                self.t += 1
                return True
            else:
                return False
        def rewarding(pos1):
            # punish collide
            collision = wall(pos1)
            # punishment by cold water or click
            if collision == True:
                reward = -0.5
            else:
                reward = -0.01
            if self.grid.grid[pos1] > 0:
                reward = self.grid.grid[pos1]
            # death
            elif self.t >= self.time_limit:
                reward = -1
            # Check if agent won (reached the goal) or lost (health reached 0)
            # attention! 需要括号， 否则reward会被更新
            done = (reward > 0 or self.t >= self.time_limit)
            return reward, done
        def rewarding_hole(pos1):
            # punish collide
            collision = wall(pos1)
            # punishment by cold water or click
#                 if collision == True:
#                     reward = -0.5
            # else:
            reward = -0.01
            if self.grid.grid[pos1] > 0:
                reward = self.grid.grid[pos1]
            # death
            elif self.t >= self.time_limit:
                reward = -1
            # Check if agent won (reached the goal) or lost (health reached 0)
            # attention! 需要括号， 否则reward会被更新
            done = (reward > 0 or self.t >= self.time_limit)
            return reward, done
        def rewarding_bar(pos1):
            # punish collide
            collision = wall(pos1)
            # punishment by cold water or click
#                 if collision == True:
#                     reward = -0.5
            # else:
            reward = -0.01
            reward = reward - 0.01 * np.int(action != torch.max(action0).cpu().data.numpy())
            if self.grid.grid[pos1] > 0:
                reward = self.grid.grid[pos1]
            # death
            elif self.t >= self.time_limit:
                reward = -1
            # Check if agent won (reached the goal) or lost (health reached 0)
            # attention! 需要括号， 否则reward会被更新
            done = (reward > 0 or self.t >= self.time_limit)
            return reward, done

        def rewarding_scale(pos1):
            # punish collide
            collision = wall(pos1)
            # punishment by cold water or click
#                 if collision == True:
#                     reward = -0.5
            # else:
            reward = -0.01
            if self.grid.grid[pos1] > 0:
                reward = self.grid.grid[pos1]
            # death
            elif self.t >= self.time_limit:
                reward = -1
            done = (reward > 0 or self.t >= self.time_limit)
            return reward, done
        # rewarding accdonig to tasks
        if self.task == 'hole':
            reward, done = rewarding_hole(pos1)
        elif self.task == 'bar':
            reward, done = rewarding_bar(pos1)
        elif self.task == 'scale':
            reward, done = rewarding_scale(pos1)
        else:
            reward, done = rewarding(pos1)
        # render next state for terminate condition check after wall condition
        state_t1 = self.visible_state
        # record action, vision, position
        if test == False:
            self.Pos.append(self.placefield(self.agent.pos))
            self.Actions.append(action0)
            self.Visions.append(torch.FloatTensor(state_t0).resize(1, 9).cuda(self.gpu))
            self.Rewards.append(torch.FloatTensor([reward]).cuda(self.gpu))
        elif decode == True:
            self.Pos.append(torch.FloatTensor([pos0[0], pos0[1]]).resize(1, 2)).cuda(self.gpu)
#         print (pos1)
        return state_t1, action, reward, done

    def velocity(self, stim=torch.zeros(1, 9), hidden0=torch.randn(1, 512, requires_grad=True), action=4):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        # value for four action
        if action <= 3:
            self.action = torch.eye(4)[action].resize(1, 4)
        else:
            self.action = 0.25 * torch.ones(4).resize(1, 4)
        velocity = self.net.velocity(stim, hidden0, self.action, self.placefield(self.pos_reward))
        return velocity

    def reset_body(self, train = True, reset_pos = None, reward_control = 0, init_net = False, size = 15, size_range = [15], limit_set = 8):
        func = getattr(self, self.task)
        self.reset_env(func, train = train, size = size, size_range = size_range, reward_control = reward_control, limit_set = limit_set)
        if init_net == True:
            self.hidden = self.net.initHidden()
            # set action
            self.action = self.net.initAction()
        if reset_pos != None:
            self.agent.pos = reset_pos
#   record N sessions of game, save reward, vision, actions in buffer
    def episode(self, epochs=10, epsilon='default', reward_control=None, size_range=(10, 20),
                train_hidden = True, test=False, decode=False, implicit = False):
        # randomnize context
        if reward_control == None:
            self.reset_body(reward_control=np.random.randint(len(self.set_reward)), size_range=size_range, init_net = True)
        else:
            self.reset_body(reward_control=reward_control, size_range=size_range, init_net = True)
        done = False
        # train only hidden to output
        if epsilon != 'default':
            self.e = epsilon
        k = 0
        self.Actions_batch = []
        self.Visions_batch = []
        self.Rewards_batch = []
        self.Contexts = []
        self.Hiddens0 = []
        def Done(state, action, init_net = True):
            # data record
            # print (self.Actions, self.Visions, self.Rewards)
            self.Actions.append(self.action)
            self.Visions.append(torch.FloatTensor(state).resize(1, 9).cuda(self.gpu))
            self.Actions_batch.append(torch.stack(self.Actions))
            self.Visions_batch.append(torch.stack(self.Visions))
            self.Rewards_batch.append(torch.stack(self.Rewards))
            self.trace = []
            self.Qs = []
            self.Actions = []
            self.Visions = []
            self.Rewards = []
            # reset
            done = False
            if reward_control == None:
                self.reset_body(reward_control=np.random.randint(len(self.set_reward)),size_range=size_range, init_net = init_net)
            else:
                self.reset_body(reward_control=reward_control, size_range = size_range, init_net = init_net)
        # start record
        for i in range(epochs):
            # start episode
            self.reset_body(reward_control=np.random.randint(len(self.set_reward)), size_range=size_range, init_net=True)
            self.Hiddens0.append(self.hidden.cuda(self.gpu))
            done = False
            while done == False:
                state_t1, action, reward, done = self.step(self.softmaxplay, epsilon=epsilon, train_hidden=train_hidden,
                                                      test=test, decode=decode)

            Done(state_t1, action, init_net = True)
            self.Contexts.append(self.placefield(self.pos_reward).cuda(self.gpu))
            if implicit == True:
                self.Contexts.append(self.placefield(self.pos_reward_).cuda(self.gpu))
                
    # update value function, pass the action, vsion , initial hidden states and reward history, recursively change the values function until converges,, compute the qs
    def TD(self, Q_now, Q_next, action, reward):
        # target Q is for state before updated, it only needs to update the value assocate with action taken
        targetQ = Q_now.clone().detach()
        # new Q attached with the new state
        # max of q for calculating td error
        Qmax = torch.max(Q_next)
        delta  =  torch.FloatTensor([reward]).cuda(self.gpu) + self.discount*Qmax  - targetQ[0, action]

        # eligilibty trace for updating all last states before because of the information about new state
        self.trace = [e * self.discount * self.lam for e in self.trace]
        # eligibility trace attach new state
        self.Qs.append(targetQ)
        self.trace.append(1)
        # corresponding features h
        # update all last action values with eligibility trace, the q will add a new updated value
        def f(e, delta, q):
#             print (e, delta, q)
            q[0, action] = q[0, action] + self.alpha * delta * e
            return q
#         print (self.trace)
        self.Qs = [f(e, delta, q) for e, q in zip(self.trace, self.Qs)]

    # turn histroy of data into a target vector, the action is this step chosen action, so action1, the reward is this stepreward, so the reward0
    def Value_backward(self, Predicts, Actions, Rewards):
        self.trace = []
        # print (len(Actions), len(Rewards), len(Predicts))
        for Q_now, Q_next, action, reward in zip(Predicts[:-1], Predicts[1:], Actions[1:], Rewards):
            action = np.int(torch.argmax(action).cpu().data.numpy())
            # print (action)
            self.TD(Q_now, Q_next, action, reward)


    # if use random tensor , there is no memory leak,  if use hiddens and targets for beta but not really update weight, still leak,  so the error is in between. Only recording vision aciton 
    def train_sgd(self, lr_rate=1e-5):
        # it is important to keep readout update quick 

        Optimizer_q = torch.optim.Adam(
                [
                    {'params': self.net.i2h, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.a2h, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2h, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.bh, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2q, 'lr': 10 * lr_rate, 'weight_decay': 0},
                    {'params': self.net.bq, 'lr': 10 * lr_rate, 'weight_decay': 0},
                ]
            )

        Optimizer_p = torch.optim.Adam(
                [
                    {'params': self.net.h2p, 'lr': 1 * lr_rate, 'weight_decay': 0},
                    {'params': self.net.bp, 'lr': 1 * lr_rate, 'weight_decay': 0},
                ]
            )
        # read N = epoch histories of data, compute values functions, all hidden states, and do back prob
        Data = [(v, a, r, h, c) for v, a, r, h, c in zip(self.Visions_batch, self.Actions_batch, self.Rewards_batch , self.Hiddens0, self.Contexts)]
        # sample history one by one
        Loss_p = 0
        Loss_q= 0
#         print('loss', Loss, 'h2p', self.net.h2p, 'h2q', self.net.h2q)
        for (Visions, Actions, Rewards, hidden0, context) in Data:
            # reconstruct hidden states
            policys, q_predicts = self.net.forward_sequence_values(Visions, Actions, hidden0, context)
            # backward to compute values, but the backward values should be from last state
#             print('policy', policys)
            self.Value_backward(q_predicts, Actions, Rewards)
            if len(self.Qs)>=2:
                loss_q = torch.mean((q_predicts[:-1] - torch.stack(self.Qs)) ** 2)
                loss_q.backward()
                for p, name in zip([self.net.i2h, self.net.a2h, self.net.h2h, self.net.bh], ['i2h','a2h', 'h2h', 'bh']):
                    p.grad.data.clamp_(-1, 1)
                Optimizer_q.step()
                self.net.zero_grad()
                # stop gradient here, so that the loss gradient will not go through Qnet during the actor period
                loss_p = - torch.mean(q_predicts.detach().clone() * policys)
#                 print ('p', policys[0],  'logp', torch.log(policys[0]))
                loss_p.backward()
                Optimizer_p.step()
                self.net.zero_grad()
                Loss_p += loss_p
                Loss_q += loss_q
            self.Qs = []
        print('lossp', Loss_p, 'lossq', Loss_q, 'q', q_predicts.mean())
        # gradient descent for several replays 
        Loss = 0

        
    # save the data in batch and pass to network
    def experiment(self, epochs = 10, epsilon=0, reward_control=None, train_hidden = True,
                   decode=False, size_range=[10], test=False, lr_rate = 1e-5, implicit = False):
        # initialize, might take data during test
        self.trace = []
        self.Qs = []
        self.Hiddens = []
        self.Pos = []
        self.episode(epochs=epochs, epsilon=epsilon, reward_control=reward_control, size_range=size_range,
                     train_hidden=train_hidden, test=test,
                     decode=decode, implicit=implicit)

        self.train_sgd(lr_rate = lr_rate)
        self.Targets_batch = []
        self.Pos_batch = []
        process = psutil.Process(os.getpid())
        print('clear session data', process.memory_info().rss)





