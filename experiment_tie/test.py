import math
import numpy as np
from itertools import count
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import torch
import torch.utils.data

from environment import RatEnv
from RNN import RNN
from exp_bsd import Rat, Session


class FakeNN(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, output_size, inertia=0.5, k_action=1, max_size=20):
        super(FakeNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Parameter(torch.randn(input_size, hidden_size) * 10 * np.sqrt(2.0 / (input_size + hidden_size)))
        self.h2h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 1.0 * np.sqrt(2.0 / hidden_size))
        self.h2o = nn.Parameter(torch.randn(hidden_size, output_size) * 1 * np.sqrt(2.0 / (hidden_size + output_size)))
        self.a2h = nn.Parameter(torch.randn(action_size, hidden_size) * 1 * np.sqrt(2.0 / (hidden_size + 4)))
        self.bh = nn.Parameter(torch.zeros(1, hidden_size))
        self.bo = nn.Parameter(torch.zeros(1, output_size))
        self.r = nn.Parameter(inertia * torch.ones(1, hidden_size))

        self.his = []

    def forward(self, input_, hidden, action):
        # dim should be same except catting dimension
        # print(input_.shape, hidden.shape, action.shape)
        self.his.append({'input_':input_, 'hidden': hidden, 'action': action})
        hidden = torch.ones(1, self.hidden_size).to(input_.device)
        output = torch.FloatTensor([1,0,0,0,0,0,0,0]).to(input_.device)
        return output, hidden

    def forward_sequence_values(self, inputs, hidden0, actions):
        print(inputs.shape, hidden0.shape, actions.shape)
        print(inputs[:, 0].shape, actions[:, 0].shape)
        squence_length = inputs.shape[1]
        outputs = []
        hidden = torch.squeeze(hidden0)
        for i in range(squence_length):
            # print(inputs[:, i].shape, hidden.shape, actions[:, i].shape)
            output, hidden = self.forward(inputs[:, i], hidden, actions[:, i])
            outputs.append(output)
        return outputs

    def initHidden(self, batchsize=1):
        return torch.zeros(batchsize, self.hidden_size)

class Env:
    def __init__(self):
        self.limit = 10
        self.t = 0

    def reset(self):
        self.t += 1
        reward = -0.02
        done = False
        return [np.array([0, 0]), np.array([0]), np.array([self.t, 0, 0, 0])], reward, done, ''

    def step(self, action=None):
        self.t += 1
        if self.t % 5 == 0:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return [np.array([0, 0]), np.array([0]), np.array([self.t, 0, 0, 0])], reward, done, ''

class FakeRat:
    def __init__(self, goal, limit):
        pass
        self.goal = np.array(goal)
        self.limit = limit
        self.last_action = None
        self.losses = []


    def reset(self, init_net, init_sequence, phase):
        pass

    def remember(self, *args):
        pass

    def train(self, *args):
        self.losses.append(1)

    def int2angle(self, action):
        self.action_space = 8
        angle = action * 2 * math.pi / self.action_space
        # return np.array([np.sign(math.cos(angle)), np.sign(math.sin(angle))])
        return np.array([math.cos(angle), math.sin(angle)])

    def act(self, state):
        direction = np.sign(self.goal - state[0])
        if direction[0] >= 0 and direction[1] >= 0:
            action = 1
        if direction[0] <= 0 and direction[1] >= 0:
            action = 3
        if direction[0] <= 0 and direction[1] <= 0:
            action = 5
        if direction[0] >= 0 and direction[1] <= 0:
            action = 7
        pos_prediction = int(self.area(state[0], grid=3, env_limit=self.limit))
        a = np.zeros((3*3))
        a[pos_prediction] = 1
        return action, pos_prediction

    def area(self, pos, grid, env_limit):
        """
        caculate the area of a single pos
        :return:
        """
        assert pos.shape == (2,)
        pos = np.floor(pos / env_limit * grid)
        return pos[0] * grid + pos[1]


def test_worker(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005):
    """
    test rat.remember, _rl_act, _epsilon, _greedy, _init_Action
    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    n_train = 1
    rat = Rat(memory_size=200, input_type=input_type, train_paras=train_paras)
    rat.net = FakeNN(input_size=4, action_size=2, hidden_size=512, output_size=8).to(rat.device)
    env = Env()
    session = Session(rat, env)

    rat.epsilon = 0
    session.phase = 'train'
    session.episode(epochs=2)
    # print(rat.net.his)
    # test remember
    assert abs(session.rat.memory[0]['rewards'][0].item() + 0.02) < 1e-3
    assert abs(session.rat.memory[0]['rewards'][5].item() + 0.02) < 1e-3
    assert abs(session.rat.memory[0]['rewards'][4].item() - 1) < 1e-3
    # test _initAction
    assert abs(session.rat.memory[0]['actions'][0].item()) > 1e-3
    assert abs(session.rat.memory[0]['actions'][5].item()) > 1e-3
    # test _rl_act, _epsilon
    assert abs(session.rat.memory[0]['actions'][3].item()) < 1e-3
    assert abs(session.rat.memory[0]['actions'][6].item()) < 1e-3
    # test int2angle
    assert torch.dist(session.rat.memory[0]['action_angles'][4], torch.DoubleTensor([1,0]).to(rat.device)).item() < 1e-3
    assert np.linalg.norm(rat.int2angle(1) - np.array([0.707, 0.707])) < 1e-3
    # test net input self.last_action, self.hidden
    assert torch.dist(rat.net.his[0]['input_'], torch.FloatTensor([1, 0, 0, 0]).to(rat.device)).item() < 1e-3
    assert torch.dist(rat.net.his[5]['input_'], torch.FloatTensor([7, 0, 0, 0]).to(rat.device)).item() < 1e-3
    assert torch.dist(rat.net.his[0]['hidden'], torch.zeros(1, 512).to(rat.device)).item() < 1e-3
    assert torch.dist(rat.net.his[4]['hidden'], torch.ones(1, 512).to(rat.device)).item() < 1e-3
    assert torch.dist(rat.net.his[3]['action'], torch.FloatTensor([1, 0]).to(rat.device)).item() < 1e-3
    assert torch.dist(rat.net.his[0]['action'], torch.FloatTensor([1, 0]).to(rat.device)).item() > 1e-3


def test_RNN():
    pass

def test_Decoder():
    pass

def test_train(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    n_train = 1
    rat = Rat(memory_size=200, input_type=input_type, train_paras=train_paras)
    rat.net = RNN(input_size=4, action_size=2, hidden_size=512, output_size=8).to(rat.device)
    env = Env()
    session = Session(rat, env)

    rat.epsilon = 0
    session.phase = 'train'
    session.rat.memory = [{
        'hidden0': session.rat.net.initHidden().to(rat.device),
        'touches': torch.zeros(10, 4).to(rat.device),
        'rewards': torch.FloatTensor([0,0,0,0,0,0,0,0,0,1]).to(rat.device),
        'actions': torch.FloatTensor([0,0,0,0,0,0,0,0,0,1]).to(rat.device),
        'action_angles': torch.zeros(10, 2).to(rat.device),
    }]
    print(session.rat.memory)
    for i in range(10000):
        session.rat.train(lr_rate=1e-9)
    plt.plot(session.rat.losses)
    plt.show()

def test_value_back():
    """
    :param predicts: tensor [50, 100, 8]
        :param actions: tensor [50, 100, 1]
        :param rewards: tensor [50, 100, 1]
    :return:
    """
    predicts = torch.zeros((1,10,8))
    actions = torch.zeros((1, 10, 1)).long()
    rewards = torch.FloatTensor([[[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]]])
    rat = Rat(memory_size=200,input_type='touch', train_paras='all')
    rat.discount = 0.99
    rat.lam = 0.3
    print(rat.value_back(predicts, actions, rewards))

def test_tool():
    rat = FakeRat(goal=[10, 10], limit=100)
    env = RatEnv(dim=[15, 15, 100], speed=1., collect=False, goal=[10, 10, 1],
                 limit=100, wall_offset=1., touch_offset=2., wall_reward=0, step_reward=0)
    session = Session(rat, env)
    for i in range(100):

        print(i)

        session.phase = 'train'
        session.experiment(epochs=5)

        session.phase = 'test'
        session.experiment(epochs=1)

        if (i + 1) % 2 == 0:
            session.save_png(str(i) + 'test_tool.png', phase='pre_train')



def test_area():
    rat = Rat(memory_size=1000, device='cuda:1', train_stage='pre_train')
    rat.grid = 2
    rat.env_limit = 10
    a = torch.FloatTensor([[[3,3], [3,9], [9,2], [9, 9]]]).to(rat.device)
    assert torch.dist(rat.area(a), torch.FloatTensor([0,1,2,3]).to(rat.device)).item() < 1e-3

    env = RatEnv(dim=[15, 15, 100], speed=1., collect=False, goal=[10, 10, 1],
                 limit=100, wall_offset=1., touch_offset=2., wall_reward=0, step_reward=0)
    session = Session(rat, env)
    assert  session.area(np.array([3,4]), grid=2, env_limit=20) < 1e-3
    assert session.area(np.array([3, 9]), grid=2, env_limit=20) - 1 < 1e-3
    assert session.area(np.array([9, 2]), grid=2, env_limit=20) - 2 < 1e-3
    assert session.area(np.array([9, 9]), grid=2, env_limit=20) - 3 < 1e-3

def test_pos_predict(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005, train_stage='q_learning'):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    n_train = 500
    rat = Rat(memory_size=200, input_type=input_type, train_paras=train_paras, device='cuda:1', train_stage='pre_train')
    env = RatEnv(dim=[15, 15, 100], speed=1., collect=False, goal=[10, 10, 1],
                 limit=100, wall_offset=1., touch_offset=2., wall_reward=wall_reward, step_reward=step_reward)
    session = Session(rat, env)
    for i in range(n_train):

        print(i)

        session.phase = 'train'
        session.experiment(epochs=5)

        session.phase = 'test'
        session.experiment(epochs=1)

        if (i + 1) % 10 == 0:
            session.save_png(str(i) + 'pos.png', phase='pre_train')


# standard tests
# test_worker()
test_area()
# test_tool()
# new test

test_pos_predict(input_type='touch', epsilon=(1, 0.002, 0.1), train_paras='all', wall_reward=0, step_reward=-0.005)


