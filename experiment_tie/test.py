
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
        self.his.append([input_, hidden, action])
        hidden = torch.ones(1, self.hidden_size)
        output = torch.FloatTensor([1,0,0,0,0,0,0,0])
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


def test_worker(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005):
    """

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
    # print(session.rat.memory)
    assert abs(session.rat.memory[0]['rewards'][0].item() + 0.02) < 1e-3
    assert abs(session.rat.memory[0]['rewards'][5].item() + 0.02) < 1e-3
    assert abs(session.rat.memory[0]['rewards'][4].item() - 1) < 1e-3

    assert abs(session.rat.memory[0]['actions'][0].item()) > 1e-3
    assert abs(session.rat.memory[0]['actions'][5].item()) > 1e-3

    # print(session.rat.net.his)

def test_RNN():
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
    pass


def test_pos_predict(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005, train_stage='q_learning'):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    n_train = 500
    rat = Rat(memory_size=1000, input_type=input_type, train_paras=train_paras, device='cuda:1', train_stage='pre_train')
    env = RatEnv(dim=[15, 15, 100], speed=1., collect=False, goal=[10, 10, 1],
                 limit=100, wall_offset=1., touch_offset=2., wall_reward=wall_reward, step_reward=step_reward)
    session = Session(rat, env)
    for i in range(n_train):
        if i < 50:
            rat.epsilon = 1
        else:
            rat.epsilon = epsilon[0] - i * epsilon[1] \
                if epsilon[0] - i * epsilon[1] > epsilon[2] else epsilon[2]
        print(i, rat.epsilon)

        session.phase = 'train'
        session.experiment(epochs=5)

        session.phase = 'test'
        session.experiment(epochs=5)

        if (i + 1) % 10 == 0:
            session.save_png(input_type + '[' + str(epsilon[0]) + ' ' +
                             str(epsilon[1]) + ' ' + str(epsilon[2]) + ']' +
                             train_paras + ' ' + str(wall_reward) + str(step_reward) + str(i) + 'pos.png')


test_worker()
test_pos_predict(input_type='touch', epsilon=(1, 0.002, 0.1), train_paras='all', wall_reward=0, step_reward=-0.005)


