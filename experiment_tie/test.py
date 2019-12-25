import sys
import math
import numpy as np
import random

import torch.optim as optim
import torch
import torch.utils.data

from RNN import RNN
from environment import RatEnv
from matplotlib import pyplot as plt
from exp_bsd import Rat, Session


class Env:
    def __init__(self):
        self.limit = 10
        self.t = 0

    def reset(self):
        self.t += 1
        reward = -0.02
        done = False
        return [np.array([0, 0]), np.array([0]), np.array([1, 0, 0, 0])], reward, done, ''

    def step(self, action=None):
        self.t += 1
        if self.t % 5 == 0:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return [np.array([0, 0]), np.array([0]), np.array([1, 0, 0, 0])], reward, done, ''


def worker(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    n_train = 1
    rat = Rat(memory_size=200, input_type=input_type, train_paras=train_paras)
    env = Env()
    session = Session(rat, env)
    for i in range(n_train):
        print(i)
        rat.epsilon = epsilon[0] - i * epsilon[1] \
            if epsilon[0] - i * epsilon[1] > epsilon[2] else epsilon[2]

        session.phase = 'train'
        session.experiment(epochs=1)

        # session.phase = 'test'
        # session.experiment(epochs=10)



worker()