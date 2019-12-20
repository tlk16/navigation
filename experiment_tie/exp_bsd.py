import sys
import math
import numpy as np
import random

import torch.optim as optim
import torch
import torch.utils.data

from RNN import RNN
import bsd_env.rat_env as rat_env


class Rat():
    def __init__(self):
        self.net = RNN(input_size=4, action_size=2, hidden_size=512, output_size=8)
        self.memory = []

        self.num_step = 0
        self.hidden_state = self.net.initHidden()
        self.action0 = self._initAction()
        self.sequence = None

    def _epsilon_choose_action(self, q_value, epsilon=0.2):
        if random.random() > epsilon:
            return self._greedy_choose_action(q_value)

        return random.randint(0, 7)

    def _greedy_choose_action(self, q_value):
        return torch.argmax(q_value).item()

    def int2angle(self, action):
        angle = action * math.pi / 4
        return np.array([math.cos(angle), math.sin(angle)])

    def _initAction(self):
        return random.randint(0, 7)

    def act(self, state, epsilon):
        """
        state[2]: array or list shape=[4,]
        """
        with torch.no_grad():
            touch = state[2]
            touch = torch.from_numpy(touch).float()
            # touch.shape = [4,]
            last_action = torch.from_numpy(self.int2angle(self.sequence['actions'][-1])).float()
            output, self.hidden_state = self.net(touch, self.hidden_state, last_action)
            # output.shape = [1,8] self.hidden_state.shape = [1,512]
        self.num_step += 1
        return self._epsilon_choose_action(output, epsilon)

    def reset(self, init_net=True):
        """
        reset hidden_state, nump_step and sequence
        """

        if init_net:
            self.hidden_state = self.net.initHidden()

        self.num_step = 0
        self.sequence = {
            'hidden0': self.hidden_state,
            'positions': [],
            'observations': [],
            'touches': [],
            'rewards': [],
            'actions': []
        }

    def remember(self, state, reward, action, done):
        # a_(t-1), s_t
        self.sequence['positions'].append(state[0])
        # self.sequence['observations'].append(state[1])
        self.sequence['touches'].append(state[2])
        self.sequence['rewards'].append(reward)
        self.sequence['actions'].append(action)
        if done:
            self.memory.append(self.sequence)
        if len(self.memory) > 20:
            del self.memory[0]

    def train(self, lr_rate=1e-6):
        Optimizer_q = torch.optim.Adam(
            [
                # {'params': self.net.i2h, 'lr': lr_rate, 'weight_decay': 0},
                # {'params': self.net.a2h, 'lr': lr_rate, 'weight_decay': 0},
                # {'params': self.net.h2h, 'lr': lr_rate, 'weight_decay': 0},
                # {'params': self.net.bh, 'lr': lr_rate, 'weight_decay': 0},
                {'params': self.net.h2o, 'lr': lr_rate, 'weight_decay': 0},
                {'params': self.net.bo, 'lr': lr_rate, 'weight_decay': 0},
                # {'params': self.net.r, 'lr': lr_rate, 'weight_decay': 0},
            ]
        )

        # Optimizer_q.zero_grad()
        for sequence in self.memory:
            Optimizer_q.zero_grad()
            self.net.zero_grad()
            q_predicts = self.net.forward_sequence_values(
                torch.from_numpy(np.array(sequence['touches'])).float(),
                sequence['hidden0'],
                torch.from_numpy(np.array([self.int2angle(action) for action in sequence['actions']])).float()
            )
            actions = torch.from_numpy(np.array(sequence['actions'])).float()
            # print(actions)
            Qs = self.value_backward(q_predicts, actions, sequence['rewards'])

            if len(Qs) >= 2:
                loss_q = torch.mean((torch.stack(q_predicts[:-1]) - torch.stack(Qs).detach()) ** 2)
                loss_q.backward()
                print('loss', loss_q * 1000)
            Optimizer_q.step()

    def value_backward(self, Predicts, Actions, Rewards):
        """
        Predicts: list, length=[num_step], each item is tensor, size=[1, 8]
        Actions: Tensor, size=[num_step]
        Rewards: list length=[num_step]
        """

        # print(Predicts[0].shape, Actions, Rewards)

        trace = []
        Qs = []
        self.discount = 0.99
        self.lam = 0.9
        self.alpha = 0.2
        # print('q_pre', len(Predicts), Predicts)
        for Q_now, Q_next, action, reward \
                in zip(Predicts[:-1], Predicts[1:], Actions[1:], Rewards):
            action = int(action.item())
            targetQ = Q_now.clone().detach()  # [1, 8]
            Qmax = torch.max(Q_next)   # float

            delta = torch.FloatTensor([reward]) + self.discount * Qmax  - targetQ[0, action]
            trace = [e * self.discount * self.lam for e in trace]
            Qs.append(targetQ)
            trace.append(1)
            def f(e, delta, q):
                q[0, action] = q[0, action] + self.alpha * delta * e
                return q
            Qs = [f(e, delta, q) for e, q in zip(trace, Qs)]
        # print('QS', len(Qs), Qs)
        return Qs


class Session:
    def __init__(self):
        self.rat = Rat()
        self.env = rat_env.RatEnv(dim=[30, 30, 100], speed=1.,
                                  goal=[10, 10, 1], limit=200,
                                  wall_offset=1., touch_offset=2.)

    def reset_body(self):
        self.env.reset(random=False, pos=np.array([20, 20]))             # pos or random
        self.rat.reset()

    def episode(self, epochs=10, epsilon=0.):

        for _ in range(epochs):
            self.rat.reset()
            state, reward, done, info = self.env.reset(random=False, pos=np.array([15, 15]))
            self.rat.remember(state, reward, self.rat.action0, done)
            done = False

            while not done:
                action = self.rat.act(state, epsilon=epsilon)
                state, reward, done, info = self.env.step(self.rat.int2angle(action))
                self.rat.remember(state, reward, action, done)
                # print(self.rat.num_step, state[0], state[2], self.rat.int2angle(action))
            print('running reward', reward)

    def experiment(self, eposilon):
        # initialize, might take data during test
        self.episode(epochs=10, epsilon=eposilon)
        self.rat.train()



if __name__ == '__main__':
    n_train = 100
    session = Session()
    for i in range(n_train):
        session.experiment(eposilon=0.5)
        print('testing')
        session.episode(epsilon=0.)
