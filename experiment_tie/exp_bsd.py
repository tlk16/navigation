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


class Rat():
    """
    agent reset, (act, remember), train
    """

    def __init__(self, memory_size=1000, input_type='touch', train_paras='two', device='cuda:1'):
        if input_type == 'touch':
            self.net = RNN(input_size=4, action_size=2, hidden_size=512, output_size=8).to(device)
        elif input_type == 'pos':
            self.net = RNN(input_size=2, action_size=2, hidden_size=512, output_size=8).to(device)
        else:
            raise TypeError('input tpye wrong')

        self.action_space = 8

        # parameters useless in the future
        self.input_type = input_type
        self.train_paras = train_paras

        self.device = device

        self.memory = []
        self.memory_size = memory_size

        # Q-learning parameters
        self.discount = 0.99
        self.lam = 0.3
        self.alpha = 0.2

        self.batch_size = int(self.memory_size/10)

        # changed
        self.epsilon = 0.5

        self.hidden_state = self.net.initHidden().to(device)
        self.last_action = self._initAction()
        self.sequence = None
        self.phase = 'train'
        self.losses = []

    def _epsilon_choose_action(self, q_value):
        """
        epsilon-greedy choose action according to q_value
        :param q_value: tensor, size=[1,8]
        :param epsilon: the degree of greed, float 0~1
        :return: acion, int
        """
        if random.random() > self.epsilon:
            return self._greedy_choose_action(q_value)

        return random.randint(0, self.action_space - 1)

    def _greedy_choose_action(self, q_value):
        return torch.argmax(q_value).item()

    def int2angle(self, action):
        """

        :param action: action, int
        :return: action vector, np.array
        """

        angle = action * 2 * math.pi / self.action_space
        # return np.array([np.sign(math.cos(angle)), np.sign(math.sin(angle))])
        return np.array([math.cos(angle), math.sin(angle)])

    def _initAction(self):
        return random.randint(0, self.action_space - 1)

    def act(self, state):
        """
        give action(int) accoring to state
        :param state: state[2]: array or list shape=[4,]
        :return:
        """
        if self.input_type == 'touch':
            input = state[2]
        elif self.input_type == 'pos':
            input = state[0]
        else:
            raise TypeError('input tpye wrong')

        with torch.no_grad():
            touch = torch.from_numpy(input).float().to(self.device)  # touch.shape = [4,]
            output, self.hidden_state = self.net(touch, self.hidden_state,
                                                 torch.from_numpy(self.int2angle(self.last_action)).float().to(self.device))
            # output.shape = [1,8] self.hidden_state.shape = [1,512]

        if self.phase == 'train':
            action = self._epsilon_choose_action(output)
        elif self.phase == 'test':
            action = self._greedy_choose_action(output)
        else:
            raise TypeError('rat.phase wrong')

        self.last_action = action
        return action

    def reset(self, init_net, init_sequence, phase):
        """
        reset hidden_state, phase, last_action and sequence
        """

        if init_net:
            self.hidden_state = self.net.initHidden().to(self.device)

        self.last_action = self._initAction()
        self.phase = phase
        if init_sequence:
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
        """

        :param state: s_0 ~s_t
        :param reward: s_0 ~s_t
        :param action: a_-1 ~ a_t-1
        :param done:
        :return:
        """
        self.sequence['positions'].append(torch.from_numpy(state[0]))
        # self.sequence['observations'].append(state[1])
        self.sequence['touches'].append(torch.from_numpy(state[2]))
        self.sequence['rewards'].append(reward)
        self.sequence['actions'].append(action)

        if done:
            self.sequence['touches'] = torch.stack(self.sequence['touches']).to(self.device)
            self.sequence['positions'] = torch.stack(self.sequence['positions']).to(self.device)
            self.sequence['actions'] = torch.from_numpy(np.array(self.sequence['actions'])).to(self.device)
            self.sequence['action_angles'] = torch.from_numpy(np.array([self.int2angle(action) for action in self.sequence['actions']])).to(self.device)
            self.sequence['rewards'] = torch.FloatTensor(self.sequence['rewards']).to(self.device)
            # print(self.sequence)
            self.memory.append(self.sequence)
        if len(self.memory) > self.memory_size:
            del self.memory[0:int(self.memory_size/5)]

    def train(self, lr_rate=1e-6):
        if self.train_paras == 'two':
            Optimizer_q = torch.optim.Adam(
                [
                    {'params': self.net.h2o, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.bo, 'lr': lr_rate, 'weight_decay': 0},
                ]
            )
        elif self.train_paras == 'all':
            Optimizer_q = torch.optim.Adam(
                [
                    {'params': self.net.i2h, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.a2h, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2h, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.bh, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2o, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.bo, 'lr': lr_rate, 'weight_decay': 0},
                    {'params': self.net.r, 'lr': lr_rate, 'weight_decay': 0},
                ]
            )
        else:
            raise TypeError('train paras wrong')

        if self.batch_size <= len(self.memory):
            train_memory = random.sample(self.memory, self.batch_size)
        else:
            train_memory = self.memory
        loss_all = []
        Optimizer_q.zero_grad()
        if self.input_type == 'touch':
            inputs = torch.stack([sequence['touches'] for sequence in train_memory])
        elif self.input_type == 'pos':
            inputs = torch.stack([sequence['positions'] for sequence in train_memory])
        else:
            raise TypeError('input type wrong')

        q_predicts = self.net.forward_sequence_values(
            inputs.float(),
            torch.stack([sequence['hidden0'] for sequence in train_memory]).float(),
            torch.stack([sequence['action_angles'] for sequence in train_memory]).float()
        )

        actions = torch.stack([sequence['actions'] for sequence in train_memory])
        actions = actions.unsqueeze(2).long()

        rewards = torch.stack([sequence['rewards'] for sequence in train_memory])
        rewards = rewards.unsqueeze(2)

        q_predicts = torch.stack(q_predicts).permute((1, 0, 2))

        Qs = self.value_back(q_predicts, actions, rewards)

        loss_q = torch.mean((q_predicts[:, :-1, :] - Qs.detach()) ** 2)
        loss_q.backward()
        loss_all.append(loss_q.detach().item())
        Optimizer_q.step()
        self.losses.append(np.array(loss_all).mean())

    def value_back(self, predicts, actions, rewards):
        """

        :param predicts: tensor [50, 100, 8]
        :param actions: tensor [50, 100, 1]
        :param rewards: tensor [50, 100, 1]
        :return:
        """
        print('value_back', predicts.shape, actions.shape, rewards.shape)
        q = torch.unsqueeze(torch.max(predicts[:, 1:, :], 2)[0], 2)  # [5,9,1]  # q_max
        q = q * self.discount + rewards[:, 1:, :]

        g_last = q
        g_sum = q.clone()
        # print(g_sum)
        for i in range(2, 10):
            g_now = rewards[:, :(-i), :] + self.discount * g_last[:, 1:, :]
            g_sum[:, :(-i + 1), :] += (self.lam ** (i - 1)) * g_now
            g_last = g_now

        # print(g_sum)
        g_re = predicts[:, :-1, :].clone()  # 5,9,4
        g_re.scatter_(2, actions[:, :-1, :], g_sum * (1 - self.lam))
        return g_re



class Session:
    def __init__(self, rat, env):
        self.rat = rat
        self.env = env
        self.phase = 'train'
        self.rewards = {'train':[], 'test':[]}
        self.mean_rewards = {'train':[], 'test':[]}

    def episode(self, epochs=10):

        for epoch in range(epochs):
            sum_step = 0
            sr = 0
            while sum_step < self.env.limit:
                self.rat.reset(init_net=(sum_step==0), init_sequence=(sum_step==0), phase=self.phase)
                state, reward, done, _ = self.env.reset()
                sr += reward
                sum_step += 1
                if self.phase == 'train':
                    self.rat.remember(state, reward, self.rat.last_action, sum_step == self.env.limit)
                if sum_step == self.env.limit:
                    break

                while not done:
                    action = self.rat.act(state)
                    state, reward, done, _ = self.env.step(self.rat.int2angle(action))
                    sr += reward
                    sum_step += 1

                    if self.phase == 'train':
                        self.rat.remember(state, reward, action, sum_step == self.env.limit)
                    if sum_step == self.env.limit:
                        break

            self.rewards[self.phase].append(sr)

        self.mean_rewards[self.phase].append(np.array(self.rewards[self.phase]).mean())
        self.rewards = {'train': [], 'test': []}

    def experiment(self, epochs=10):
        # initialize, might take data during test
        if self.phase == 'train':
            self.episode(epochs)
            self.rat.train()
        elif self.phase == 'test':
            self.episode(epochs)
        else:
            raise TypeError('session.phase wrong')

    def save_png(self, filename):
        def smooth(list_a, n=3):
            weights = np.ones(n) / n
            return np.convolve(weights, list_a)[0:-n + 1]
        plt.figure()
        line1, = plt.plot(self.rat.losses, label='loss')
        line2, = plt.plot(self.mean_rewards['train'], label='train')
        line3, = plt.plot(self.mean_rewards['test'], label='test')
        cum_train = smooth(self.mean_rewards['train'], 10)
        cum_test = smooth(self.mean_rewards['test'], 10)
        line4, = plt.plot(cum_train, label='train_cum')
        line5, = plt.plot(cum_test, label='test_cum')
        plt.legend(handles=[line1, line2, line3, line4, line5])
        plt.savefig(filename)
        plt.ion()
        plt.pause(5)
        plt.close()




if __name__ == '__main__':
    # parameters
    # q-learning lamda discount alpha
    # memory size
    # lr_rate
    # epsilon
    # trained parameters
    # env limit dim goal

    pass


    def worker(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005):
        """

        :param input_type:
        :param epsilon:
        :param train_paras:
        :return:
        """
        n_train = 500
        rat = Rat(memory_size=200, input_type=input_type, train_paras=train_paras)
        env = RatEnv(dim=[15, 15, 100], speed=1., collect=False, goal=[10, 10, 1],
                     limit=100, wall_offset=1., touch_offset=2., wall_reward=wall_reward, step_reward=step_reward)
        session = Session(rat, env)
        for i in range(n_train):
            print(i)
            rat.epsilon = epsilon[0] - i * epsilon[1] \
                if epsilon[0] - i * epsilon[1] > epsilon[2] else epsilon[2]

            session.phase = 'train'
            session.experiment(epochs=5)

            session.phase = 'test'
            session.experiment(epochs=10)

            if (i + 1) % 20 == 0:
                session.save_png(input_type + '[' + str(epsilon[0]) + ' ' +
                                 str(epsilon[1]) + ' ' + str(epsilon[2]) + ']' +
                                 train_paras + ' ' + str(wall_reward) + str(step_reward) + str(i) + '.png')


    worker()
    # try to train in int-grid situation
    # memory a_t-1, s_t, r_t,

    # memory hundreds~sampling
    # gpu batch
    # just train some weights
    # grid int action

