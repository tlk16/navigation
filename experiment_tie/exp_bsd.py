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

    def __init__(self, memory_size=50, input_type='touch', train_paras='two', device='cuda:0'):
        if input_type == 'touch':
            self.net = RNN(input_size=4, action_size=2, hidden_size=512, output_size=8).to(device)
        if input_type == 'pos':
            self.net = RNN(input_size=2, action_size=2, hidden_size=512, output_size=8).to(device)
        self.input_type = input_type
        self.train_paras = train_paras

        self.device = device

        self.memory = []
        self.memory_size = memory_size

        self.discount = 0.99
        self.lam = 0.3
        self.alpha = 0.2

        self.epsilon = 0.5


        self.num_step = 0
        self.hidden_state = self.net.initHidden().to(device)
        self.action0 = self._initAction()
        self.sequence = None
        self.phase = 'train'
        self.losses = []

    def _epsilon_choose_action(self, q_value, epsilon=0.2):
        """
        epsilon-greedy choose action according to q_value
        :param q_value: tensor, size=[1,8]
        :param epsilon: the degree of greed, float 0~1
        :return: acion, int
        """
        if random.random() > epsilon:
            return self._greedy_choose_action(q_value)

        return random.randint(0, 7)

    def _greedy_choose_action(self, q_value):
        return torch.argmax(q_value).item()

    def int2angle(self, action):
        """

        :param action: action, int
        :return: action vector, np.array
        """

        angle = action * math.pi / 4
        # return np.array([np.sign(math.cos(angle)), np.sign(math.sin(angle))])
        return np.array([math.cos(angle), math.sin(angle)])

    def _initAction(self):
        return random.randint(0, 7)

    def act(self, state):
        """
        give action(int) accoring to state
        :param state: state[2]: array or list shape=[4,]
        :return:
        """
        if self.input_type == 'touch':
            with torch.no_grad():
                touch = state[2]
                touch = torch.from_numpy(touch).float().to(self.device)
                # touch.shape = [4,]
                last_action = torch.from_numpy(self.int2angle(self.sequence['actions'][-1])).float().to(self.device)
                output, self.hidden_state = self.net(touch, self.hidden_state, last_action)
                # output.shape = [1,8] self.hidden_state.shape = [1,512]
                output = output.to('cpu')
            self.num_step += 1
            if self.phase == 'train':
                return self._epsilon_choose_action(output, epsilon=self.epsilon)
            return self._greedy_choose_action(output)
        elif self.input_type == 'pos':
            with torch.no_grad():
                pos = state[0]
                pos = torch.from_numpy(pos).float().to(self.device)
                # touch.shape = [4,]
                last_action = torch.from_numpy(self.int2angle(self.sequence['actions'][-1])).float().to(self.device)
                output, self.hidden_state = self.net(pos, self.hidden_state, last_action)
                output = output.to('cpu')
                # output.shape = [1,8] self.hidden_state.shape = [1,512]
            self.num_step += 1
            if self.phase == 'train':
                return self._epsilon_choose_action(output, epsilon=self.epsilon)
            return self._greedy_choose_action(output)
        else:
            print('input_size wrong')

    def reset(self, init_net=True, phase='train'):
        """
        reset hidden_state, phase, nump_step and sequence
        """

        if init_net:
            self.hidden_state = self.net.initHidden().to(self.device)

        self.num_step = 0
        self.phase = phase
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
            del self.memory[0]

    def train(self, lr_rate=1e-6):
        if self.train_paras == 'two':
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
            print('train_paras wrong')

        # Optimizer_q.zero_grad()
        loss_all = []
        Optimizer_q.zero_grad()
        if self.input_type == 'touch':
            inputs = torch.stack([sequence['touches'] for sequence in self.memory])
        elif self.input_type == 'pos':
            inputs = torch.stack([sequence['positions'] for sequence in self.memory])
        else:
            print('input_size wrong')

        q_predicts = self.net.forward_sequence_values(
            inputs.float(),
            torch.stack([sequence['hidden0'] for sequence in self.memory]).float(),
            torch.stack([sequence['action_angles'] for sequence in self.memory]).float()
        )
        actions = torch.stack([sequence['actions'] for sequence in self.memory]).float()
        # print(actions)
        q_predicts = torch.stack(q_predicts).permute((1, 0, 2))
        Qs = self.value_back(q_predicts, actions.unsqueeze(2).long(),
                             torch.stack([sequence['rewards'] for sequence in self.memory]).unsqueeze(2)
                             )
        # print(q_predicts.shape, Qs.shape)
        if len(Qs) >= 2:
            loss_q = torch.mean((q_predicts[:, :-1, :] - Qs.detach()) ** 2)
            loss_q.backward()
            # print('loss', loss_q * 1000)
            loss_all.append(10 * loss_q.detach().item())
        # for para in self.net.parameters():
        #     print(para.grad)
        Optimizer_q.step()
        self.losses.append(np.array(loss_all).mean())

    def value_backward(self, Predicts, Actions, Rewards):
        """
        Predicts: list, length=[num_step], each item is tensor, size=[1, 8]
        Actions: Tensor, size=[num_step]
        Rewards: list length=[num_step]
        """

        # print(Predicts[0].shape, Actions, Rewards)

        trace = []
        Qs = []
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
        return Qs

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

    def reset_body(self):
        self.env.reset()             # pos or random
        self.rat.reset()

    def episode(self, epochs=10):

        for epoch in range(epochs):
            self.rat.reset(self.phase)
            sum_step = 0
            sr = 0
            while sum_step < self.env.limit:
                state, reward, done, _ = self.env.reset()
                sum_step += 1
                if self.phase == 'train':
                    self.rat.remember(state, reward, self.rat.action0, sum_step == self.env.limit)
                else:
                    self.rat.remember(state, reward, self.rat.action0, done=False)
                if sum_step == self.env.limit:
                    break


                while not done:
                    action = self.rat.act(state)
                    state, reward, done, _ = self.env.step(self.rat.int2angle(action))
                    # print(sum_step)
                    sum_step += 1

                    if self.phase == 'train':
                        self.rat.remember(state, reward, action, sum_step == self.env.limit)
                    if sum_step == self.env.limit:
                        break

                    # print(self.rat.num_step, self.rat.int2angle(action), state[0], state[2], step)
                # print(self.phase, epoch, 'running reward', reward)
                    sr += reward

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
            print('wrong phase')

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


    def smooth(list_a, n=3):
        weights = np.ones(n) / n
        return np.convolve(weights, list_a)[0:-n+1]
    a = np.array([1,2,3,4,5,6, 7, 10])
    print(smooth(a, 3))


    # try to train in int-grid situation
    # memory a_t-1, s_t, r_t,

    # memory hundreds~sampling
    # gpu batch
    # just train some weights
    #

