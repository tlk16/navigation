"""
class Rat and Session, core precess of RL
"""
import random
import math
import numpy as np

import torch.optim as optim
import torch
import torch.utils.data

from matplotlib import pyplot as plt

from RNN import RNN, Decoder

class Rat():
    """
    agent reset, (act, remember), train
    """

    def __init__(self, input_type, train_paras, device, train_stage,
                 action_space, env_limit, grid,
                 net_hidden_size, memory_size, batch_size, lr_rate,
                 pre_lr_rate, keep_p,
                 discount, lam):

        # parameters useless in the future
        self.input_type = input_type
        self.train_paras = train_paras
        self.train_stage = train_stage

        self.action_space = action_space
        self.env_limit = env_limit
        self.grid = grid

        self.device = device

        if input_type == 'touch':
            self.net = RNN(input_size=4, action_size=2,
                           hidden_size=net_hidden_size, output_size=self.action_space).to(device)
        elif input_type == 'pos':
            self.net = RNN(input_size=2, action_size=2,
                           hidden_size=net_hidden_size, output_size=self.action_space).to(device)
        else:
            raise TypeError('input tpye wrong')

        self.decoder = Decoder(hidden_size=net_hidden_size, output_size=self.grid ** 2).to(device)
        self.mem_decoder = Decoder(hidden_size=net_hidden_size, output_size=5).to(device)
        self.lr_rate = lr_rate
        self.pre_lr_rate = pre_lr_rate
        self.optimizer_q, self.optimizer_pos, self.optimizer_mem = self._init_optimizer()

        self.memory = []
        self.memory_size = memory_size

        self.keep_p = keep_p

        # Q-learning parameters
        self.discount = discount
        self.lam = lam

        self.batch_size = batch_size

        # changed
        self.epsilon = None

        self.hidden_state = self.net.initHidden().to(device)
        self.last_action = self._initAction()
        self.sequence = None
        self.phase = 'train'  # decide epsilon-greedy or not greedy, controlled by session
        self.pre_phase = 'train'  # decide whehter or train or test in pre_train
        self.losses = []
        self.accuracy = {'train': [], 'test': []}

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

    def _random_act(self, state):
        # print(np.mean(np.array(state[2]) - np.zeros(4)))
        if np.mean(np.array(state[2]) - np.zeros(4)) > 1e-3 or random.random() > self.keep_p:
            return random.randint(0, self.action_space - 1)
        return self.last_action

    def _rl_act(self, net_output):
        if self.phase == 'train':
            action = self._epsilon_choose_action(net_output)
        elif self.phase == 'test':
            action = self._greedy_choose_action(net_output)
        else:
            raise TypeError('rat.phase wrong')
        return action

    def _update_hidden(self, state):
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
            net_output, self.hidden_state = self.net(touch, self.hidden_state,
                                                     torch.from_numpy(self.int2angle(self.last_action)).float().to(self.device))
        return net_output

    def act(self, state):
        net_output = self._update_hidden(state)
        if self.train_stage == 'pre_train' \
                or self.train_stage == 'pre_train_mem' \
                or self.train_stage == 'pre_train_mem_dict':
            action = self._random_act(state)
        elif self.train_stage == 'q_learning':
            action = self._rl_act(net_output)
        else:
            raise TypeError('train_stage wrong')

        pos_predict = self.decoder(self.hidden_state.to(self.device))\
            .squeeze().detach().cpu().numpy()
        mem_predict = self.mem_decoder(self.hidden_state.to(self.device))\
            .squeeze().detach().cpu().numpy()
        self.last_action = action
        return action, pos_predict, mem_predict

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
                # 'observations': [],
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
            self.sequence['action_angles'] = torch.from_numpy(np.array([self.int2angle(action) for action in self.sequence['actions']])).to(self.device)
            self.sequence['actions'] = torch.from_numpy(np.array(self.sequence['actions'])).to(self.device)
            self.sequence['rewards'] = torch.FloatTensor(self.sequence['rewards']).to(self.device)
            # print(self.sequence)
            self.memory.append(self.sequence)

        if len(self.memory) > self.memory_size:
            del self.memory[0:int(self.memory_size/5)]

    def _init_optimizer(self):
        if self.train_paras == 'two':
            optimizer_q = torch.optim.Adam(
                [
                    {'params': self.net.h2o, 'lr': self.lr_rate, 'weight_decay': 0},
                    {'params': self.net.bo, 'lr': self.lr_rate, 'weight_decay': 0},
                ]
            )
            optimizer_pos = torch.optim.Adam(
                [
                    {'params': self.decoder.h2p, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.decoder.bp, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                ]
            )
            optimizer_mem = torch.optim.Adam(
                [
                    {'params': self.mem_decoder.h2p, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.mem_decoder.bp, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                ]
            )

        elif self.train_paras == 'all':
            optimizer_q = torch.optim.Adam(
                [
                    {'params': self.net.i2h, 'lr': self.lr_rate, 'weight_decay': 0},
                    {'params': self.net.a2h, 'lr': self.lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2h, 'lr': self.lr_rate, 'weight_decay': 0},
                    {'params': self.net.bh, 'lr': self.lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2o, 'lr': self.lr_rate, 'weight_decay': 0},
                    {'params': self.net.bo, 'lr': self.lr_rate, 'weight_decay': 0},
                    {'params': self.net.r, 'lr': self.lr_rate, 'weight_decay': 0},
                ]
            )

            optimizer_pos = torch.optim.Adam(
                [
                    {'params': self.net.i2h, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.a2h, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2h, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.bh, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2o, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.bo, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.r, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.decoder.h2p, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.decoder.bp, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                ]
            )

            optimizer_mem = torch.optim.Adam(
                [
                    {'params': self.net.i2h, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.a2h, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2h, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.bh, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.h2o, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.bo, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.net.r, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.mem_decoder.h2p, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                    {'params': self.mem_decoder.bp, 'lr': self.pre_lr_rate, 'weight_decay': 0},
                ]
            )

        else:
            raise TypeError('train paras wrong')

        return optimizer_q, optimizer_pos, optimizer_mem

    def train(self):

        if self.batch_size <= len(self.memory):
            train_memory = random.sample(self.memory, self.batch_size)
        else:
            train_memory = self.memory

        self.optimizer_q.zero_grad()
        self.optimizer_pos.zero_grad()

        if self.input_type == 'touch':
            inputs = torch.stack([sequence['touches'] for sequence in train_memory])
        elif self.input_type == 'pos':
            inputs = torch.stack([sequence['positions'] for sequence in train_memory])
        else:
            raise TypeError('input type wrong')

        q_predicts, hiddens = self.net.forward_sequence_values(
            inputs.float(),
            torch.stack([sequence['hidden0'] for sequence in train_memory]).float(),
            torch.stack([sequence['action_angles'] for sequence in train_memory]).float()
        )

        if self.train_stage == 'q_learning':

            actions = torch.stack([sequence['actions'] for sequence in train_memory])
            actions = actions.unsqueeze(2).long()

            rewards = torch.stack([sequence['rewards'] for sequence in train_memory])
            rewards = rewards.unsqueeze(2)

            # print(torch.stack(q_predicts).shape)
            q_predicts = torch.stack(q_predicts).permute((1, 0, 2))
            Qs = self.value_back(q_predicts, actions, rewards)

            loss_q = torch.mean((q_predicts[:, :-1, :] - Qs.detach()) ** 2)
            loss_q.backward()
            self.optimizer_q.step()
            self.losses.append(loss_q.detach().item())

        elif self.train_stage == 'pre_train':
            hiddens = torch.stack(hiddens).permute((1, 0, 2))
            pos_predicts = self.decoder.forward_sequence_values(hiddens)
            pos_predicts = torch.stack(pos_predicts).permute((1, 0, 2))

            pos = torch.stack([sequence['positions'] for sequence in train_memory]).float()
            pos = self.area(pos).long()
            loss_pos_layer = torch.nn.CrossEntropyLoss()
            # print(pos_predicts.reshape((-1, pos_predicts.shape[2])).shape)
            loss_pos = loss_pos_layer(pos_predicts.reshape((-1, pos_predicts.shape[2])), pos)
            if self.pre_phase == 'train':
                loss_pos.backward()
                self.optimizer_pos.step()
                self.losses.append(loss_pos.detach().item())

                accuracy = torch.eq(torch.argmax(pos_predicts.reshape(-1, pos_predicts.shape[2]), 1), pos)\
                               .sum().float().item() / torch.numel(pos)
                print('train accuracy', accuracy)
                self.accuracy['train'].append(accuracy)

            elif self.pre_phase == 'test':
                accuracy = torch.eq(torch.argmax(pos_predicts.reshape(-1, pos_predicts.shape[2]), 1), pos) \
                               .sum().float().item() / torch.numel(pos)
                print('test accuracy', accuracy)
                self.accuracy['test'].append(accuracy)
            else:
                raise TypeError('pre_phase wrong')

        elif self.train_stage == 'pre_train_mem' or self.train_stage == 'pre_train_mem_dict':
            hiddens = torch.stack(hiddens).permute((1, 0, 2))
            mem_predicts = self.mem_decoder.forward_sequence_values(hiddens)
            mem_predicts = torch.stack(mem_predicts).permute((1, 0, 2))

            mem = torch.stack([sequence['touches'] for sequence in train_memory]).float()
            if self.train_stage == 'pre_train_mem':
                mem = self.touched(mem)
            else:
                mem = self.will_touch(mem)

            loss_mem_layer = torch.nn.CrossEntropyLoss()
            loss_mem = loss_mem_layer(mem_predicts.reshape((-1, mem_predicts.shape[2])), mem)
            loss_mem.backward()
            self.optimizer_mem.step()
            self.losses.append(loss_mem.detach().item())

            accuracy = torch.eq(torch.argmax(mem_predicts.reshape(-1, mem_predicts.shape[2]), 1), mem)\
                           .sum().float().item() / torch.numel(mem)
            print('train accuracy', accuracy)
            self.accuracy['train'].append(accuracy)

        else:
            raise TypeError('train stage wrong')

    def value_back(self, predicts, actions, rewards):
        """

        :param predicts: tensor [50, 100, 8]
        :param actions: tensor [50, 100, 1]
        :param rewards: tensor [50, 100, 1]
        :return:
        """
        assert predicts.shape[0:2] == actions.shape[0:2]
        assert rewards.shape == actions.shape
        q = torch.unsqueeze(torch.max(predicts[:, 1:, :], 2)[0], 2)  # [5,9,1]  # q_max
        q = q * self.discount + rewards[:, 1:, :]

        g_last = q
        g_sum = q.clone()
        # print(g_sum)
        for i in range(2, actions.shape[1]):
            g_now = rewards[:, :(-i), :] + self.discount * g_last[:, 1:, :]
            g_sum[:, :(-i + 1), :] += (self.lam ** (i - 1)) * g_now
            g_last = g_now

        # print(g_sum)
        g_re = predicts[:, :-1, :].clone()  # 5,9,4
        g_re.scatter_(2, actions[:, :-1, :], g_sum * (1 - self.lam))
        return g_re

    def area(self, pos):
        """

        :param pos: tensor [batch_size, step_num, 2]
        :return:
        """
        pos = torch.floor(pos / self.env_limit * self.grid)
        pos = pos[:, :, 0] * self.grid + pos[:, :, 1]
        pos = pos.reshape(-1)
        return pos

    def touched(self, touch):
        """

        :param touch: tensor [batch_size, step_num, 4]
        :return: tensor [batch_size, step_num, 5]
        """
        touch = torch.cat((touch, torch.zeros(touch.shape[0], touch.shape[1], 1).to(self.device)), dim=2)
        for sequence in touch:
            for i in range(1, touch.shape[1]):
                if torch.max(sequence[i]).data < 1e-3:
                    sequence[i] = sequence[i - 1]
                if torch.max(sequence[i - 1]).data < 1e-3:
                    sequence[i - 1, 4] = 1
            if torch.max(sequence[-1]).data < 1e-3:
                sequence[-1, 4] = 1
        return torch.max(touch, dim=2)[1].reshape(-1)

    def will_touch(self, touch):
        """

        :param touch: tensor [batch_size, step_num, 4]
        :return: tensor [batch_size, step_num, 5]
        """
        touch = torch.cat((touch, torch.zeros(touch.shape[0], touch.shape[1], 1).to(self.device)), dim=2)
        for sequence in touch:
            for i in range(1, touch.shape[1]):
                sequence[i - 1] = sequence[i]
                if torch.max(sequence[i - 1]).data < 1e-3:
                    sequence[i - 1, 4] = 1
            if torch.max(sequence[-1]).data < 1e-3:
                sequence[-1, 4] = 1
        return torch.max(touch, dim=2)[1].reshape(-1)





class Session:
    def __init__(self, rat, env):
        self.rat = rat
        self.env = env
        self.phase = 'train'
        self.rewards = {'train':[], 'test':[]}
        self.mean_rewards = {'train':[], 'test':[]}
        self.pos_accuracy = {'train':[], 'test':[]}
        self.mem_accuracy = {'train': [], 'test': []}

    def episode(self, epochs):

        pos_acc = 0
        mem_acc = 0
        mem_pre_acc = 0
        predict_num = 0
        for epoch in range(epochs):
            sum_step = 0
            sr = 0
            touched = 4
            mem_predicted = 4

            while sum_step < self.env.limit:
                self.rat.reset(init_net=(sum_step == 0), init_sequence=(sum_step == 0), phase=self.phase)
                state, reward, done, _ = self.env.reset()
                sr += reward
                if np.linalg.norm(state[2]) > 1e-3:
                    touched = np.argmax(state[2])
                sum_step += 1
                if self.phase == 'train':
                    self.rat.remember(state, reward, self.rat.last_action, sum_step == self.env.limit)
                if sum_step == self.env.limit:
                    break

                while not done:
                    action, pos_predict, mem_predict = self.rat.act(state)

                    if sum_step > 20: # not predict the first 20 steps, because the information is not enough
                        predict_num += 1
                        if self.area(state[0], grid=self.rat.grid, env_limit=self.rat.env_limit) == np.argmax(pos_predict):
                            pos_acc += 1
                        # print(touched, np.argmax(mem_predict))
                        if touched == np.argmax(mem_predict):
                            mem_acc += 1
                        if (mem_predicted == np.argmax(state[2])) or \
                                (np.linalg.norm(state[2]) < 1e-3 and mem_predicted == 4):
                            mem_pre_acc += 1
                        # print(mem_predicted, np.argmax(state[2]))
                    mem_predicted = np.argmax(mem_predict)
                    state, reward, done, _ = self.env.step(self.rat.int2angle(action))
                    sr += reward
                    if np.linalg.norm(state[2]) > 1e-3:
                        touched = np.argmax(state[2])
                    sum_step += 1

                    if self.phase == 'train':
                        self.rat.remember(state, reward, action, sum_step == self.env.limit)
                    if sum_step == self.env.limit:
                        break

            self.rewards[self.phase].append(sr)

        self.mean_rewards[self.phase].append(np.array(self.rewards[self.phase]).mean())
        self.rewards = {'train': [], 'test': []}
        if self.phase == 'test':
            self.pos_accuracy['test'].append(pos_acc / predict_num)
        if self.phase == 'test':
            if self.rat.train_stage == 'pre_train_mem':
                self.mem_accuracy['test'].append(mem_acc / predict_num)
            elif self.rat.train_stage == 'pre_train_mem_dict':
                self.mem_accuracy['test'].append(mem_pre_acc / predict_num)
            else:
                raise TypeError('wrong train_stage')

    def experiment(self, epochs):
        # initialize, might take data during test
        if self.phase == 'train':
            self.episode(epochs)
            # print(self.rat.memory)
            self.rat.train()
        elif self.phase == 'test':
            self.episode(epochs)
        else:
            raise TypeError('session.phase wrong')

    def save_png(self, filename, phase):
        def smooth(list_a, n):
            weights = np.ones(n) / n
            return np.convolve(weights, list_a)[0:-n + 1]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()
        if phase == 'q_learning':
            line1, = ax2.plot(self.rat.losses, label='loss')
            line2, = ax.plot(self.mean_rewards['train'], label='train')
            line3, = ax.plot(self.mean_rewards['test'], label='test')
            cum_train = smooth(self.mean_rewards['train'], 10)
            cum_test = smooth(self.mean_rewards['test'], 10)
            line4, = ax.plot(cum_train, label='train_cum')
            line5, = ax.plot(cum_test, label='test_cum')
            plt.legend(handles=[line1, line2, line3, line4, line5])
        elif phase == 'pre_train':
            line1, = ax2.plot(self.rat.losses, label='loss', color='b')
            cum_test_pos = smooth(self.pos_accuracy['test'], 10)
            line6, = ax.plot(self.pos_accuracy['test'], label='test_pos', color='g')
            line7, = ax.plot(cum_test_pos, label='test_pos_cum', color='r')
            line8, = ax.plot(self.rat.accuracy['train'], label='train_acc', color='y')
            line9, = ax.plot(self.rat.accuracy['test'],label='test_acc', color='g')
            plt.legend(handles=[line1, line8, line6, line7, line9])
        elif phase == 'pre_train_mem' or 'pre_train_mem_dict':
            line1, = ax2.plot(self.rat.losses, label='loss', color='b')
            cum_test_mem = smooth(self.mem_accuracy['test'], 10)
            line6, = ax.plot(self.mem_accuracy['test'], label='test_mem', color='g')
            line7, = ax.plot(cum_test_mem, label='test_mem_cum', color='r')
            line8, = ax.plot(self.rat.accuracy['train'], label='train_acc', color='y')
            line9, = ax.plot(self.rat.accuracy['test'],label='test_acc', color='g')
            plt.legend(handles=[line1, line8, line6, line7, line9])
        else:
            raise TypeError('phase wrong')
        plt.savefig(filename)
        plt.ion()
        plt.pause(5)
        plt.close()

    def area(self, pos, grid, env_limit):
        """
        caculate the area of a single pos
        :return:
        """
        assert pos.shape == (2,)
        pos = np.floor(pos / env_limit * grid)
        return pos[0] * grid + pos[1]




if __name__ == '__main__':
    pass

    # try to train in int-grid situation

    # memory hundreds~sampling
    # gpu batch
    # just train some weights
    # grid int action

    # problem of value back
    # update environment, touch signal in the corner
    # old problems of the environment. not serious
    # test of pre_train, not necessary
    # if pre_train is not ok, try to divide train/test dataset
    # why are multipro processings similar?
