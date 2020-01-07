"""
multi-processing for testing super-parameters
"""
import os
import time
import random
import pickle
import multiprocessing
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from environment import RatEnv
from exp_bsd import Rat, Session

typical_args = {    # typical args of q_learning
    'rat_args':{
        'input_type': 'touch',
        'device': 'cuda:1',
        'train_stage': 'q_learning',

        # knowledge about environment
        'action_space': 8,
        'env_limit': 15,
        'grid': 3,

        # settings used by pre_train and q_learning
        'net_hidden_size': 512,
        'memory_size': 1000,
        'batch_size': 100,    # typically memory_size/10

        # pre_train
        'pre_lr_rate': 1e-5,
        'keep_p': 0.8,

        # q_learning
        'train_paras': 'all',
        'lr_rate': 1e-5,
        'discount': 0.99,
        'lam': 0.3,
    },

    'env_args':{
        'wall_reward': 0.0,
        'step_reward': -0.005,
        'dim': [15, 15, 100],
        'speed': 1.,
        'collect': False,
        'goal': [10, 10, 1],
        'limit': 100,
        'wall_offset': 1., # > 1
        'touch_offset': 2., # > 1
    },

    'epsilon': [1, 0.002, 0.1],
    'start': 50,
    'train_epochs': 50,
    'test_epochs': 10,
    'n_train': 4000,
    'show_time': 100
}

args = {
    'rat_args':{
        'input_type': 'touch',
        'device': 'cuda:1',
        'train_stage': 'pre_train',

        # knowledge about environment
        'action_space': 8,
        'env_limit': 15,     # must be the same as env_args dim
        'grid': 3,

        # settings used by pre_train and q_learning
        'net_hidden_size': 512,
        'memory_size': 200,
        'batch_size': 50,    # typically memory_size/10
        'train_paras': 'all',   # when pre_train. two means just train decoder; when q_learning, two means just train output layer

        # pre_train
        'pre_lr_rate': 1e-5,
        'keep_p': 0.8,

        # q_learning
        'lr_rate': 1e-5,
        'discount': 0.99,
        'lam': 0.3,
    },

    'env_args':{
        'wall_reward': 0.0,
        'step_reward': -0.005,
        'dim': [15, 15, 100],
        'speed': 1.,
        'collect': False,
        'goal': [10, 10, 1],
        'limit': 100,
        'wall_offset': 1., # > 1
        'touch_offset': 2., # > 1
    },

    'epsilon': [1, 0.002, 0.1],
    'start': 50,
    'train_epochs': 20,  # if train_epochs larger than rat_args batch_size, experience will be somehow wasted
    'test_epochs': 5,
    'n_train': 2000,
    'show_time': 100, # save fig per _ step

}


def worker(args, png_name):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    try:
        rat = Rat(**args['rat_args'])
        env = RatEnv(**args['env_args'])

        n_train = args['n_train']
        epsilon = args['epsilon']
        train_epochs = args['train_epochs']
        test_epochs = args['test_epochs']
        start = args['start']
        show_time = args['show_time']

        session = Session(rat, env)
        for i in range(n_train):
            if i < start:
                rat.epsilon = 1
            else:
                rat.epsilon = epsilon[0] - i * epsilon[1] \
                    if epsilon[0] - i * epsilon[1] > epsilon[2] else epsilon[2]
            print(i, rat.epsilon)

            session.phase = 'train'
            session.experiment(epochs=train_epochs)

            session.phase = 'test'
            session.experiment(epochs=test_epochs)

            if (i + 1) % show_time == 0:
                session.save_png(png_name + '.png', phase=args['rat_args']['train_stage'])
    except Exception as e:
        print(e)
        print('over', os.getpid())

def pre_worker(args, png_name):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    # batch_size, memory_size > 100000, pre_lr_rate
    try:
        rat = Rat(**args['rat_args'])
        env = RatEnv(**args['env_args'])

        session = Session(rat, env)

        files = []
        for filename in os.listdir(os.getcwd()):
            if filename.startswith('mem1024ory'):
                files.append(filename)
        random.shuffle(files)

        train_files = files[0: int(len(files) * 0.8)]
        test_files = files[int(len(files) * 0.8): ]

        for i in range(30000):
            print(i)

            # train
            file = random.choice(train_files)
            with open(file, 'rb') as f:
                memory = pickle.load(f)
            for sequence in memory:
                for k in sequence:
                    sequence[k] = sequence[k].to(args['rat_args']['device'])
            session.rat.memory = memory

            session.phase = 'train'
            session.rat.pre_phase = 'train'
            session.rat.train()

            # test
            file = random.choice(test_files)
            with open(file, 'rb') as f:
                memory = pickle.load(f)
            for sequence in memory:
                for k in sequence:
                    sequence[k] = sequence[k].to(args['rat_args']['device'])
            session.rat.memory = memory

            session.phase = 'test'
            session.episode(1)

            session.rat.pre_phase = 'test'
            session.rat.train()


            if i % 10 == 0:
                session.save_png(png_name + '.png', args['rat_args']['train_stage'])

    except Exception as e:
        print(e)

def get_data():
    args['rat_args']['memory_size'] = 10000000  # modify global
    args['rat_args']['net_hidden_size'] = 1024  # modify global

    rat = Rat(**args['rat_args'])
    env = RatEnv(**args['env_args'])

    session = Session(rat, env)

    session.phase = 'train'
    session.episode(epochs=3000)
    time.sleep(int(os.getpid()/1000))
    print('run is ok')
    for sequence in session.rat.memory:
        for k in sequence:
            sequence[k] = sequence[k].to('cpu')
    print('to cpu is ok')
    time.sleep(int(os.getpid() / 100))
    with open('mem1024ory' + str(os.getpid()) + '.pkl', 'wb') as f:
        pickle.dump(session.rat.memory, f)

def execute():
    """

    :return:
    """
    pool = multiprocessing.Pool(8)

    for pre_lr_rate in [1e-3, 1e-4, 1e-5]:
        for keep_p in [0.8]:
            for memory_size in [200, 500]:
                for train_epochs in [10, 20, 50]:
                    for train_paras in ['two', 'all']:
                        for device, batch_size in zip(['cuda:0', 'cuda:1'], [50, 100]):
                            for net_hidden_size in [512, 256]:
                                used_args = deepcopy(args)
                                used_args['rat_args']['keep_p'] = keep_p
                                used_args['rat_args']['pre_lr_rate'] = pre_lr_rate
                                used_args['rat_args']['memory_size'] = memory_size
                                used_args['rat_args']['batch_size'] = batch_size
                                used_args['train_epochs'] = train_epochs
                                used_args['rat_args']['net_hidden_size'] = net_hidden_size
                                used_args['rat_args']['device'] = device
                                used_args['rat_args']['train_paras'] = train_paras
                                png_name = 'pre' + \
                                           'lr' + str(pre_lr_rate) + \
                                           'memory' + str(memory_size) + \
                                           'epochs' + str(train_epochs) + \
                                           'batch' + str(batch_size) + \
                                           'hidden' + str(net_hidden_size) + \
                                           'train_paras' + train_paras
                                pool.apply_async(worker, (used_args, png_name))

    pool.close()
    pool.join()

def pre_execute():
    """

    :return:
    """
    # data_pool = multiprocessing.Pool(8)
    # for i in range(8):
    #     print(i)
    #     data_pool.apply_async(get_data)
    # data_pool.close()
    # data_pool.join()
    # print('ok')

    pool = multiprocessing.Pool(9)
    for pre_lr_rate in [1e-2, 1e-3, 1e-4]:
        for keep_p in [0.8]:
            for memory_size in [180000]:
                for batch_size, device in zip([500, 200], ['cuda:0', 'cuda:1']):
                    for net_hidden_size in [1024]:
                        used_args = deepcopy(args)
                        used_args['rat_args']['keep_p'] = keep_p
                        used_args['rat_args']['pre_lr_rate'] = pre_lr_rate
                        used_args['rat_args']['memory_size'] = memory_size
                        used_args['rat_args']['batch_size'] = batch_size
                        used_args['rat_args']['net_hidden_size'] = net_hidden_size
                        used_args['rat_args']['device'] = device
                        png_name = 'pre' + \
                                   'lr' + str(pre_lr_rate) + \
                                   'batch' + str(batch_size) + \
                                   'hidden' + str(net_hidden_size)
                        pool.apply_async(pre_worker, (used_args, png_name))

    pool.close()
    pool.join()

if __name__ == '__main__':
    execute()
