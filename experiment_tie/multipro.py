"""
multi-processing for testing super-parameters
"""
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
    'n_train': 800
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
    'train_epochs': 20,  # if train_epochs larger than rat_args batch_size, experience will be somehow wasted
    'test_epochs': 5,
    'n_train': 4000,
    'show_time': 100, # save fig per _ step

}


def worker(args, png_name):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
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

def pre_worker(args, png_name):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    rat = Rat(**args['rat_args'])
    env = RatEnv(**args['env_args'])

    n_train = args['n_train']
    epsilon = args['epsilon']
    train_epochs = args['train_epochs']
    test_epochs = args['test_epochs']
    start = args['start']
    show_time = args['show_time']

    session = Session(rat, env)

    session.phase = 'train'
    session.episode(epochs=train_epochs)
    for i in range(train_epochs):
        session.rat.train()

    session.phase = 'test'
    session.experiment(epochs=test_epochs)

    session.save_png(png_name + '.png', phase=args['rat_args']['train_stage'])

def execute():
    """

    :return:
    """
    pool = multiprocessing.Pool(15)

    pool.apply_async(worker, (typical_args, 'q_leanring_typical'))
    for pre_lr_rate in [1e-5, 1e-4, 1e-3]:
        for keep_p in [0.8]:
            for memory_size in [200, 500]:
                for train_epochs in [10, 20, 50]:
                    for batch_size in [50, 100]:
                        for net_hidden_size in [256, 512]:
                            used_args = deepcopy(args)
                            used_args['rat_args']['keep_p'] = keep_p
                            used_args['rat_args']['pre_lr_rate'] = pre_lr_rate
                            used_args['rat_args']['memory_size'] = memory_size
                            used_args['rat_args']['batch_size'] = batch_size
                            used_args['train_epochs'] = train_epochs
                            used_args['rat_args']['net_hidden_size'] = net_hidden_size
                            png_name = 'pre' + \
                                       'lr' + str(pre_lr_rate) + \
                                       'memory' + str(memory_size) + \
                                       'epochs' + str(train_epochs) + \
                                       'batch' + str(batch_size) + \
                                       'hidden' + str(net_hidden_size)
                            pool.apply_async(worker, (used_args, png_name))

    pool.close()
    pool.join()

if __name__ == '__main__':
    execute()
