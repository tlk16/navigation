"""
multi-processing for testing super-parameters
"""
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
from environment import RatEnv
from exp_bsd import Rat, Session

args = {
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

        if (i + 1) % 50 == 0:
            session.save_png(png_name + ' i.png', phase=args['rat_args']['train_stage'])



def execute():
    """

    :return:
    """
    pool = multiprocessing.Pool(4)

    for lam in [0, 0.3, 0.6, 0.9]:
        args['rat_args']['lam'] = lam
        pool.apply_async(worker, (args, 'lam' + str(lam)))

    pool.close()
    pool.join()

if __name__ == '__main__':
    execute()
