"""
multi-processing for testing super-parameters
"""
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
from environment import RatEnv
from exp_bsd import Rat, Session


def worker(input_type='touch', epsilon=(0.9, 0.002, 0.1), train_paras='all', wall_reward=0.0, step_reward=-0.005, lam=0.3):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    n_train = 800
    rat = Rat(memory_size=1000, input_type=input_type, train_paras=train_paras, device='cuda:1')
    rat.lam = lam
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
        session.experiment(epochs=50)

        session.phase = 'test'
        session.experiment(epochs=10)

        if (i + 1) % 50 == 0:
            session.save_png('last ' + input_type + '[' + str(epsilon[0]) + ' ' +
                             str(epsilon[1]) + ' ' + str(epsilon[2]) + ']' +
                             train_paras + ' ' + str(wall_reward) + str(step_reward) + str(i) + '.png', phase='q_learning')



def execute():
    """

    :return:
    """
    pool = multiprocessing.Pool(4)

    for lam in [0, 0.3, 0.6, 0.9]:
        pool.apply_async(worker, ('touch', (1, 0.002, 0.1), 'all', 0, -0.005, lam))

    pool.close()
    pool.join()

if __name__ == '__main__':
    execute()
