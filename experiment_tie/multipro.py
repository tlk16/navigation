"""
multi-processing for testing super-parameters
"""
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
from environment import RatEnv
from exp_bsd import Rat, Session

def worker(input_type='touch', epsilon=(0.5, 0.002, 0.1), train_paras='two', wall_reward=0.0, step_reward=0.0):
    """

    :param input_type:
    :param epsilon:
    :param train_paras:
    :return:
    """
    n_train = 500
    rat = Rat(memory_size=100, input_type=input_type, train_paras=train_paras)
    env = RatEnv(dim=[15, 15, 100], speed=1., collect=False, goal=[10, 10, 1],
                 limit=100, wall_offset=1., touch_offset=2., wall_reward=wall_reward, step_reward=step_reward)
    session = Session(rat, env)
    for i in range(n_train):
        print(i)
        rat.epsilon = epsilon[0] - i * epsilon[1] \
            if epsilon[0] - i * epsilon[1] > epsilon[2] else epsilon[2]

        session.phase = 'train'
        session.experiment(epochs=50)

        session.phase = 'test'
        session.experiment(epochs=10)

        if (i+1) % 20 == 0:
            session.save_png(input_type + '[' + str(epsilon[0]) + ' ' +
                        str(epsilon[1]) + ' ' + str(epsilon[2]) + ']' +
                        train_paras + ' ' + str(wall_reward) + str(step_reward) + str(i) + '.png')



def execute():
    """

    :return:
    """
    pool = multiprocessing.Pool(4)

    for input_type in ['touch']:
        for train_paras in ['all', 'two']:
            for rewards in [(0, 0)]:
                for epsilon in [(0.8, 0.004, 0.2), (0.8, 0.002, 0.2)]:
                    pool.apply_async(worker, (input_type, epsilon, train_paras, rewards[0], rewards[1]))

    pool.close()
    pool.join()

if __name__ == '__main__':
    execute()
