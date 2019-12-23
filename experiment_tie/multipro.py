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
    n_train = 300
    rat = Rat(memory_size=100, input_type=input_type, train_paras=train_paras)
    env = RatEnv(dim=[15, 15, 100], speed=1., collect=False, goal=[10, 10, 1],
                 limit=60, wall_offset=1., touch_offset=2., wall_reward=wall_reward, step_reward=step_reward)
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
            plt.figure()
            line1, = plt.plot(session.rat.losses, label='loss')
            line2, = plt.plot(session.mean_rewards['train'], label='train')
            line3, = plt.plot(session.mean_rewards['test'], label='test')
            cum_train = np.true_divide(
                np.cumsum(np.array(session.mean_rewards['train'])),
                np.array([i + 1 for i in range(len(session.mean_rewards['train']))])
            )
            cum_test = np.true_divide(
                np.cumsum(np.array(session.mean_rewards['test'])),
                np.array([i + 1 for i in range(len(session.mean_rewards['test']))])
            )
            line4, = plt.plot(cum_train, label='train_cum')
            line5, = plt.plot(cum_test, label='test_cum')
            plt.legend(handles=[line1, line2, line3, line4, line5])
            plt.savefig(input_type + '[' + str(epsilon[0]) + ' ' +
                        str(epsilon[1]) + ' ' + str(epsilon[2]) + ']' +
                        train_paras + ' ' + str(wall_reward) + str(step_reward) + str(i) + '.png')
            plt.ion()
            plt.pause(5)
            plt.close()



def execute():
    """

    :return:
    """
    pool = multiprocessing.Pool(2)

    for input_type in ['touch']:
        for train_paras in ['two', 'all']:
            for rewards in [(-0.05, -0.01), (0, 0)]:
                pool.apply_async(worker, (input_type, (0.8, 0.004, 0.2), train_paras, rewards[0], rewards[1]))

    pool.close()
    pool.join()

if __name__ == '__main__':
    execute()
