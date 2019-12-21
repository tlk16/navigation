import sys
import time
import multiprocessing
from matplotlib import pyplot as plt
from exp_bsd import Rat, Session
from environment import RatEnv

def worker():
    n_train = 1000
    rat = Rat(memory_size=100)
    env = RatEnv(dim=[30, 30, 100], speed=1., goal=[10, 10, 2], limit=100, wall_offset=1., touch_offset=2.)
    session = Session(rat, env)
    for i in range(n_train):
        rat.epsilon = 0.5 - i * 0.002 if i < 200 else 0.1

        session.phase = 'train'
        session.experiment(epochs=10)

        session.phase = 'test'
        session.experiment(epochs=10)

        if i % 10 == 0:
            plt.figure()
            line1, = plt.plot(session.rat.losses, label='loss')
            line2, = plt.plot(session.mean_rewards['train'], label='train')
            line3, = plt.plot(session.mean_rewards['test'], label='test')
            plt.legend(handles=[line1, line2, line3])
            plt.savefig(str(i) + '.png')
            plt.ion()
            plt.pause(5)
            plt.close()



def execute():
    process = []
    for i in range(4):
        t = multiprocessing.Process(target=worker, args=(i,))
        t.start()
        print(t.pid)
        process.append(t)

    for p in process:
        p.join()

execute()