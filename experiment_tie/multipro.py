import sys
import time
import multiprocessing

def session(index):
    print(index)
    time.sleep(1000)

def execute():
    process = []
    for i in range(4):
        t = multiprocessing.Process(target=session, args=(i,))
        t.start()
        process.append(t)

    for p in process:
        p.join()

execute()