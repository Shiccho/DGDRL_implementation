import multiprocessing
import multiprocessing.connection as mpConnection
from environment import Env
import pybullet as p


def worker_process(remote: mpConnection, mode):
    #create pybullet env
    env = Env(mode=mode, enableRealTimeSimulation=0, endEffecterId = 7)

    #wait for instructions from connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "reset":
            remote.send(env.reset(data[0],data[1]))
        elif cmd == "getInfo":
            remote.send(env.getInfo())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    def __init__(self, mode=p.GUI):
        self.child, parent = multiprocessing.Pipe()
        self.step = 0
        self.process = multiprocessing.Process(target=worker_process, args=(parent, mode))
        self.process.start()

if __name__ == "__main__":
    import numpy as np
    worker1 = Worker(mode=p.DIRECT)
    worker2 = Worker()
    workers = [worker1, worker2]
    while 100:
        for worker in workers:
            worker.child.send(("reset", ([0,0.4,0.5], [0.4,0,0.5])))
        for worker in workers:
            worker.child.recv()
        #print("init obs", obs)
        action = np.zeros((1,7))
        action[:, 0] = 0.1/np.pi
        done = False
        count = 0
        while count < 100:
            i = 1
            for worker in workers:
                worker.child.send(("step",i*action))
                i += 4
            for worker in workers:
                obs, reward, done = worker.child.recv()
        for worler in workers:
            info = worker.child.send(("getInfo", None))
        #print("reward",info["reward"], "length", info["length"])
        count += 1
    for worker in workers: 
        worker.child.send("close", None)