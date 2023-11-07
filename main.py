from worker import Worker
import numpy as np
from model import ActorCriticNet
import torch
from torch import optim
from torch.distributions import MultivariateNormal, Normal
import random
import pybullet as p
import time
from environment import Env
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class Main:
    def __init__(self, PsPgList):
        self.gamma = 0.99
        self.lamda = 0.95

        self.updates = 100000
        self.epochs = 4
        self.numWorkers = 8
        self.workerSteps = 64
        self.maxSteps = self.workerSteps * 3
        self.numMiniBatch = 1
        self.batchSize = 512
        self.miniBatchSize = self.batchSize // self.numMiniBatch
        assert (self.batchSize % self.numMiniBatch == 0)

        self.deltaMax = np.pi/60

        self.workers = []
        for i in range(self.numWorkers):
            if i == 0:
                self.workers.append(Worker(mode=p.GUI))
            else:
                self.workers.append(Worker(mode=p.DIRECT))
        
        self.PsPgList = PsPgList[0:700]
        self.testPsPgList = PsPgList[700:1000]
        self.obs = np.zeros((self.numWorkers, 21*7), dtype=np.float32)
        for worker in self.workers:
            PsPg = self.PsPgList[np.random.randint(0, len(self.PsPgList)-1)]
            worker.child.send(("reset", PsPg))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        self.model = ActorCriticNet().to(device)
        self.model.load_state_dict(torch.load('DynamicPath4/data/model.pth'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

        self.prebRstd = 1.

        self.log = {}
        self.log['reward'] = 0
        self.log['length'] = 0
        self.log['policy_reward'] = 0
        self.log['vf_loss'] = 0
        self.log['entropy_bonus'] = 0

        self.testLog = {}
        self.testLog['success'] = 0
        self.testObs = np.zeros((self.numWorkers, 21*7), dtype=np.float32)

    
    def obsToTorch(self, obs):
        #normalize to Normal[0, 1]
        #obs = (obs - np.mean(obs, axis=0, keepdims=True)) / (np.std(obs, axis=0, keepdims=True) + 1e-4)
        #scale [-pi, pi] to [0,1]
        obs = (obs/(2.*np.pi)) + 0.5
        return torch.tensor(obs, dtype=torch.float32, device=device)

    def sample(self):
        """sample data with current policy"""
        rewards = np.zeros((self.numWorkers, self.workerSteps), dtype = np.float32)
        actions = np.zeros((self.numWorkers, self.workerSteps, 7), dtype = np.float32)
        done = np.zeros((self.numWorkers, self.workerSteps), dtype = bool)
        obs = np.zeros((self.numWorkers, self.workerSteps, 21*7), dtype = np.float32)
        values = np.zeros((self.numWorkers, self.workerSteps), dtype = np.float32)
        logPis = np.zeros((self.numWorkers, self.workerSteps), dtype=np.float32)
        actionParams = np.zeros((self.numWorkers, self.workerSteps, 7), dtype=np.float32)

        for w, worker in enumerate(self.workers):
            if worker.step > self.maxSteps:
                PsPg = self.PsPgList[np.random.randint(0,len(self.PsPgList)-1)]
                #print(PsPg)
                worker.child.send(("reset", PsPg))
        for w, worker in enumerate(self.workers):
            if worker.step > self.maxSteps:
                self.obs[w] = worker.child.recv()
                worker.step = 0
        
        for t in range(self.workerSteps):
            with torch.no_grad():
                obs[:, t] = self.obs
                pi, v = self.model(self.obsToTorch(self.obs))
                a = pi.sample()
                action = self.deltaMax * torch.tanh(a)
                values[:, t] = v.cpu().numpy()
                actionParams[:, t] = a.cpu().numpy()
                actions[:, t] = action.cpu().numpy()
                logPis[:, t] = pi.log_prob(a).cpu().numpy()
                
            for w, worker in enumerate(self.workers):
                worker.step += 1
                worker.child.send(("step", actions[w, t].reshape(1, -1)))
            for w, worker in enumerate(self.workers):
                self.obs[w], rewards[w, t], done[w, t] = worker.child.recv()
            for w, worker in enumerate(self.workers):
                if done[w, t] == True:
                    worker.child.send(("getInfo", None))
            for w, worker in enumerate(self.workers):
                if done[w, t] == True:
                    info = worker.child.recv()
                    if info:
                        self.log['reward'] = info['reward']
                        self.log['length'] = info['length']
            for w, worker in enumerate(self.workers):
                if done[w, t] == True:
                    PsPg = self.PsPgList[np.random.randint(0,len(self.PsPgList)-1)]
                    #print(PsPg)
                    worker.child.send(("reset", PsPg))
            for w, worker in enumerate(self.workers):
                if done[w, t] == True:
                    self.obs[w] = worker.child.recv()
                    worker.step = 0
        
        #self.prebRstd = rewards.std()
        #rewards = rewards / self.prebRstd

        advantages = self._calcAdvantages(done, rewards, values)
        samples = {
            'obs' : obs,
            'actions' : actions,
            'values' : values,
            'advantages' : advantages,
            'logPis' : logPis,
            'actionParams' : actionParams
        }

        # [worker, t] table -> [worker * t] table to ignore t
        # we want to get the relation only between [obs, a]. w, t have no value. 
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0]*v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = self.obsToTorch(v)
            elif k == 'advantages' :
                #samples_flat[k] = torch.tensor(v, device=device).clamp(min=-10., max=10.)
                samples_flat[k] = torch.tensor(v, device=device)
            else:
                samples_flat[k] = torch.tensor(v, device=device)
        return samples_flat

    def _calcAdvantages(self, done, rewards, values):
        advantages = np.zeros((self.numWorkers, self.workerSteps), dtype=np.float32)
        lastAdvantage = 0

        _, lastValue = self.model(self.obsToTorch(self.obs))
        lastValue = lastValue.cpu().data.numpy()

        for t in reversed(range(self.workerSteps)):
            mask = 1.0 - done[:, t]
            lastValue = lastValue * mask
            lastAdvantage = lastAdvantage * mask
            delta = rewards[:, t] + self.gamma * lastValue - values[:, t]
            lastAdvantage = delta + self.gamma * self.lamda * lastAdvantage
            advantages[:, t] = lastAdvantage
            lastValue = values[:, t]
        
        return advantages

    def train(self, samples, learningRate, clipRange):
        for _ in range(self.epochs):
            indexes = torch.randperm(self.batchSize)

            for start in range(0, self.batchSize, self.miniBatchSize):
                end = start + self.miniBatchSize
                miniBatchIndexes = indexes[start: end]
                miniBatch = {}
                for k, v in samples.items():
                    miniBatch[k] = v[miniBatchIndexes]
                
                loss = self._calcLoss(clipRange=clipRange, samples=miniBatch)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = learningRate
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5, norm_type=2.0)
                self.optimizer.step()

    @staticmethod
    def _normalize(adv):
        #return (adv - adv.mean()) / (adv.std() + 1e-4)
        return (adv) / (adv.std() + 1e-4)
    
    def _calcLoss(self, samples, clipRange):
        sampledReturn = samples['values'] + samples['advantages']
        #sampledReturn = self._normalize(sampledReturn) #
        #sampledNormalizedAdvantage = self._normalize(samples['advantages'])
        sampledAdvantage = samples['advantages']
        sampledValue = samples['values']
        pi, value= self.model(samples['obs'])
        #print('value', value[0].item(),'\t', 'actionParam', pi.sample()[0][0].item())
        logPi = pi.log_prob(samples['actionParams'])
        logPiOld = samples['logPis']
        ratio = torch.exp(logPi - logPiOld)
        clippedRatio = ratio.clamp(min=1.0 - clipRange, max=1.0 + clipRange)
        policyReward = torch.min(ratio * sampledAdvantage, 
                                 clippedRatio * sampledAdvantage)
        policyReward = policyReward.mean()

        entropyBonus = pi.entropy()
        entropyBonus = entropyBonus.mean()
        valClipRange = 1.e2 * clipRange
        clippedValue = sampledValue + (value - sampledValue).clamp(min = -valClipRange, max = valClipRange)
        #clippedValue = value.clamp(min = (1.0-clipRange)*sampledValue, max = (1.0+clipRange)*sampledValue)
        Lvf = torch.max((value - sampledReturn)**2, (clippedValue - sampledReturn)**2)
        Lvf = Lvf.mean()

        c1 = 5e-1
        c2 = 1e-2
        loss = -(policyReward - c1*Lvf + c2*entropyBonus)

        clip_fraction = (abs((ratio - 1.0)) > clipRange).to(torch.float).mean()

        self.log['policy_reward'] = policyReward.item()
        self.log['vf_loss'] = Lvf.item()
        self.log['entropy_bonus'] = entropyBonus.item()
        self.log['clip_fraction'] = clip_fraction.item()
        
        return loss

    def test(self):
        actions = np.zeros((self.numWorkers, 7), dtype = np.float32)
        done = np.zeros((self.numWorkers), dtype = bool)
        obs = np.zeros((self.numWorkers, 21*7), dtype = np.float32)
        testSuccess = np.zeros((self.numWorkers), dtype=bool)
        testStep = 0
        #reset env to test
        for w, worker in enumerate(self.workers):
            PsPg = self.testPsPgList[np.random.randint(0,len(self.testPsPgList)-1)]
            worker.child.send(("reset", PsPg))
        for w, worker in enumerate(self.workers):
            self.testObs[w] = worker.child.recv()
            worker.step = 0
        
        while testStep < self.maxSteps * 2:
            testStep += 1
            with torch.no_grad():
                obs = self.testObs
                pi, _ = self.model(self.obsToTorch(obs))
                a = pi.sample()
                actions = (self.deltaMax * torch.tanh(a)).cpu().numpy()
            for w, worker in enumerate(self.workers):
                worker.step += 1
                worker.child.send(("step", actions[w].reshape(1, -1)))
            for w, worker in enumerate(self.workers):
                self.testObs[w], _, done[w] = worker.child.recv()
            for w, worker in enumerate(self.workers):
                if done[w] == True:
                    testSuccess[w] = True
        self.testLog['success'] = (testSuccess.tolist().count(True))
        #reset env to train
        for w, worker in enumerate(self.workers):
            PsPg = self.PsPgList[np.random.randint(0,len(self.PsPgList)-1)]
            worker.child.send(("reset", PsPg))
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()
            worker.step = 0          


    def runTrainingLoop(self):
        writer = SummaryWriter(log_dir="logs")
        for update in range(0, self.updates):
            progress = update / self.updates
            learningRate = 2.5e-4 * (1-(1.0/2.5)*progress)
            clipRange = 0.2 * (1-progress)

            samples = self.sample()
            self.train(samples, learningRate, clipRange)

            if (update + 1) % 1 == 0:
                print('===========================================================================================================')
                print("Progress:{0}/{1}\t|Loss:{2:.3f}\t Reward:{3:.1f}\tLength:{4}\n \
                       -----------------------------------------------------------------------------------\n \
                       |PolicyReward:{5:.3f}\tVF_Loss:{6:.3f}\tEntropyBounus:{7:.3f}\tClipFraction:{8:.2f}".format(
                       update, self.updates,
                       -(self.log['policy_reward'] - 5e-1 * self.log['vf_loss'] + 1e-2 * self.log['entropy_bonus']),
                       self.log['reward'],
                       self.log['length'],
                       self.log['policy_reward'],
                       self.log['vf_loss']*5e-1, 
                       self.log['entropy_bonus']*1e-2,
                       self.log['clip_fraction']
                       ))
                print('===========================================================================================================')
                writer.add_scalar('Loss/Lclip', self.log['policy_reward'], global_step=update)
                writer.add_scalar('Loss/Lvf', self.log['vf_loss'], global_step=update)
                writer.add_scalar('Loss/S(pi)', self.log['entropy_bonus'], global_step=update)
                
            if (update + 1) % 1 == 0:
                model_path = 'model.pth'
                torch.save(self.model.state_dict(), model_path)
                self.test()
                writer.add_scalar('Success Rate', self.testLog['success']/self.numWorkers, global_step=update)
            
          

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))
            p.disconnect(self.testEnv.physicsClientId)


if __name__ == "__main__":
    PsPgList = []
    pList = []
    f = open('DynamicPath4/data/points.txt', 'r')
    data = f.read()
    points = data.split('\n')
    for point in points:
        pList.append(list(map(float, point.split(' '))))
    f.close()
    for _ in range(1000):
        Ps = random.choice(pList)
        Pg = random.choice(pList)
        PsPgList.append((Ps, Pg))
    length = len(PsPgList)

    m = Main(PsPgList)
    m.runTrainingLoop()
    m.destroy()


