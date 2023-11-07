from turtle import distance
import pybullet as p
import time
import pybullet_data
import numpy as np

class Env:
    #call reset after init
    def __init__(self, mode=p.GUI, enableRealTimeSimulation=0, endEffecterId = 7):
        # create physicsServer and init joints' positions
        self.physicsClientId = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                        physicsClientId = self.physicsClientId)
        p.setTimeStep(1./60.)
        p.setRealTimeSimulation(enableRealTimeSimulation=enableRealTimeSimulation,
                        physicsClientId = self.physicsClientId)

        self.planeId = 0
        self.cubeId = 0
        self.armId = 0
        self.numJoints = 0

        # tensor for a stack of 20 frames
        self.obs_20 = np.zeros((20, 7))
        self.obs_21 = np.zeros((21, 7))
        self.obs_21_flatten = self.obs_21.reshape(1, -1)

        # keep track of the episode rewards
        self.rewards = []

        self.Js = np.zeros((1,7))
        self.Jg = np.zeros((1,7))
        self.Jt = np.copy(self.Js)

        # limit Jt
        self.limit = np.pi
        self.lowerLimits=[-np.pi,
                          -np.pi/2.,
                          -np.pi,
                          0.0,
                          -np.pi/2,
                          0.0,
                          -np.pi/2]
        self.upperLimits=[np.pi,
                          np.pi/2.,
                          np.pi,
                          np.pi/2,
                          np.pi/2,
                          np.pi/2,
                          np.pi/2]
        self.jointRanges = (np.array(self.upperLimits)-np.array(self.lowerLimits)).tolist()
        self.restPoses = [0.0,
                          0.0,
                          0.0,
                          np.pi/4.,
                          0.0,
                          np.pi/4.,
                          0.0]
        self.done = False
        # episode is done if diff between Jt and Jg is small threshold 
        self.doneThreashold = 0.3
        self.endEffecterId = endEffecterId

    def step(self, action):
        # execute the action in the pybullet environment
        for jointIndex in range(1, self.numJoints):
            jt = p.getJointState(self.armId, jointIndex,
                            physicsClientId = self.physicsClientId)[0]
            jt = jt + action[0, jointIndex-1]
            p.setJointMotorControl2(self.armId, jointIndex, 
                            controlMode=p.POSITION_CONTROL,
                            targetPosition = jt,
                            physicsClientId = self.physicsClientId)
        p.stepSimulation(physicsClientId = self.physicsClientId)
        time.sleep(1./60.)
        for jointIndex in range(1, self.numJoints):
            jt = p.getJointState(self.armId, jointIndex,
                            physicsClientId = self.physicsClientId)[0]
            self.Jt[0,jointIndex-1] = jt
        obs = self.Jt
        ####   define reward   ###
        lambda_o = 0.05
        lambda_j = np.pi * (1/12)
        #culc d_o
        distances_o = [lambda_o]
        closestArmBox = p.getClosestPoints(self.armId, self.cubeId,
                        distance = lambda_o,
                        linkIndexB = -1, 
                        physicsClientId = self.physicsClientId)
        closestArmPlane = p.getClosestPoints(self.armId, self.planeId,
                        distance = lambda_o,
                        linkIndexA = 7,
                        linkIndexB = -1,
                        physicsClientId = self.physicsClientId)
        for point in closestArmBox:
            distances_o.append(point[8])
        for point in closestArmPlane:
            distances_o.append(point[8])
        d_o = np.min(distances_o)
        #culc d_j
        distances_j = []
        distances_j.extend((self.upperLimits - self.Jt).reshape(-1).tolist())
        distances_j.extend((self.Jt - self.lowerLimits).reshape(-1).tolist())
        d_j = np.min(distances_j)
        #def reward
        normReward = -(3e-1)*np.linalg.norm(self.Jt - self.Jg)
        #objectPenalty = - (2.5e1)*np.max([0.0,lambda_o-d_o])
        #jointPenalty = - (1e0)*np.max([0.0,lambda_j-d_j])
        objectPenalty = - (1e1)*np.max([0.0,lambda_o-d_o])
        jointPenalty = - (5e-1)*np.max([0.0,lambda_j-d_j])
        reward = normReward + objectPenalty + jointPenalty
        #print('normReward:',normReward, '\tobjectPenalty:', objectPenalty, '\tjointPenalty', jointPenalty)
        if np.linalg.norm(self.Jt - self.Jg)**2 < self.doneThreashold:
            self.done = True

        # maintain rewards for each step
        self.rewards.append(reward)

        """if self.done:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()"""
        if not self.done:
            # push it to the stack of 20 frames
            self.obs_20 = np.roll(self.obs_20, shift=-1, axis=0)
            self.obs_20[-1] = obs
        self.obs_21[0:20] = self.obs_20
        self.obs_21[20] = self.Jg
        self.obs_21_flatten = self.obs_21.reshape(1, -1)
        return self.obs_21_flatten, reward, self.done

    def reset(self, Ps, Pg):
        """
        ### Reset environment
        Clean up episode info and 4 frame stack
        """
        # reset pybullet environment
        p.resetSimulation(physicsClientId=self.physicsClientId)
        p.setGravity(0, 0, -9.8, physicsClientId = self.physicsClientId)
        self.done = False
        # import object
        self.planeId = p.loadURDF("plane.urdf",
                        physicsClientId = self.physicsClientId)
        self.cubeId = p.loadURDF("cube.urdf",
                        globalScaling = 0.3,
                        basePosition = [0.3, 0, 0.15],
                        baseOrientation = [1, 0, 0, 0],
                        useFixedBase = 1,
                        physicsClientId = self.physicsClientId)
        startPos = [0, 0, 0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0], 
                        physicsClientId = self.physicsClientId)
        self.armId = p.loadURDF("xarm/xarm7_robot.urdf",
                        startPos, startOrientation,
                        physicsClientId = self.physicsClientId)
        self.numJoints = p.getNumJoints(self.armId,
                        physicsClientId = self.physicsClientId)
        for index in range(1, self.numJoints):
            p.changeDynamics(self.armId, index, 
                             jointLowerLimit=self.lowerLimits[index-1],
                             jointUpperLimit=self.upperLimits[index-1],
                             physicsClientId = self.physicsClientId)
        
        # set Js, Jg, Jt
        Os = p.getQuaternionFromEuler([np.pi, 0, np.arctan2(Ps[1], Ps[0])], 
                        physicsClientId = self.physicsClientId)
        Og = p.getQuaternionFromEuler([np.pi, 0, np.arctan2(Pg[1], Pg[0])], 
                        physicsClientId = self.physicsClientId)
        tupleJs = p.calculateInverseKinematics(self.armId, self.endEffecterId, Ps, Os,
                        lowerLimits=self.lowerLimits,
                        upperLimits=self.upperLimits,
                        jointRanges=self.jointRanges,
                        restPoses=self.restPoses,
                        physicsClientId = self.physicsClientId)
        for i in range(len(tupleJs)):
            self.Js[:,i] = tupleJs[i]
        tupleJg = p.calculateInverseKinematics(self.armId, self.endEffecterId, Pg, Og,
                        lowerLimits=self.lowerLimits,
                        upperLimits=self.upperLimits,
                        jointRanges=self.jointRanges,
                        restPoses=self.restPoses,
                        physicsClientId = self.physicsClientId)
        for i in range(len(tupleJg)):
            self.Jg[:,i] = tupleJg[i]
        self.Jt = np.copy(self.Js)

        #set joint states to Js (numJoints = 8, jointIndex is 1, ..., 7. 0 is fixed joint on base)
        for jointIndex in range(1, self.numJoints):
            p.resetJointState(self.armId, jointIndex, self.Js[0,jointIndex-1],
                            physicsClientId = self.physicsClientId)
        
        # reset caches
        for i in range(20):
            self.obs_20[i] = self.Js
        self.obs_21[0:20] = self.obs_20
        self.obs_21[20] = self.Jg
        self.rewards = []
        self.obs_21_flatten = self.obs_21.reshape(1, -1)

        return self.obs_21_flatten
    
    def getInfo(self):
        # set episode information
        episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
        return episode_info




if __name__ == "__main__":
    env = Env(enableRealTimeSimulation=0)
    while p.isConnected(env.physicsClientId):
        obs = env.reset([0.4, 0.4, 0.5],[0,0,0.5])
        #print("init obs", obs)
        action = np.zeros((1,7))
        action[:, 6] = 0.1/np.pi
        done = False
        while not done:
            #obs, reward, done = env.step(0*action)
            for jointIndex in range(1, env.numJoints):
                p.setJointMotorControl2(env.armId, jointIndex, 
                                controlMode=p.POSITION_CONTROL,
                                targetPosition = env.restPoses[jointIndex-1],
                                physicsClientId = env.physicsClientId)
                print(jointIndex, p.getJointState(env.armId, jointIndex,
                            physicsClientId = env.physicsClientId)[0], env.restPoses[jointIndex-1])
            p.stepSimulation(physicsClientId = env.physicsClientId)
            
            #print(obs[0][6])
            #print(reward)
            #print(env.Jt[0])
        info = env.getInfo()
        #print("reward",info["reward"], "length", info["length"])
        #print(env.Jt)
    p.disconnect()