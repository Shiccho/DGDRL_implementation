import pybullet as p
import time
import pybullet_data
import numpy as np
END_EFFECTER_ID = 7

physicsClientId = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
#VRPhysicsClientId = p.connect(p.SHARED_MEMORY, "1E30BDE6")
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = physicsClientId)  # optionally
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(enableRealTimeSimulation=0, physicsClientId = physicsClientId)
p.resetSimulation()
planeId = p.loadURDF("plane.urdf", physicsClientId = physicsClientId)
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = physicsClientId)
armId = p.loadURDF("xarm/xarm7_robot.urdf", startPos, startOrientation, physicsClientId = physicsClientId)
deltaJt = 0.1/3.14

Js = p.calculateInverseKinematics(armId, END_EFFECTER_ID, [0.3,0,0.2],p.getQuaternionFromEuler([np.pi,0,np.arctan(0.5/0.5)]),
                physicsClientId = physicsClientId)
print(Js)
print(p.getLinkState(armId, 7)[5])
for jointIndex in range(1, p.getNumJoints(armId)):
    p.resetJointState(armId, jointIndex, Js[jointIndex-1],
                    physicsClientId = physicsClientId)
    print(jointIndex, p.getJointState(armId, jointIndex)[0])
print(p.getLinkState(armId, 7)[5])
# set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
while p.isConnected(physicsClientId):
    for jointIndex in range(0, p.getNumJoints(armId)-1):
        if jointIndex == 1:
            jt = p.getJointState(armId, jointIndex)[0]# + deltaJt
            p.setJointMotorControl2(armId, jointIndex, 
            controlMode=p.POSITION_CONTROL,
            targetPosition = jt)
        #print(jointIndex, p.getJointState(armId, jointIndex)[0])
    p.stepSimulation(physicsClientId = physicsClientId)
    time.sleep(1./240.)
    #print(p.getVREvents(physicsClientId = VRPhysicsClientId))
p.disconnect()