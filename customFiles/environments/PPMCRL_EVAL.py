"""
This is the main training file
"""
from customFiles.environments.constants import constTaskPPMCEval as const
from customFiles.environments.constants import simSettings
from customFiles.environments.functions import starFunctions
import numpy as np
import math



class PPMCRL_EVALX():
    def __init__(self, simulatorObj, inoutObj):
        """

        :param inDim:f

        :param outDim:
        :param sceneLocation:
        """
        print("PPMCRL EVAL")
        self.simulator = simulatorObj
        self.inout = inoutObj
        self.isFirst = True
        self.frame = 0
        self.elapsedTime = 0
        self.checkPointReached = False
        self.failedAttempts = 0
        self.epWin = True
        # rewards and penalties
        self.aliveBonus = 1
        self.rewardsMax = 0
        self.rewTrack = 0
        self.velRewTrack = 0
        self.torquePenaltyTrack = 0
        self.fallPenaltyTrack = 0
        self.orientationPenaltyTrack = 0
        self.backupPenaltyTrack = 0
        self.turnPenaltyTrack = 0
        self.alivePenaltyTrack = 0
        self.finishBonusReceived = True
        self.backupSpeed = 0
        self.fallPenaltyReceived = False
        self.minDist = 999
        # actions
        self.action = None
        self.oldAction = np.zeros(const.actionDim)
        self.clippedAction = None
        self.goalCounter = 0
        # print statements
        self.printFinish = 1
        self.printTime = 4
        self.printState = 6
        self.printReward = 9
        #goal setup
        if const.presetWaypoints:
            self.create_presetGoals()
        else:
            self.create_randomGoals(numGoals=const.numGoals)
        self.updateGoals()
        self.completedGoals = []

    def resetVariables(self):
        self.isFirst = True
        self.frame = 0
        self.elapsedTime = 0
        self.checkPointReached = False
        self.completedGoals = []
        # rewards and penalties
        self.aliveBonus = 1
        self.rewardsMax = 0
        self.rewTrack = 0
        self.velRewTrack = 0
        self.fallPenaltyTrack = 0
        self.torquePenaltyTrack = 0
        self.orientationPenaltyTrack = 0
        self.backupPenaltyTrack = 0
        self.turnPenaltyTrack = 0
        self.alivePenaltyTrack = 0
        self.finishBonusReceived = True
        self.backupSpeed = 0
        self.inout.initiated = False
        # actions
        self.action = None
        self.oldAction = np.zeros(const.actionDim)
        self.clippedAction = None
        self.goalCounter = 0
        if const.presetWaypoints:
            self.create_presetGoals()
        else:
            self.create_randomGoals(numGoals=const.numGoals)
        self.updateGoals()
        self.fallPenaltyReceived = False
        self.epWin = False
        self.minDist = 999




    def initiate(self):
        if self.isFirst:
            self.moveMarkers(const.numGoals, self.goalCounter, self.waypointCoords)


    def getState(self):
        """
        Used in RL STaR
        get minimal state for system to learn
        :return:
        """
        self.xyz = self.inout.getXYZ()
        xDist, yDist = self.getWaypoints()
        self.targetArrayABS = np.array([xDist, yDist, 0])
        self.rpy = self.inout.getRPY()
        self.targetArrayREL = starFunctions.abs2rel(self.targetArrayABS, self.rpy)
        self.absVelocityXYZ, self.angularVeloc = self.inout.getVel()

        # self calculate # add xyz in robot frame?
        self.relVelocityXYZ = starFunctions.abs2rel(self.absVelocityXYZ, self.rpy)
        self.distance2Waypoint = self.getDistance2Waypoint(self.xyz)
        self.angle2Waypoint = starFunctions.getAngle2Point(self.waypointArray[0:2], self.xyz, self.rpy)
        # misc
        self.turnSpeed = starFunctions.getTurnSpeed(self.angularVeloc)
        self.backupSpeed = starFunctions.getBackupSpeed(self.relVelocityXYZ)

        self.elapsedTime, self.frame = self.simulator.getTime()

        # normalizing to make it easier to learn
        xyz = self.xyz / const.posFactor
        relVelxyz = self.relVelocityXYZ / const.velFactor  # maybe this should be in terms of actionmult
        angVel = self.angularVeloc / const.angFactor  # maybe this should be in terms of velFactor and actionmult
        angle2Waypoint = self.angle2Waypoint / const.angFactor
        elapsedTime = self.elapsedTime  # / const.maxTime
        targetCoords = np.array(self.targetArrayREL) / const.posFactor
        distance2Waypoint = self.distance2Waypoint / (2 * const.posFactor * 1.41)

        self.state = np.concatenate([xyz[0:2], relVelxyz[0:2], [angVel[2]],
                                     [angle2Waypoint], [distance2Waypoint], targetCoords[0:2]])  # 5


        self.check_boundry()
        # no wheel force

        return self.state


    def getDistance2Waypoint(self, xyz):
        """

        :return:
        """
        if not self.isFirst and not self.checkPointReached:
            self.previousDistance = self.distance2Waypoint
        else:
            self.previousDistance = np.linalg.norm(
                [self.waypointArray[1] - xyz[1],
                 self.waypointArray[0] - xyz[0]]
            )
        self.distance2Waypoint = np.linalg.norm(
            [self.waypointArray[0] - xyz[0],
             self.waypointArray[1] - xyz[1]]
        )

        return self.distance2Waypoint


    def getReward(self):
        self.goalProgress = self.getVelocity2Waypoint()
        self.fallPenaltyReceived, self.fallPenalty = starFunctions.getFallPenalty(self.xyz, self.fallPenaltyReceived)
        # calc rewards and penalties
        self.velocityReward = const.velocityRewardConstant * self.goalProgress
        self.alivePenalty = const.aliveConstant
        self.turnPenalty = const.turnPenaltyConstant * self.turnSpeed  # degree to radians for scaling down
        self.backupPenalty = const.backupPenaltyConstant * self.backupSpeed  # scale down

        self.rewards = [
            self.velocityReward,
            -self.alivePenalty,
            -self.fallPenalty,
            -self.turnPenalty,
            -self.backupPenalty]

        self.reward = sum(self.rewards)

        self.velRewTrack += self.velocityReward
        self.fallPenaltyTrack += self.fallPenalty
        self.alivePenaltyTrack += self.alivePenalty
        self.backupPenaltyTrack += self.backupPenalty
        self.turnPenaltyTrack += self.turnPenalty
        self.rewTrack += self.reward
        self.checkPointReached = False
        self.isFirst = False

        return self.reward


    def updateGoals(self):
        """
        This function sets the randomized goals both at start and when
        a checkpoint is reached.
        Waypoint coords --> all coords

        """
        goalsLeft = (const.numGoals - self.goalCounter)
        if goalsLeft == 0:
            # done this shouldn't be called
            print("done")
        elif goalsLeft == 1:
            # 1 goal left
            self.currentWaypointX = self.waypointCoords[2 * self.goalCounter]
            self.currentWaypointY = self.waypointCoords[2 * self.goalCounter + 1]
            self.nextWaypointX = self.currentWaypointX
            self.nextWaypointY = self.currentWaypointY
        elif goalsLeft > 1:
            self.currentWaypointX = self.waypointCoords[2 * self.goalCounter]
            self.currentWaypointY = self.waypointCoords[2 * self.goalCounter + 1]
            self.nextWaypointX = self.waypointCoords[2 * self.goalCounter + 2]
            self.nextWaypointY = self.waypointCoords[2 * self.goalCounter + 3]

        self.waypointArray = np.array(
            [self.currentWaypointX, self.currentWaypointY, self.nextWaypointX, self.nextWaypointY])

        return

    def getVelocity2Waypoint(self):
        """
        ideally should be a factor of both distance traveled and time taken
        :return:
        """
        goalsLeft = const.numGoals - self.goalCounter
        if goalsLeft <= 0:
            velocity = 0.1
        elif self.checkPointReached:
            goalTime = const.maxTime / (const.numGoals - self.goalCounter)  # broken at end goal
            dist2waypoint = np.linalg.norm([self.completedGoals[-1], self.completedGoals[-2]])
            tRemain = goalTime - self.elapsedTime  # find leftover time for the episode
            print("tRemain = ", tRemain)
            # velocity = (dist2waypoint*dist2waypoint) * 0.01 * (tRemain) + 0.5 # bonus for reaching checkpoint #is this flipped? should amx?
            velocity = 0.5
        else:
            velocity = (self.previousDistance - self.distance2Waypoint) * simSettings.dt

        return velocity

    def getGoalProgress(self):
        """
        alternative to get velocity
        :return:
        """
        epsilon = 0  # 0.05 # maybe this can scale
        movement2Goal = self.minDist - self.distance2Waypoint
        if self.isFirst or self.checkPointReached:
            self.minDist = self.distance2Waypoint  # or could manually calc
            progress = 0.5
            if self.goalCounter == const.numGoals:
                progress = 1.5
        elif movement2Goal > epsilon:
            progress = movement2Goal
            self.minDist = self.distance2Waypoint
        else:
            progress = 0

        return progress


    def getWaypoints(self):
        """
        returns the global frame x and y distance between the robot and the target
        :return:
        """
        self.updateGoals()
        tx = self.waypointArray[0]
        ty = self.waypointArray[1]
        dX = tx - self.xyz[0]
        dY = ty - self.xyz[1]

        return dX, dY

    def getWaypointsAlt(self):
        """

        :return:
        """
        self.updateGoals()

        return self.waypointArray


    def check_boundry(self):
        dis_threshold = 1  # was 1
        if abs(self.waypointArray[1] - self.xyz[1]) < dis_threshold and abs(
                self.waypointArray[0] - self.xyz[0]) < dis_threshold:
            self.completedGoals.append(self.waypointArray[0])
            self.completedGoals.append(self.waypointArray[1])
            self.checkPointReached = True
            self.goalCounter += 1
            self.moveMarkers(const.numGoals, self.goalCounter, self.waypointCoords)
            self.updateGoals()

        return


    def create_randomGoals(self, numGoals=2, center=0, width=10, length=10, minDif=3, boundry=4):
        """
        If isStatis = True: will use user set waypoints
            if False: will randomly give waypoints, as in training
        :param numGoals:
        :param center:
        :param width:
        :param length:
        :param minDif:
        :return:
        """
        if self.failedAttempts > const.numAttempts or self.epWin:
            self.failedAttempts = 0
            rannum = np.random.rand(numGoals * 2)
            rannumCorrected = rannum - 0.5
            waypoints = center + (rannumCorrected) * (width - boundry) * 2 + np.sign(rannumCorrected) * boundry
            for i in range(0, len(rannum) - 2, 2):
                distBetweenWaypoints = np.linalg.norm(
                    [waypoints[i] - waypoints[i + 2], waypoints[i + 1] - waypoints[i + 3]])
                lastCoord = [waypoints[i + 2], waypoints[i + 3], waypoints[i + 2], waypoints[i + 3]]
                if distBetweenWaypoints < minDif:
                    self.create_randomGoals(numGoals=const.numGoals)
            self.waypointCoords = np.append(waypoints, lastCoord)  # repeat twice
        else:
            print("retry same episode")

        print("waypointCoords", self.waypointCoords)
        message = "Waypoint 1: X=%d, Y=%d \nWaypoint2: X=%d, Y=%d" % (
                    self.waypointCoords[0], self.waypointCoords[1], self.waypointCoords[2], self.waypointCoords[3])
        self.simulator.statusBar(message)  # only print first two for now

        return


    def create_presetGoals(self, numGoals=2, center=0, width=10, length=10, minDif=3, boundry=4):

        assert const.presetWaypoints

        self.waypointCoords = [2,0,     0,2,   -2,0,    0,-2,   2,0,
                                4,-4,   -4,-4,  -4,4,    4,4,    4,-4,
                                10,0,   0,10,   -10,0,   0,-10,  10,0,
                                0, 0, 0, 0]

        print("waypointCoords", self.waypointCoords)
        message = "Waypoint 1: X=%d, Y=%d \nWaypoint2: X=%d, Y=%d" % (
        self.waypointCoords[0], self.waypointCoords[1], self.waypointCoords[2], self.waypointCoords[3])
        self.simulator.statusBar(message)  # only print first two for now

        return


    def moveMarkers(self, numGoals, goalCounter, waypointList):
        """
        moves the goal and completed markers

        :return:
        """
        print("waypointlist ", waypointList)
        height1 = 2
        height2 = 6
        numMarkers = 2
        if goalCounter > numGoals - 1:
            currentGoalHeight = height1
            nextGoalHeight = height2
        elif goalCounter == numGoals:
            currentGoalHeight = height2
            nextGoalHeight = height2
        else:
            currentGoalHeight = height1
            nextGoalHeight = height1
        if self.frame<2:
            loc = [waypointList[0], waypointList[1], height1]
            name = "goal1"
            self.simulator.moveObject(loc, name)
            loc = [waypointList[2], waypointList[3], height1]
            name = "goal2"
            self.simulator.moveObject(loc, name)
            loc = [20, 20, height2]
            name = "completed1"
            self.simulator.moveObject(loc, name)
            loc = [20, 20, height2]
            name = "completed2"
            self.simulator.moveObject(loc, name)

        if goalCounter > 0 and goalCounter <= const.numGoals:
            #for i in range(0, goalCounter):
                #name = "completed"+str(i+1)
                #vrep.simxSetObjectPosition(self.clientID, self.handles[name], self.absoluteFrame,
                 #                     [waypointList[2*i], waypointList[2*i+1], height1],vrep.simx_opmode_oneshot)
            remainingGoals = (numGoals - goalCounter)
            loc = [waypointList[2*goalCounter], waypointList[2*goalCounter+1], height1]
            name = "goal1"
            self.simulator.moveObject(loc, name)
            loc = [waypointList[2*goalCounter-2], waypointList[2*goalCounter-1], height2]
            name = "completed1"
            self.simulator.moveObject(loc, name)

            if goalCounter < const.numGoals:
                loc = [waypointList[2*goalCounter-2], waypointList[2*goalCounter-1], height1]
                name = "completed1"
                self.simulator.moveObject(loc, name)
                loc = [waypointList[2 * goalCounter+2], waypointList[2 * goalCounter + 3], height1]
                name = "goal2"
                self.simulator.moveObject(loc, name)
            if goalCounter > 1:
                loc = [waypointList[2*goalCounter-4], waypointList[2*goalCounter-3], height1]
                name = "completed2"
                self.simulator.moveObject(loc, name)

        return



    def getAction(self, action):
        """

        :return:
        """
        self.action = action


    def print(self):
        printStateFreq = 5
        printRewardFreq = 15

        if self.elapsedTime % printStateFreq == 1:
            self.printStatements(self.printState)  # 2 prints state data
        if self.elapsedTime % printRewardFreq == 1:
            self.printStatements(self.printReward)  # 2 prints state data


    def printStatements(self, statement):
        """
        This function prints statements
        Guide:
            1 = end of episode statements
            2 = states
            3 = actions
            4 = frame and time
            5 = velocity of wheels, robot
        """
        if statement == 1:
            print("Goals: ", self.waypointCoords[0], " ", self.waypointCoords[1], " ", self.waypointCoords[2], " ",
                  self.waypointCoords[3], " ")
            print("Robot Pos: ", self.xyz[0], " ", self.xyz[1])
            print("Ep Reward: ", self.rewTrack)
            print("vel rew: ", self.velRewTrack)
            print("time pen: ", -self.alivePenaltyTrack)
            print("fall pen: ", -self.fallPenaltyTrack)
            # print("torque pen: ", -self.torquePenaltyTrack)
            # print("orient pen: ", -self.orientationPenaltyTrack)
            print("turn pen: ", -self.turnPenaltyTrack)
            print("backup pen: ", -self.backupPenaltyTrack)
        if statement == 2:
            print("walk_target_theta = ", math.degrees(self.walk_target_theta))
            print("angle to target = ", math.degrees(self.angle_to_target))
            print("yaw = ", math.degrees(self.yaw))
            # print("state = ", self.s)
            print("walk target dist = ", self.walk_target_dist)
        if statement == 3:
            print("left front wheel output = ", self.LFWheelOutputs)
            print("right front wheel output = ", self.RFWheelOutputs)
            print("left back wheel output = ", self.LBWheelOutputs)
            print("right back wheel output = ", self.RBWheelOutputs)
            print("actions = ", self.action)
        if statement == 4:
            # print("frame = ", self.frame)
            print("Elapsed time, Frame = ", self.elapsedTime, self.frame)
        if statement == 5:
            print("velocity = ", self.velocity)
            print("wheel speeds = ", self.wheelSpeeds)
            print("walk target dist = ", self.walk_target_dist)
        if statement == 6:
            print("target relative position ", self.targetArrayREL, " meters")
            print("angular speed yaw ", np.rad2deg(self.angularVeloc[2]), " degrees/s")
            print("rel vel ", self.relVelocityXYZ, " m/s")
            print("angle2target ", np.rad2deg(self.angle2Waypoint), " degrees")
            print("distance2target ", self.distance2Waypoint, "meters")
            print("Time ", self.elapsedTime, "seconds")
            #print("actions = ", self.clippedAction, "rad/s")
            print("actions = ", self.action, "rad/s")
        if statement == 7:
            print("backup speed: ", self.backupSpeed)
            wheelNames = ["left front", "right front", "left back", "right back"]
            print(wheelNames, self.wheelSpeeds)
            print("turn speed ", self.turnSpeed)
            if self.action is not None:
                print("action vectors L-R", self.action)
                print("clipped actions", self.clippedAction)
        if statement == 8:
            # real units for states
            # print("state data")
            # print("state", self.state)
            print("position xyz ", self.xyz, " meters")
            print("ang rpy ", np.rad2deg(self.rpy), " degrees")
            print("abs velocity ", self.absVelocityXYZ, " m/s")
            print("rel vel ", self.relVelocityXYZ, " m/s")
            print("ang vel ", self.angularVeloc, " deg/s")
            print("yaw sincos ", self.yawSinCos)
            print("angle2target, sincos ", np.rad2deg(self.angle2Waypoint), self.angle2WaypointSinCos, " degrees")
            print("distance2target ", self.distance2Waypoint, "meters")
            print("Wheel Force ", self.wheelForce, "Newtons")
            print("Wheel Speed ", self.wheelSpeed, "rad/s")
            print("Time ", self.elapsedTime, "seconds")
            print("Goals ", self.waypointArray, "meters")
            print("Goals REL ", self.waypointArrayREL, "meters")
            print("actions = ", self.clippedAction, "rad/s")
        if statement == 9:
            print("Goals: ", self.waypointArray[0], " ", self.waypointArray[1])
            print("Robot Pos: ", self.xyz[0], " ", self.xyz[1])
            print("Sum Reward: ", self.reward)
            print("vel rew: ", self.velocityReward)
            print("alive, fall pen: ", -self.alivePenalty, -self.fallPenalty)
            # print("torque, turn, backup pen: ", -self.torquePenalty, -self.turnPenalty, -self.backupPenalty)
            # print("orient pen: ", -self.orientationPenaltyTrack)

        if statement == 11:
            print("target relative position ", self.targetArrayREL, " meters")
            print("angular speed yaw ", np.rad2deg(self.angularVeloc[2]), " degrees/s")
            print("rel vel ", self.relVelocityXYZ, " m/s")
            print("angle2target, ABS, yaw ", np.rad2deg(self.angle2Waypoint), np.rad2deg(self.angle2WaypointABS),
                  np.rad2deg(self.rpy[2]), " degrees")
            print("distance2target ", self.distance2Waypoint, "meters")
            print("Time ", self.elapsedTime, "seconds")
            print("actions = ", self.clippedAction, "rad/s")
        return



    def isDoneX(self):
        """
        This functon checks the robot against fail conditions.
            Inputs:
                None
            Outputs:
                isDone: A boolean denoting if the robot failed or not
                reason: A string denoted which fail condition the robot hit
        """
        self.elapsedTime, self.frame = self.simulator.getTime()
        isDoneY = False
        reason = "none"
        if self.goalCounter == const.numGoals:
            reason = "reached all checkpoints; episode time"
            self.printStatements(self.printFinish)  # 1 prints end statements
            isDoneY = True
        if self.goalCounter == 0 and self.elapsedTime > const.maxTime / 2:
            reason = "Failed to reach 1st checkpoint"
            self.printStatements(4)  # 4 prints frame rate and time
            self.printStatements(self.printFinish)  # 1 prints end statements
            isDoneY = True
        elif self.elapsedTime > const.maxTime:
            reason = "Failed to reach 2nd checkpoint"
            self.printStatements(self.printFinish)  # 1 prints end statements
            isDoneY = True
        elif self.goalCounter == 2:
            reason = "Win!"
            self.printStatements(self.printFinish)  # 1 prints end statements
            self.epWin = True
            self.failedAttempts = 0
        elif abs(self.rpy[1]) > 30 or abs(self.rpy[0]) > 30:
            reason = "Roll or Pitch"
            isDoneY = True
            self.printStatements(self.printFinish)  # 1 prints end statements
        elif self.xyz[2] < -10:
            reason = "fell off map"
            isDoneY = True
            self.printStatements(self.printFinish)  # 1 prints end statements
        elif -self.turnPenaltyTrack < const.turnPenaltyMax or -self.backupPenaltyTrack < const.backupPenaltyMax and self.goalCounter < 1:
            reason = "Turn or backup Penalty too big"
            isDoneY = True
            self.printStatements(self.printFinish)  # 1 prints end statements

        if isDoneY and reason != "Win!":
            self.failedAttempts += 1
        # print("fail cases checked")
        return isDoneY, reason
