import math
import numpy as np
from customFiles.environments.constants import constTaskPPMCEval as const


class starFunctions():
    """
    This class contains functions that are applicable to multiple tasks and robots
    """
    def abs2rel(absValue, rpy):
        """
        might be worth comparing this to the VREP derived rel veloc
        transforms from an absolute frame to a local frame given yaw pitch roll of the robot
        yaw uses right hand rule for convention so turning left is pos, but also my x-y axis are mixed from the norm
        :return:
        """  # x is roll, y is pitch, z is yaw its rpy not rpy
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(rpy[0]), np.sin(rpy[0])],
             [0, -np.sin(rpy[0]), np.cos(rpy[0])]])
        Ry = np.array(
            [[np.cos(rpy[1]), 0, np.sin(rpy[1])],
             [0, 1, 0],
             [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
        Rz = np.array(
            [[np.cos(rpy[2]), np.sin(rpy[2]), 0],
             [-np.sin(rpy[2]), np.cos(rpy[2]), 0],
             [0, 0, 1]])

        relValue = Rz @ absValue

        return relValue

    def getTurnSpeed(angularVelocity):
        """
        turn speed should be proportional to angular yaw
        only get values if noticeable ("e")
        :return:
        """
        e = 0.025
        if abs(angularVelocity[2]) > e:
            turnSpeed = abs(angularVelocity[2])
        else:
            turnSpeed = 0

        return turnSpeed



    def getTurnSpeedAlternative(wheelSpeed):
        """
        alternative turn speed function
        :return:
        """
        turnSpeed = abs(wheelSpeed[0] - wheelSpeed[1])

        return turnSpeed

    def getBackupSpeed(relVelocity):
        """
        get the backup speed of the rover
        :return:
        """
        e = 0.0
        if relVelocity[0] < -e:
            backupSpeed = 1
        else:
            backupSpeed = 0

        return backupSpeed


    def getFallPenalty(xyz, fallPenaltyReceived):
        """
        only receive the fall penalty once, fallPenaltyReceived indicates if already received or not
        :return:
        """
        if xyz[2] < -3 and not fallPenaltyReceived:
            fallPenaltyReceived = True
            fallPenalty = const.fallConstant
        else:
            fallPenalty = 0

        return fallPenaltyReceived, fallPenalty


    def getAngle2Point(waypointArray, xyz, rpy):
        """
        arctan
        corrects the final angle to be between -pi, pi
        :return:
        """
        yaw = rpy[2]
        dy = waypointArray[1] - xyz[1]
        dx = waypointArray[0] - xyz[0]
        angle2PointABS = np.arctan2(dy, dx)
        angle2PointREL = angle2PointABS - yaw
        if angle2PointREL > math.pi:
            angle2PointREL -= 2 * math.pi
        elif angle2PointREL < -math.pi:
            angle2PointREL += 2 * math.pi

        return angle2PointREL


    def getSinCos(angle):
        """

        :return:
        """
        sinx = np.sin(angle)
        cosx = np.cos(angle)
        angles = np.array([sinx, cosx])

        return angles


    def actionClipper(oldAction, action, relVelocityXYZ):
        """
        This functon clips the action to make sure there is not too drastic of a change
            Inputs:
                action
            Outputs:
                clipped action
        """
        clippedAction = np.zeros(const.actionDim)
        delta = action - oldAction
        for i in range(len(delta)):
            if delta[i] > const.maxDeltaV:  # current action > old action
                clippedAction[i] = oldAction[i] + const.maxDeltaV
            if delta[i] < -const.maxDeltaV:  # old action > current action
                clippedAction[i] = oldAction[i] - const.maxDeltaV
            else:
                clippedAction[i] = action[i]
        oldAction = clippedAction

        clippedAction = action
        if relVelocityXYZ[0] < 0:
            clippedAction /= 2  #  want it to be more careful backwards
        clippedAction = clippedAction

        return clippedAction