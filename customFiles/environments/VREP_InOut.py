from customFiles.environments.constants import robotDefinition as robot
import numpy as np

"""
This file should handle the bulk of the inout related functionality for  RL STaR as an 
intermediary between the simulator file and the task file

This file initiates the data streaming networks and then retrieves and sends data 
"""


class VREP_inout():
    def __init__(self, simulatorObj):
        self.simulator = simulatorObj
        self.initiated = False


    def initiate(self):
        """
        VREP requires all data be initiated before getting the data
        initiates the simulator to be ready to stream data about the base: XYZ location, RPY angles, and velocity
        initiates the simulator to stream joint speed and force/torque
        if your robot is drastically different from Clover, might need to modify this
        :return:
        """
        self.get_handles()
        self.simulator.initiateXYZ()
        self.simulator.initiateRPY()
        self.simulator.initiateVelocity()
        self.simulator.initiateWheelSpeed()
        self.simulator.initiateWheelForce()
        self.initiated = True

        return


    def get_handles(self):
        """
        This function gets the handles for all joints, links, body and markers, this is required for coppeliaSim
            Outputs:
                handles: dictionary of handles for all joints, links and body
        """
        for joint in robot.jointNames:
            self.simulator.getHandle(joint)
        self.simulator.getHandle(robot.baseName)
        for name in robot.Markers:
            self.simulator.getHandle(name)

        return


    def getXYZ(self):
        """
        get the xyz location of the object relative to absolute (origin is 0,0)
        :return:
        """
        xyz = np.array(self.simulator.getXYZ())
        assert self.initiated == True

        return xyz


    def getRPY(self):
        """
        get the roll pitch and yaw of the object
        :return:
        """
        rpy = np.array(self.simulator.getRPY())

        return rpy


    def getVel(self):
        """
        absolute velocity, both linear and angular
        both are 3 dimensions
        :return:
        """

        velX, velAng = np.array(self.simulator.getVel())

        return velX, velAng


    def getForce(self):
        """
        gets the force/torque of the wheel/actuator
        :return:
        """
        force = np.array(self.simulator.getWheelForce())

        return force


    def getWheelSpeed(self):
        """
        gets the wheel/actuator speed
        :return:
        """
        wheelSpeed = np.array(self.simulator.getWheelSpeed())

        return wheelSpeed


    def setWheelSpeed(self, name, action):
        """
        sets the output
        :param action:
        :return:
        """
        self.simulator.setWheelSpeed(name, action)

        return
