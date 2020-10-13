import numpy as np
import gym
from gym import spaces, error, utils
from customFiles.environments.constants import simSettings  # this class contains simulation parameters
from customFiles.environments.constants import basics
from customFiles.environments.constants import robotDefinition as robot  # this class contains info about the robot
from customFiles.environments.VREPcommands import SimCommands as simulatorX  # this file interfaces with the simulator
from customFiles.environments.VREP_InOut import VREP_inout as inoutX  # this file interfaces all inOUT between task and simualtor files

"""
choose task and constant task class 
"""
from customFiles.environments.PPMCRL import PPMCRLX as taskX  # pick the task you want and import as taskX
from customFiles.environments.constants import constTaskPPMC as const
#from customFiles.environments.PPMCRL_EVAL import PPMCRL_EVALX as taskX
#from customFiles.environments.constants import constTaskPPMCEval as const  # pick the task class in const and import as const

print(taskX)

class clover(gym.Env):
    def __init__(self):  # , headless=False):
        """
        mandatory initiator. Grabs scene, starts simulation, gets handles,
        sets number of dimensions, defines action space and asserts that it started

        """
        self.isFirst = True
        self.createObjects()
        self.initiateModules()
        self.simulator.initiate_vrep()
        high = np.array([np.inf] * const.obsDim)  # idk
        self.action_space = spaces.Box(0 * np.ones(const.actionDim, dtype=np.float32),
                                       np.ones(const.actionDim, dtype=np.float32))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        return


    def createObjects(self):
        """
        creates objects of the simulator, inout and task classes, passes these as appropriate
        :return:
        """
        self.simulator = simulatorX(basics.vrepLocation, const.sceneLocation)
        self.inout = inoutX(self.simulator)
        self.task = taskX(self.simulator, self.inout)
        return


    def initiateModules(self):
        """
        should be initiated only after the simulator has been started. initiates variables, simulator data streams, etc
        :return:
        """
        self.inout.initiate()
        self.initiateInOutModule()
        self.task.initiate()

        return

    def initiateInOutModule(self):
        self.inout.initiate()


    def self_observe(self):
        """
        This function grabs all the state data from the simulator and prints out some info that can be useful for debugging/monitoring training
        """
        self.state = self.task.getState()
        self.elapsedTime, self.frame = self.simulator.getTime()
        self.task.print()

        return


    def step(self, action):
        """
        This function steps the simulation forward. VREP is run in synch mode so it waits to be triggered.
            Inputs:
                action: a list of all the actions for the robot to take, such as next joint velocities
            Outputs:
                isDone: a boolean telling is the robot created a failing state which would signal us to end the simulation
                reward: reward from the previous state
        """
        self.self_observe()
        self.task.getAction(action)
        outputsSet = self.setOutputs(action)
        assert outputsSet
        for i in range(simSettings.numStep):  # step simulation forward numstep number of times, was 5 for rlstar
            self.simulator.step()
        reward = self.calcReward()
        isDone, reason = self.task.isDoneX()

        if isDone:
            print("\nReason: ", reason)

        return self.state, reward, isDone, {}


    def calcReward(self):
        """
        This function calculations reward for the previous state
            Inputs:
                None
            Outputs:
                reward: the reward for the last state. We must pick a proper reward to train the robot to perform the desired task
        """
        reward = self.task.getReward()

        return reward

    def setOutputs(self, action):
        """
        This function sets the outputs for the robot's actuators, namely its wheel speeds
            Inputs:
                action, obtained from the neural network/RL algorithm
            Outputs:
                None
        clover's wheel radius is 0.11m; equation for velocity is v = rw_rad or 2*Pi*rW_cyc
        clover is rear wheel drive, skid steer robot this means left side wheels go the same speed and the right side
        wheels go the same speed. the robot turns by creating a differential in the wheel speeds on the two sides
        """
        action *= const.actionMultiplier
        self.inout.setWheelSpeed(robot.frontJoints, action)
        self.inout.setWheelSpeed(robot.backJoints, action)

        outputsSet = True

        return outputsSet


    def reset(self):
        """
        This function stops and resets the simulation and the different file variables.
        """
        self.task.resetVariables()
        self.simulator.restart()
        self.initiateModules()
        self.self_observe()

        return self.state
