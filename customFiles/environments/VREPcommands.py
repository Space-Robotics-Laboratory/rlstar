import subprocess as sp
import random
from customFiles.environments.constants import robotDefinition as robot
from customFiles.environments.constants import simSettings
from collections import OrderedDict
import numpy as np
import sys
"""
This file should handle the bulk of VREP related functionality for PPMCRL
"""
try:
    import vrep
except:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')

print('Program started vrep')
class SimCommands():
    def __init__(self, vrepLoc, sceneLoc):
        self.frame = 0
        self.resetNum = 0
        self.handles = OrderedDict()
        self.getVelocity = 2012  # vrep key
        self.absoluteFrame = -1  # vrep key
        self.first = True
        self.clientID = -1
        self.vrepLoc = vrepLoc
        self.sceneLoc = sceneLoc

    def initiate_vrep(self):
        """
        starts vrep
        """
        self.start_vrep()

        return

    def start_vrep(self):
        """
        opens vrep (coppeliasim) on a random port
        :return:
        """
        self.port_num = int(random.random() * 1000 + 19999)
        args = [self.vrepLoc, '-gREMOTEAPISERVERSERVICE_' + str(self.port_num) + '_FALSE_TRUE', 'scene_file.ttt']
        self.instance = instance(args)
        self.clientID = -1

    def start_simulation(self):
        """
        This function starts the simulation. Creates a clientID, loads scene, establishes synch mode, starts sim and checks ping
                Inputs:
                        scenelocation: where is the scene saved and name of scene
                Outputs:
                        simualtion started: a boolean asserting that simulation has been started
                        clientID: clientID
        """
        startErrors = OrderedDict()
        if self.first == True:
            self.instance.start()
            retries = 0
            while True:
                print('trying to connect to server on port', self.port_num, 'retry:', retries)
                self.clientID = vrep.simxStart(
                    '127.0.0.1', self.port_num,
                    waitUntilConnected=True,
                    doNotReconnectOnceDisconnected=True,
                    timeOutInMs=1000,
                    commThreadCycleInMs=0)  # Connect to V-REP
                if self.clientID != -1:
                    print('Successfully connected')
                    print("ClientID is: ", self.clientID)
                    break
                else:
                    retries += 1
                    if retries > 30:
                        self.end()
                        raise RuntimeError('Problem: unable to connect after 30 retries.')
            startErrors['loadSceneCode'] = vrep.simxLoadScene(self.clientID, self.sceneLoc, 0,
                                                              vrep.simx_opmode_blocking)
            print("start errors :", startErrors)
        self.first = False
        startErrors['pingCode'], pingTime = vrep.simxGetPingTime(self.clientID)  # provides simulator time to configure
        startErrors['syncMode'] = vrep.simxSynchronous(self.clientID, True)  # in sync mode
        startErrors['start'] = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        if all(x == 0 for x in startErrors.values()) == True:
            print("No errors during startup")
        simulationStarted = True
        vrep.simxSetIntegerSignal(self.clientID, 'first', 1, vrep.simx_opmode_blocking)
        #  Now send some data to V-REP in a non-blocking fashion:
        self.resetNum += + 1
        nameX = 'Simulation started: t num time = %d' % (self.resetNum)
        vrep.simxAddStatusbarMessage(self.clientID, nameX, vrep.simx_opmode_oneshot)
        nameX = 'b episode num time = %d' % (self.frame)
        vrep.simxAddStatusbarMessage(self.clientID, nameX, vrep.simx_opmode_oneshot)

        return


    def getHandle(self, name):
        """
        gets the handle of a coppeliasim object using its name
        stores it in an ordered dictionary
        :return:
        """
        error, self.handles[name] = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_blocking)
        if error != 0:
            print("Error retreiving handle for " + name)
        return


    def initiateXYZ(self):
        """
        initializes getting the object position of the base. Coppeliasim requires initialization before getting data
        :return:
        """
        name = robot.baseName
        error, xyz = vrep.simxGetObjectPosition(self.clientID,self.handles[name], self.absoluteFrame, vrep.simx_opmode_streaming)
        print("initiateXYZ")
        if error > 1:
            print("Error initiating xyz")
        return


    def getXYZ(self):
        """
        gets the xyz location of a data in absolute frame
        :return:
        """
        name = robot.baseName
        error, xyz = vrep.simxGetObjectPosition(self.clientID,self.handles[name], self.absoluteFrame, vrep.simx_opmode_buffer)
        if error > 1:
            print("Error getting xyz")

        return xyz


    def initiateRPY(self):
        """
        initiates rpy
        :return:
        """
        name = robot.baseName
        error, rpy = vrep.simxGetObjectOrientation(self.clientID, self.handles[name], self.absoluteFrame, vrep.simx_opmode_streaming)
        if error > 1:
            print("Error initiating rpy")

        return


    def getRPY(self):
        """
        gets rpy
        :return:
        """
        name = robot.baseName
        error, rpy = vrep.simxGetObjectOrientation(self.clientID, self.handles[name], self.absoluteFrame, vrep.simx_opmode_buffer)
        if error > 1:
            print("Error getting rpy")

        return rpy


    def initiateVelocity(self):
        """
        initiates velocity
        :return:
        """
        name = robot.baseName
        error, absVeloc, angularVeloc = vrep.simxGetObjectVelocity(self.clientID, self.handles[name], vrep.simx_opmode_streaming)
        if error > 1:
            print("Error initiating velocity")

        return


    def getVel(self):
        """
        gets velocity: both absolute linear velocity and angular velocity
        :return:
        """
        name = robot.baseName
        error, absVeloc, angularVeloc = vrep.simxGetObjectVelocity(self.clientID, self.handles[name],vrep.simx_opmode_buffer)
        if error > 1:
            print("Error getting velocity")

        return absVeloc, angularVeloc


    def initiateWheelSpeed(self):
        """
        radians/s --> m/s is probably better
        initiates getting the rotational velocity of the wheels
        :return:
        """
        wheelSpeed = np.zeros(robot.numWheels)
        i=0
        for wheel in robot.jointNames:
            error, wheelSpeed[i] = \
                vrep.simxGetObjectFloatParameter(self.clientID, self.handles[wheel],
                                                 self.getVelocity, vrep.simx_opmode_buffer)
            i+=1
        if error > 1:
            print("Error initiating wheel velocity")

        return


    def getWheelSpeed(self):
        """
        gets the wheel speed
        :return:
        """
        wheelSpeed = np.zeros(robot.numWheels)
        i=0
        for wheel in robot.jointNames:
            error, wheelSpeed[i] = \
                vrep.simxGetObjectFloatParameter(self.clientID, self.handles[wheel],
                                                 self.getVelocity, vrep.simx_opmode_streaming)
            i+=1
        if error > 1:
            print("Error getting wheel speed")

        return wheelSpeed


    def initiateWheelForce(self):
        """
        Newtons / Nm
        Initiates getting the wheel force/torque
        :return:
        """
        force = np.zeros(robot.numWheels)
        i = 0
        for wheel in robot.jointNames:
            error, force[i] = \
                vrep.simxGetJointForce(self.clientID, self.handles[wheel], vrep.simx_opmode_buffer)
            i+=1
        if error > 1:
            print("Error initiating wheel force")

        return


    def getWheelForce(self):
        """
        gets the wheel force/torque
        :return:
        """
        force = np.zeros(robot.numWheels)
        i = 0
        for wheel in robot.jointNames:
            error, force[i] = \
                vrep.simxGetJointForce(self.clientID, self.handles[wheel], vrep.simx_opmode_streaming)
            i+=1
        if error > 1:
            print("Error getting wheel force")

        return force


    def setWheelSpeed(self, name, action):
        """
        sets the output in VREP
        Sets the joint motors a certain speed
        :return:
        """
        error = []
        for i in range(len(action)):
            e = vrep.simxSetJointTargetVelocity(self.clientID, self.handles[name[i]], action[i], vrep.simx_opmode_oneshot)
            error.append(e)
        errors = np.array(error)
        if errors.any() > 1:
            print("Error setting wheel velocity for ", errors)

        return


    def step(self):
        """
        steps vrep forward
        :return:
        """
        stepError = vrep.simxSynchronousTrigger(self.clientID)  # steps simulation
        e = vrep.simxGetPingTime(self.clientID)
        self.frame += 1
        self.elapsedTime = self.frame * simSettings.dt

        if self.clientID == -1:
            print('Connected NOT Successful')
            sys.exit('Error: Could Not Connected')
        if stepError == 8:
            print("Step Error", stepError, self.clientID)
            reason = "Error 8"
            self.endGracefully(reason)
        elif stepError == 64:
            print("Step Error", stepError, self.clientID)
            reason = "Error 64: simxStart was not yet called"
            self.endGracefully(reason)

        return


    def simReset(self):
        """
        resets frame and elapsedtime
        :return:
        """
        self.frame = 0
        self.elapsedTime = 0


    def restart(self):
        """
        restarts VREP by stopping the simulation and then starting it again
        :return:
        """
        x = vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        pingError, pingTime = vrep.simxGetPingTime(self.clientID)
        print("\n\nRestarting \nPingtime is: ", pingTime, "\n\n")
        self.simReset()
        self.start_simulation()

        return


    def getTime(self):
        """
        returns elapsedtime and frame
        :return:
        """

        return self.elapsedTime, self.frame


    def moveObject(self, newLocation, name):
        """
        moves an object with handle name to location newLocation
        name is a string, with the handle
        newLocation is a 3D x,y,z absolute location
        :return:
        """
        vrep.simxSetObjectPosition(self.clientID, self.handles[name], self.absoluteFrame,
                                   newLocation, vrep.simx_opmode_oneshot)

        return


    def statusBar(self, message):
        """
        sets a status bar in coppeliasim on the bottom of the page
        :param message:
        :return:
        """
        vrep.simxAddStatusbarMessage(self.clientID, message, vrep.simx_opmode_oneshot)
        return


    def endGracefully(self, reason):
        """
        Ends the program gracefully
        """
        x = vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxFinish(self.clientID)
        self.instance.end()
        print("Program Ended : ", reason)
        sys.exit()


    def close(self):
        """
        This function helps close vrep session gracefully
        """
        reason = "end"
        self.endGracefully(reason)
        return


# https://github.com/ctmakro/vrepper/blob/master/vrepper/vrepper.py
# taken from VREPPER to open multiple VREP scenes
# MIT LICENSE
list_of_instances = []

class instance():
    def __init__(self, args):
        self.args = args
        list_of_instances.append(self)

    def start(self):
        print('(instance) starting...')
        try:
            self.inst = sp.Popen(self.args)
        except EnvironmentError:
            print('(instance) Error: cannot find executable at', self.args[0])
            raise

        return self

    def isAlive(self):
        return True if self.inst.poll() is None else False

    def end(self):
        print('(instance) terminating...')
        if self.isAlive():
            self.inst.terminate()
            retcode = self.inst.wait()
        else:
            retcode = self.inst.returncode
        print('(instance) retcode:', retcode)
        return self

