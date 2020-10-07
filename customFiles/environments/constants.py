from collections import OrderedDict
import math

"""
In order to share one constants file between the different tasks of the robot, it is easier to break them down into classes
"""
class basics():
    """
    File Locations
    """
    sceneLocation = '/home/lunar3/Desktop/PythonProjects/star/simulationScenes/clover2.ttt' #'/home/majinbuu/PycharmProjects/cloverski/customFiles/environments/clover2.ttt'
    vrepLocation = "/home/lunar3/CoppeliaSim/coppeliaSim.sh"

class robotDefinition():
    """
    Robot definition. Wheels are actuators in CoppeliaSim
    the CLOVER rover only has two motors, so the rear and front wheels travel the same speed
    """
    wheelNames = ["left_front", "right_front", "left_rear", "right_rear"]
    frontWheels = ["left_front", "right_front"]
    backWheels = ["left_rear", "right_rear"]
    numWheels = 4
    baseName = "base_link_respondable"
    jointNames = ["joint_left_front_wheel", "joint_right_front_wheel", "joint_left_rear_wheel",
                  "joint_right_rear_wheel"]
    frontJoints = ["joint_left_front_wheel", "joint_right_front_wheel"]
    backJoints = ["joint_left_rear_wheel", "joint_right_rear_wheel"]
    Markers = ["goal1", "goal2", "completed1", "completed2", "origin"]

class simSettings():
    """
    sim settings
    """
    dt = 0.10  # ms, set in sim
    numStep = 5  # dt * numStep = trueDT = 2s
    t = 0

class constTaskPPMC():
    """
    Holds all the constant for the robot tasks
    """
    """  
    Scene location  
    """
    sceneLocation = '/home/lunar3/Desktop/PythonProjects/star/simulationScenes/clover2.ttt' #'/home/majinbuu/PycharmProjects/cloverski/customFiles/environments/clover2.ttt'
    """
    Training variables 
    """
    resetNum = 0
    numAttempts = 3  # number of times allowed to fail before getting new goals
    numGoals = 2
    maxTime = 500  # 210
    """
    Normalization Constants
    """
    posFactor = 10
    angFactor = math.pi
    forceFactor = 1
    rewardsMax = 0

    """
    Robot Settings
    """
    obsDim = 9  # 5 for state, 4 for goals
    actionDim = 2  # only 2 motors
    actionMultiplier = 2  # 1 = 0.1 m/s v--> CLOVER has a max of 0.4m/s (act mult of 4)
    speedFactor = actionMultiplier
    maxDeltaV = ( 2 * actionMultiplier * simSettings.dt) / 10  # min time it would take to go from max velocity to - max velocity or vice versa
    radius = 0.11
    velFactor = 1 / 10  # radius * actionMultiplier

    """
    Reward function variables
    """
    aliveBonus = 1
    velocityRewardConstant = 500
    torquePenaltyConstant = 0
    orientationPenaltyConstant = 0
    aliveConstant = 0.50
    backupPenaltyConstant = 0
    fallConstant = 100
    turnPenaltyConstant = 0  # 1
    turnPenaltyMax = -250 * turnPenaltyConstant
    backupPenaltyMax = -5000



class constTaskPPMCEval():
    """
    Holds all the constant for the robot tasks
    """
    """
    Booleans
    """
    presetWaypoints = True # static goals, only used for training

    """
    TRAINING OR PLAY
    """
    if presetWaypoints:
        numGoals = 16 #2
        maxTime = 999999  # 500  # 210
    else:
        numGoals = 2
        maxTime = 500  # 210

    """  
    Scene location  
    """
    sceneLocation = '/home/lunar3/Desktop/PythonProjects/star/simulationScenes/clover2.ttt' #'/home/majinbuu/PycharmProjects/cloverski/customFiles/environments/clover2.ttt'
    """
    Training variables 
    """
    resetNum = 0
    numAttempts = 3  # number of times allowed to fail before getting new goals

    """
    Normalization Constants
    """
    posFactor = 10
    angFactor = math.pi
    forceFactor = 1
    rewardsMax = 0

    """
    Robot Settings
    """
    obsDim = 9  # 5 for state, 4 for goals
    actionDim = 2  # only 2 motors
    actionMultiplier = 2  # 1 = 0.1 m/s v--> CLOVER has a max of 0.4m/s (act mult of 4)
    speedFactor = actionMultiplier
    maxDeltaV = ( 2 * actionMultiplier * simSettings.dt) / 10  # min time it would take to go from max velocity to - max velocity or vice versa
    radius = 0.11
    velFactor = 1 / 10  # radius * actionMultiplier

    """
    Reward function variables
    """
    aliveBonus = 1
    velocityRewardConstant = 500
    torquePenaltyConstant = 0
    orientationPenaltyConstant = 0
    aliveConstant = 0.50
    backupPenaltyConstant = 0
    fallConstant = 100
    turnPenaltyConstant = 0#1
    turnPenaltyMax = -250 * turnPenaltyConstant
    backupPenaltyMax = -5000



