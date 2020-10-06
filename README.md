# RL STaR ![Clover Logo](./media/cloverlogotiny.png)

This README is intended to give guidance to users about how to get started using or contributing to this open platform.

## Welcome
Welcome to the Reinforcement Learning for Simulation based Training of Robots (RL STaR) Platform github page.

RL STaR was created for robotics researchers to advance research using reinforcement learning and machine learning, and to share this work with other researchers.

We created RL STaR in a modular way to be easy to use and easy to share RL applications for robotics. Training in simulation offers a safer and more time efficient environment, with the prospect of later transfer results to the real robot through simulation to reality transfer (sim2real).

We welcome all contributions, suggestions and feedback. Thank you for your interest.
## How to Use RL STaR


There are three main components to the RL STaR platform: the RL library, the simulator and the Task Trainer. You should download CopelliaSim from their website which is the simulator this platform is compatible with. This platform ships with a frozen clone of [OpenAI Baselines](https://github.com/openai/baselines), a reinforcement learning library holding a number of modern RL algorithms. This library should be swappable for a different library if you prefer to do so. The last componenet is the task trainer, which is what we created to interface between the prior two components and to train the robot for various tasks.


![RL](./media/rlblocks.png)



### Files of RL STaR
1. Constants
2. Functions
3. Main File
4. InOut
5. Simulator
6. Task

The **Constants file** is broken down into different classes that encapsulates different information. This file includes information about the robot, about the simulation conditions, about file locations and about the task, including reward function parameter values.

The **Functions file** includes different functions that can be used across different tasks.

The **Main file** groups everything together and serves as a central hub.

The **InOut file** acts as an itermediary between the task and the simulator file.

The **Simulator file** sends information to and from CoppeliaSim.

The **Task file** is where the user can define what he wants the robot to learn.

### Other Files
Aside from the files mentioned above several other files exist. Of particular importance are the CoppeliaSim "scene" files, which are what holds the environment and robot. These are placed within .../RLSTaR/simulationScenes/ and are denoted by ".ttt".

### CLOVER
The Space Robotics Lab is utilizing the CLOVER rover for lunar surface swarm and AI applications. The system is a small, lightweight robot that mainly utilizes off the shelf components, making it easy to build several of these rovers for research applications. CLOVER is a skid steer rear wheel drive vehicle. This means that there are only two motors (one for each side), and that it turns by creating a differential in wheel speeds between the left and right side.
![CLOVER](./media/clover.jpg)

### Adding New Robots and Tasks
RL STaR was designed to be modular. The OpenAI Gym framework makes it easy to share reinforcement learning problems with others, and this platform was intended to bridge the divide between machine learning and robotics applications.
We envision new applications being divided into two categories:
1. Robotic applications
2. Task applications

A **new robotic application** is when someone applies the RL STaR platform to a new robot. This could adding a bigger rover, for example.
A **new task application** is when someone applies the RL STaR platform to a new task. This could be training the robot to navigate a maze, for example.

When creating a new robot application, one should create a new main file and a new constants file, making sure to register the name of the new environment (robot) in the *.../customfiles/enviornment/\__init\__.py* and the *.../customfiles/\__init\__.py* files.

In order to add a new task, new parameters can added inside the exisiting robot constants file as a new class. This new class should be imported into the main file as *const*. A new task file should also be added, calculating the reward function, observations, etc as you see fit. Make sure that the robot action space is set appropraitely, whether [0,1], [-1,1] or some other range (including discrete). The new task should also be imported into the main file as taskX


### Linking to a new robot
Names of the robot components in CoppeliaSim and in the RL STaR constants.py file should be identical. Joints in CoppeliaSim are used as motors, and are set  to rotate a certain speed. The joints of the robot are defined in jointNames. The base of the robot is defined in baseName, and is used to get information about the robot, such as its position and the yaw angle from the simulator.

    wheelNames = ["left_front", "right_front", "left_rear", "right_rear"]
    numWheels = 4
    baseName = "base_link_respondable"
    jointNames = ["joint_left_front_wheel", "joint_right_front_wheel", "joint_left_rear_wheel",
                  "joint_right_rear_wheel"]

### Training
During the training process, you can run multiple sessions of CoppeliaSim and the training code at the same time. This is useful for tuning different parameters such as the neural network or the reward function.


Several useful commands:

The following command starts the training process for the "clover-1" environment for 1e5 timesteps. The RL algorithm chosen was PPO. The network is a custom network defined in *.../RLSTaR/baselines/baselines/common/models.py*. Several of the terms provide PPO2 paremers (gamma, nstep); more information can be found by going to the OpenAI Baselines website or by chekcing the code in *...RLSTaR/baselines/baselines/ppo2/ppo2.py*.

    OPENAI_LOGDIR=~/Desktop/Datafolder/trial01 OPENAI_LOG_FORMAT=tensorboard python3 -m baselines.run --num_timesteps=1e5 --network=mlp3 --nsteps=256 --alg=ppo2 --env=clover-v1 --gamma=0.90

The following command loads a previously trained agent and "plays" it. At this point it doesn't train anymore. The network should be identical to the one used during training.

    python3 -m baselines.run --alg=ppo2 --env=clover-v1 --num_timesteps=99999 --network=mlp3 --load_path=~/Desktop/Data/star/ppmc/test/noturn/trial01/checkpoints/XXXXX --play

This command launches a tensorboard session, allowing you to track the progress of an already trained/training session, with information about the average reward, length, etc.


    tensorboard --logdir firstTrial:~/Desktop/Datafolder/trial01,secondTrial:~/Desktop/Datafolder/trial02 --port=9999


## How To Configure RL STaR
Download [CoppeliaSim](https://www.coppeliarobotics.com/) from their website.

Follow the baseline instructions:

    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

Install Tensorflow (see below for more info)

    pip install tensorflow==1.14

or

    pip install tensorflow-gpu==1.14

Inside the RLSTaR folder, run the following command to install the necessary packages for RL STaR:

    >>>.../rlstar$ pip3 install -e .

Inside the RLSTaR/baselines folder, run the following command to install the necessary packages for Baselines:

    >>>.../rlstar/baselines$ pip3 install -e .

### Installing Cuda (for Tensorflow-gpu):
Having a GPU allows for faster simulations. You must have an NVIDIA GPU compatible with the CUDA version you would like to run.

I followed the following instructions:
https://medium.com/analytics-vidhya/installing-tensorflow-with-cuda-cudnn-gpu-support-on-ubuntu-20-04-f6f67745750a

### Compatability
**Ubuntu:**
 * 18.04
 * 20.04

**CoppeliaSim:**
* 4_1_0_Ubuntu18_04
>Probably works regardless of version unless big changes will be made in the future

**Python:**

*   3.6
>3.5 should work too (Baseline states Python >=3.5 should work. Python 3.7 and above are not compatible with tensorflow 1.4, so be careful.)

**Tensorflow:**
* 1.4-GPU
>While tensorflow 2.0 is out now, 1.4 is recommended if using the OpenAI Baselines library
>
>Non GPU should also work
>
>When installing tensorflow, you must specify if you want to install the gpu or non gpu version

**Cuda:**
* 10.0
* 11

**Gym:**
* 0.15.3 (recommended)
>(Tried on some prior versions, such as 13.~, If this is the case, some naming differences in the baselines run file exist, i.e. changing the name _entry_point --> entry_point )



## Authors
[![srl-logo-original.jpg](./media/srl-logo-original.jpg)](http://www.astro.mech.tohoku.ac.jp/e/index.html)

The authors of this paper all belong the the Space Robotics Lab, Department of Aerospace Engeering, Tohoku University:

>Tamir Blum, Gabin Paillet, Professor Mickael Laine, Professor Yoshida Kazuya

## Contributing
We would like to invite any users to share their own robots or tasks

##Citation
This platform can be cited via

 1. [RL STaR](https://arxiv.org/abs/2009.09595) 

The path planning and motion control application can be cited via:
1. PPMC Training Algorithm: A Deep Learning BasedPath Planner and Motion Controller: [IEEE Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9065237) or
[ARXIV Link](https://arxiv.org/abs/1909.06034)
2. PPMC RL Training Algorithm: Rough Terrain Intelligent Robots through Reinforcement Learning [ARXIV Link](https://arxiv.org/abs/2003.02655)


## Support
Please email TamirBlum1{at}gmail.com or open an issues ticket

## License
[MIT](https://choosealicense.com/licenses/mit/)