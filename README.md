# Affordance Learning for End-to-End Visuomotor Robot Control

In [Affordance Learning for End-to-End Visuomotor Robot Control](TODO), we introduced a modular deep neural network structure, 
that detects a container on a table, and inserts a ball into it.

We showed that our system performs its task successfully in zero-shot sim-to-real transfer manner.
Each part of our system was fully trained either with synthetic data or in a simulation.
The system was invariant to, e.g., distractor objects and textures.

The system structure:

![the structure](images/structure.png?raw=true)

We have didvided our work into the following code blocks:

* [AffordanceVAED](https://github.com/gamleksi/AffordanceVAED) extracts affordance information from an observation image, and represents it as a latent space vector. 
Figure 1 shows the structure of the model.
* [BlenderDomainRandomizer](https://github.com/gamleksi/BlenderDomainRandomizer) generates  a domain randomized dataset for VAED.
* [TrajectoryVAE](https://github.com/gamleksi/TrajectoryVAE) represents trajectories in a low-dimensional latent space, and generates a trajectory based on a given latent vector.
* [affordance_gym](https://github.com/gamleksi/affordance_gym) generates training data for TrajectoryVAE, and combines VAED and TrajectoryVAE together to perform desired trajectories based on an observation.

## Setup

Ros Kinetic, MoveIt!, and MuJoco (2.0) should be installed. 
[Colcon](https://colcon.readthedocs.io/en/released/) was used to build a workspace.

Prerequisites:
```sh
pip install -r affordance_gym/requirements.txt
sudo apt install ros-kinetic-libfranka ros-kinetic-franka-ros
... TODO
```

The work environemnt depends also on the 

Workspace creation:
```sh
mkdir -p ~/ros/src
cd ~/ros/src
git --recursive clone git@github.com:gamleksi/affordance_gym.git
cd ~/ros
colcon build
```

The folder structure:
* Lumi Testbed by [Intelligent Robotics group](http://irobotics.aalto.fi): The set of core ROS packages for Lumi, our robot. Contains URDF description, moveit configuration, mujoco configuration.
* Mujoco Ros Control by [Intelligent Robotics group](http://irobotics.aalto.fi): Interface for the MuJoCo simulator.
* Lumi Pose Estimation: This is required in kinect_simulation.launch, as it computes the camera pose of Kinect with a Aruco marker.
* Affordance Gym: Affordance Learning for End-to-End Visuomotor Robot Control

## Run (in the affordance_gym folder)

### Training Flow of the System

1) Generate a domain randomized dataset for affordance detection, train a VAED model, and update ```VAED_MODELS_PATH``` in src/env_setup/env_setup.py to match to the model's parent folder path.
2) Generate trajectory training data run ```python scripts/generate_perception_data.py```, train a TrajectoryVAE model, and update TRAJ_MODELS_PATH (in src/env_setup/env_setup.py) to match to the model's parent folder path.
3) Update ```POLICY_MODELS_PATH``` (in src/env_setup/env_setup.py) to match to the model's parent folder path and run ```python scripts/generate_perception_data.py``` to generate training policy data.
4) To train a policy model run ```python scripts/perception_policy_train.py```, and, to evaluate the policy model's performance in MuJoCo run ``` python scripts/perception_policy_train```.py.
5) Update ```KINECT_EXPERIMENTS_PATH``` (in src/env_setup/env_setup.py) and run ```python scripts/kinect_env.py``` to experiment the learned model with a real camera and with or without a real robot.

More info about each phase run ``` python scripts/file_name.py -h ```.

### Scripts and Roslaunch

The table shows which launch file is required to be running with each python file.

|Script|Simulation|
|---|---|
|generate_perception_data.py|roslaunch lumi_mujoco table_simulation.launch|
|generate_trajectories.py|roslaunch lumi_mujoco table_simulation.launch|
|kinect_debug.py|-|
|kinect_env.py|roslaunch lumi_mujoco kinect_simulation.launch (without a real robot arm)|
|kinect_test.py|roslaunch lumi_mujoco kinect_simulation.launch|
|mc_interface.py|(real robot launch file)|
|perception_policy_eval.py|roslaunch lumi_mujoco table_simulation.launch|
|perception_policy_train.py|-|
