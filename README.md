# [Affordance Learning for End-to-End Visuomotor Robot Control](TODO)

In [Affordance Learning for End-to-End Visuomotor Robot Control](TODO), we introduced a modular deep neural network structure, 
that detects a container on a table, and inserts a ball into it.

We showed that our system performs its task successfully in zero-shot sim-to-real transfer manner.
Each part of our system was fully trained either with synthetic data or in a simulation.
The system was invariant to, e.g., distractor objects and textures.

The system layout:

![the VAED structure](images/trajvae.png?raw=true)

We have didvided our work into the following code blocks:

* [AffordanceVAED](https://github.com/gamleksi/AffordanceVAED) extracts affordance information from an observation image, and represents it as a latent space vector. 
Figure 1 shows the structure of the model.
* [BlenderDomainRandomizer](https://github.com/gamleksi/BlenderDomainRandomizer) generates  a domain randomized dataset for VAED.
* [TrajectoryVAE](https://github.com/gamleksi/TrajectoryVAE) represents trajectories in a low-dimensional latent space, and generates a trajectory based on a given latent vector.
* [affordance_gym](https://github.com/gamleksi/affordance_gym) generates training data for TrajectoryVAE, and combines VAED and TrajectoryVAE together to perform desired trajectories based on an observation.

## Training Flow of the System

1) Generate a domain randomized dataset for affordance detection, train a VAED model, and update ```VAED_MODELS_PATH``` in src/env_setup/env_setup.py to match to the model's parent folder path.
2) Generate trajectory training data run ```python scripts/generate_perception_data.py```, train a TrajectoryVAE model, and update TRAJ_MODELS_PATH (in src/env_setup/env_setup.py) to match to the model's parent folder path.
3) Update ```POLICY_MODELS_PATH``` (in src/env_setup/env_setup.py) to match to the model's parent folder path and run ```python scripts/generate_perception_data.py``` to generate training policy data.
4) To train a policy model run ```python scripts/perception_policy_train.py```, and, to evaluate the policy model's performance in MuJoCo run ``` python scripts/perception_policy_train```.py.
5) Update ```KINECT_EXPERIMENTS_PATH``` (in src/env_setup/env_setup.py) and run ```python scripts/kinect_env.py``` to experiment the learned model with a real camera and with or without a real robot.

More info about each phase run ``` python scripts/file_name.py -h ```.

## Setup

Install required depedencies To install ```pip install -r requirements.txt```.

## Run

1) Generate training data with [affordance_gym](https://github.com/gamleksi/affordance_gym) (scripts/generate_trajectories.py).
3) Run ```python main.py -h``` to see how to include the generated training data and explore rest of the running options.
