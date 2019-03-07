import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import rospy

from affordance_gym.simulation_interface import SimulationInterface
from affordance_gym.perception_policy import Predictor
from affordance_gym.utils import parse_arguments, load_parameters,  use_cuda

from affordance_gym.monitor import TrajectoryEnv

from env_setup.env_setup import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON,  VAED_MODELS_PATH, TRAJ_MODELS_PATH, POLICY_MODELS_PATH
from env_setup.env_setup import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM

from TrajectoryVAE.ros_monitor import ROSTrajectoryVAE
from AffordanceVAED.ros_monitor import RosPerceptionVAE


'''

Evaluates the performance of the affordance policy in MuJoCo

'''


def main(args):

    rospy.init_node('talker', anonymous=True)

    device = use_cuda()

    # Trajectory generator
    assert(args.model_index > -1)

    bahavior_model_path = os.path.join(TRAJ_MODELS_PATH, args.vae_name)
    action_vae = ROSTrajectoryVAE(bahavior_model_path, args.latent_dim, args.num_actions,
                                  model_index=args.model_index, num_joints=args.num_joints)
    # pereception
    gibson_model_path = os.path.join(VAED_MODELS_PATH, args.g_name)
    perception = RosPerceptionVAE(gibson_model_path, args.g_latent)

    # Policy

    if (args.fixed_camera):
        policy = Predictor(args.g_latent, args.latent_dim)
    else:
        # Includes camera params as an input
        policy = Predictor(args.g_latent + 3, args.latent_dim)

    policy.to(device)

    policy_path = os.path.join(POLICY_MODELS_PATH, args.policy_name)
    load_parameters(policy, policy_path, 'model')

    # Simulation interface
    sim = SimulationInterface(arm_name='lumi_arm')
    sim.change_camere_params(LOOK_AT, DISTANCE, AZIMUTH, ELEVATION)
    env = TrajectoryEnv(action_vae, sim, args.num_actions, num_joints=args.num_joints)

    camera_distance = DISTANCE
    azimuth = AZIMUTH
    elevation = ELEVATION
    sim.change_camere_params(LOOK_AT, camera_distance, azimuth, elevation)

    cup_name = 'cup{}'.format(args.cup_id)

    rewards = []
    for idx in range(10):

        x = np.random.uniform(CUP_X_LIM[0], CUP_X_LIM[1])
        y = np.random.uniform(CUP_Y_LIM[0], CUP_Y_LIM[1])

        if (args.randomize_all):

            camera_distance = DISTANCE + np.random.uniform(-DISTANCE_EPSILON, DISTANCE_EPSILON)
            azimuth = AZIMUTH + np.random.uniform(-AZIMUTH_EPSILON, AZIMUTH_EPSILON)
            elevation = ELEVATION + np.random.uniform(-ELEVATION_EPSILON, ELEVATION_EPSILON)

            sim.change_camere_params(LOOK_AT, camera_distance, azimuth, elevation)
            cup_name = 'cup{}'.format(np.random.random_integers(1, 10))

        sim.reset_table(x, y, 0, cup_name)

        image_arr = sim.capture_image()
        image = Image.fromarray(image_arr)

        # Image -> Latent1
        latent1 = perception.get_latent(image)

        if not(args.fixed_camera):
            n_camera_distance = (camera_distance - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
            n_elevation = (elevation - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
            n_azimuth = (azimuth - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)

            camera_params = Variable(torch.Tensor([n_camera_distance, n_azimuth, n_elevation]).to(device))
            camera_params = camera_params.unsqueeze(0)
            latent1 = torch.cat([latent1, camera_params], 1)

        # latent and camera params -> latent2
        latent2 = policy(latent1)
        latent2 = latent2.detach().cpu().numpy()

        # Latent2 -> trajectory (mujoco)
        _, end_pose = env.do_latent_imitation(latent2[0])
        end_pose = np.array((
            end_pose.pose.position.x,
            end_pose.pose.position.y))

        reward = np.linalg.norm(np.array([x, y]) - end_pose)
        rewards.append(reward)

        print('Reward: {}'.format(reward))
        print("goal", x, y)
        print("end_pose", end_pose)

    print("AVG: ", np.mean(rewards), " VAR: ", np.var(rewards))


if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, gibson=True, policy=True, policy_eval=True)
    main(args)
