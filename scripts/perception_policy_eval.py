import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable

from motion_planning.simulation_interface import SimulationInterface
from motion_planning.perception_policy import Predictor
from behavioural_vae.ros_monitor import ROSTrajectoryVAE
from gibson.ros_monitor import RosPerceptionVAE

from motion_planning.utils import parse_arguments, GIBSON_ROOT, load_parameters, BEHAVIOUR_ROOT, POLICY_ROOT, use_cuda
from motion_planning.utils import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM
from motion_planning.utils import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON
from motion_planning.monitor import TrajectoryEnv


def main(args):

    device = use_cuda()

    # Trajectory generator
    # TODO Currently includes both encoder and decoder to GPU even though only encoder is used.

    assert(args.model_index > -1)

    bahavior_model_path = os.path.join(BEHAVIOUR_ROOT, args.vae_name)
    action_vae = ROSTrajectoryVAE(bahavior_model_path, args.latent_dim, args.num_actions,
                                  model_index=args.model_index, num_joints=args.num_joints)
    # pereception
    # TODO Currently includes both encoder and decoder to GPU even though only encoder is used.

    gibson_model_path = os.path.join(GIBSON_ROOT, args.g_name)
    perception = RosPerceptionVAE(gibson_model_path, args.g_latent)

    # Policy

    if (args.fixed_camera):
        policy = Predictor(args.g_latent, args.latent_dim)
    else:
        # Includes camera params as an input
        policy = Predictor(args.g_latent + 3, args.latent_dim)

    policy.to(device)

    policy_path = os.path.join(POLICY_ROOT, args.policy_name)
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

        # affordance, sample = perception.reconstruct(image) TODO sample visualize

        # Image -> Latent1
        latent1 = perception.get_latent(image)

        if not(args.fixed_camera):
            n_camera_distance = (camera_distance - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
            n_elevation = (elevation - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
            n_azimuth = (azimuth - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)

            camera_params = Variable(torch.Tensor([n_camera_distance, n_azimuth, n_elevation]).to(device))
            camera_params = camera_params.unsqueeze(0)
            latent1 = torch.cat([latent1, camera_params], 1)

            # TODO Combine latent1 with normalized camera params

        # latent and camera params -> latent2
        latent2 = policy(latent1)
        latent2 = latent2.detach().cpu().numpy() # TODO fix this!

        # Latent2 -> trajectory (mujoco)
        _, end_pose = env.do_latent_imitation(latent2[0])
        end_pose = np.array((
            end_pose.pose.position.x,
            end_pose.pose.position.y))

        reward = np.linalg.norm(np.array([x, y]) - end_pose)
        rewards.append(reward)
        # env.reset_environment(duration=3.0)

        print('Reward: {}'.format(reward))
        print("goal", x, y)
        print("end_pose", end_pose)

    print("AVG: ", np.mean(rewards), " VAR: ", np.var(rewards))


if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, gibson=True, policy=True, policy_eval=True)
    main(args)
