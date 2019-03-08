import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import rospy

from affordance_gym.simulation_interface import SimulationInterface
from affordance_gym.perception_policy import Predictor, end_effector_pose
from affordance_gym.utils import parse_policy_arguments, parse_moveit_arguments, parse_vaed_arguments, parse_traj_arguments, load_parameters,  use_cuda

from affordance_gym.monitor import TrajectoryEnv

from env_setup.env_setup import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON,  VAED_MODELS_PATH, TRAJ_MODELS_PATH, POLICY_MODELS_PATH
from env_setup.env_setup import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM, LOOK_AT_EPSILON

from TrajectoryVAE.ros_monitor import ROSTrajectoryVAE
from TrajectoryVAE.utils import MAX_ANGLE, MIN_ANGLE
from AffordanceVAED.ros_monitor import RosPerceptionVAE


'''

Evaluates the performance of the affordance policy in MuJoCo

'''


def main(args):

    rospy.init_node('policy_train', anonymous=True)

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
        policy = Predictor(args.g_latent + 5, args.latent_dim, args.num_params)

    policy.to(device)

    policy_path = os.path.join(POLICY_MODELS_PATH, args.policy_name)
    load_parameters(policy, policy_path, 'model')

    # Simulation interface
    sim = SimulationInterface(arm_name='lumi_arm')
    sim.change_camere_params(LOOK_AT, DISTANCE, AZIMUTH, ELEVATION)
    env = TrajectoryEnv(action_vae, sim, args.num_actions, num_joints=args.num_joints, trajectory_duration=args.duration)

    # Reset time

    if args.forward_kinematics:
        reset_duration = 1.5
    else:
        reset_duration = 2.5

    losses = []
    for idx in range(100):
        cup_name = 'cup{}'.format(np.random.randint(1, 10))
        x = np.random.uniform(CUP_X_LIM[0], CUP_X_LIM[1])
        y = np.random.uniform(CUP_Y_LIM[0], CUP_Y_LIM[1])

        lookat = np.array(LOOK_AT)
        camera_distance = DISTANCE
        azimuth = AZIMUTH
        elevation = ELEVATION

        if (args.randomize_all):

            lookat[0] += np.random.uniform(-LOOK_AT_EPSILON, LOOK_AT_EPSILON)
            lookat[1] += np.random.uniform(-LOOK_AT_EPSILON, LOOK_AT_EPSILON)

            camera_distance += np.random.uniform(-DISTANCE_EPSILON, DISTANCE_EPSILON)
            azimuth += np.random.uniform(-AZIMUTH_EPSILON, AZIMUTH_EPSILON)
            elevation += np.random.uniform(-ELEVATION_EPSILON, ELEVATION_EPSILON)

        sim.change_camere_params(lookat, camera_distance, azimuth, elevation)
        sim.reset_table(x, y, 0, cup_name, duration=reset_duration)

        image_arr = sim.capture_image('/lumi_mujoco/rgb')
        image = Image.fromarray(image_arr)

        # Image -> Latent1
        latent1 = perception.get_latent(image)

        if not(args.fixed_camera):

            n_lookat = (lookat[:2] - (np.array(LOOK_AT[:2]) - LOOK_AT_EPSILON)) / (LOOK_AT_EPSILON * 2)
            n_camera_distance = (camera_distance - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
            n_elevation = (elevation - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
            n_azimuth = (azimuth - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)

            camera_params = Variable(torch.Tensor([n_lookat[0], n_lookat[1], n_camera_distance, n_azimuth, n_elevation]).to(device))
            camera_params = camera_params.unsqueeze(0)
            latent1 = torch.cat([latent1, camera_params], 1)

        # latent and camera params -> latent2
        latent2 = policy(latent1)

        if (args.forward_kinematics):

            # latent2 -> trajectory
            trajectories = action_vae.model.decoder(latent2)

            # Reshape to trajectories
            trajectories = action_vae.model.to_trajectory(trajectories)

            # Get the last joint pose
            end_joint_pose = trajectories[:, :, -1]

            # Unnormalize
            end_joint_pose = (MAX_ANGLE - MIN_ANGLE) * end_joint_pose + MIN_ANGLE

            # joint pose -> cartesian
            end_pose = end_effector_pose(end_joint_pose, device)

            end_pose = end_pose.detach().cpu().numpy()
            target_pose = np.array([x, y])

            loss = np.linalg.norm(end_pose[0] - target_pose)

        else:

            latent2 = latent2.detach().cpu().numpy()
            _, end_pose = env.do_latent_imitation(latent2[0])

            loss = np.linalg.norm(np.array([x, y]) - end_pose[:2])

        losses.append(loss)
        print('loss: {}'.format(loss))
        print("goal", x, y)
        print("end_pose", end_pose)

    print("AVG: ", np.mean(losses), " VAR: ", np.var(losses))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a perception policy in MuJoCo')

    parse_policy_arguments(parser)
    parse_moveit_arguments(parser)
    parse_vaed_arguments(parser)
    parse_traj_arguments(parser)

    parser.add_argument('--forward-kinematics', dest='forward_kinematics', action='store_true', help="Compute an end effector position by solving forward kinematics")
    parser.set_defaults(forward_kinematics=False)

    parser.add_argument('--randomize-all', dest='randomize_all', action='store_true')
    parser.set_defaults(randomize_all=False)

    args = parser.parse_args()

    main(args)
