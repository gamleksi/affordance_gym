import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from TrajectoryVAE.trajectory_vae import TrajectoryVAE, load_parameters
from TrajectoryVAE.utils import MIN_ANGLE, MAX_ANGLE

from AffordanceVAED.ros_monitor import RosPerceptionVAE

from affordance_gym.perception_policy import Predictor, end_effector_pose

from affordance_gym.utils import parse_vaed_arguments, parse_traj_arguments, parse_policy_arguments, load_parameters, use_cuda
from env_setup.env_setup import VAED_MODELS_PATH, TRAJ_MODELS_PATH, POLICY_MODELS_PATH, LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, LOOK_AT_EPSILON, KINECT_EXPERIMENTS_PATH
from env_setup.env_setup import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON


'''
This runs already collected experiment samples (args.log_name) to find good cropping values (args.top_crop, args.width_crop)
and to experiment samples with different policy models. (this does not need ros) 
'''


def main(args):

    device = use_cuda()

    assert(args.model_index > -1)

    # Load trajectory VAE
    trajectory_vae = TrajectoryVAE(args.traj_latent, args.num_actions, args.num_joints, device)
    trajectory_vae.to(device)
    trajectory_model_path = os.path.join(TRAJ_MODELS_PATH, args.traj_name)
    load_parameters(trajectory_vae, trajectory_model_path, args.model_index)
    traj_decoder = trajectory_vae.decoder.to(device)

    # Load VAED
    vaed_path = os.path.join(VAED_MODELS_PATH, args.vaed_name)
    perception = RosPerceptionVAE(vaed_path, args.vaed_latent)

    # Policy
    policy = Predictor(args.vaed_latent + 5, args.traj_latent, args.num_params)
    policy.to(device)
    policy_path = os.path.join(POLICY_MODELS_PATH, args.policy_name)
    load_parameters(policy, policy_path, 'model')

    # Kinect data
    log_path = os.path.join(KINECT_EXPERIMENTS_PATH, args.log_name)

    # Getting object poses
    data = pd.read_csv(os.path.join(log_path, 'cup_log.csv'))
    data = data.values
    cup_poses = np.array(data[:, 1:3], dtype=np.float32)
    end_effector_poses = np.array(data[:, 3:5], dtype=np.float32)
    kinect_lookats = np.array(data[:, 5:7], dtype=np.float32)
    kinect_distances = np.array(data[:, 7])
    kinect_azimuths = np.array(data[:, 8])
    kinect_elevations = np.array(data[:, 9])

    # Camera param normalization
    n_lookat_xs = (kinect_lookats[:, 0] - (np.array(LOOK_AT[0]) - LOOK_AT_EPSILON)) / (LOOK_AT_EPSILON * 2)
    n_lookat_ys = (kinect_lookats[:, 1] - (np.array(LOOK_AT[1]) - LOOK_AT_EPSILON)) / (LOOK_AT_EPSILON * 2)
    n_camera_distances = (kinect_distances - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
    n_azimuths = (kinect_azimuths - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)
    n_elevations = (kinect_elevations - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)

    camera_params = np.array([n_lookat_xs, n_lookat_ys, n_camera_distances, n_azimuths, n_elevations], np.float)

    debug_images = np.array(data[:, 13], str)

    end_poses = []
    distances = []
    sim_real_errors = []

    for i in range(camera_params.shape[1]):

        image_path = os.path.join(os.path.join(log_path, "inputs"), debug_images[i])

        image = Image.open(image_path)

        width, height = image.size
        left = 0
        top = args.top_crop
        right = width - args.width_crop
        bottom = height
        image = image.crop((left, top, right, bottom))

        # Image -> Latent1
        latent1 = perception.get_latent(image)

        camera_input = Variable(torch.Tensor(camera_params[:, i]).to(device))
        camera_input = camera_input.unsqueeze(0)
        latent1 = torch.cat([latent1, camera_input], 1)

        # latent and camera params -> latent2
        latent2 = policy(latent1)
        trajectories = traj_decoder(latent2)

        # Reshape to trajectories
        trajectories = trajectory_vae.to_trajectory(trajectories)

        end_joint_pose = trajectories[:, :, -1]

        end_joint_pose = (MAX_ANGLE - MIN_ANGLE) * end_joint_pose + MIN_ANGLE
        # joint pose -> cartesian
        end_pose = end_effector_pose(end_joint_pose, device)
        end_pose = end_pose.cpu().detach().numpy()[0]

        end_poses.append(end_pose)
        distance = np.linalg.norm(end_pose - cup_poses[i])
        distances.append(distance)
        sim_real_error = np.linalg.norm(end_pose - end_effector_poses[i])
        sim_real_errors.append(sim_real_error)

    end_poses = np.array(end_poses)
    save_path = os.path.join(log_path, args.policy_name)

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    f = open(os.path.join(save_path, 'avg_errors_t_{}_w_{}_crops.txt'.format(args.top_crop, args.width_crop)), 'w')
    f.write("avg goal distance {}\n".format(np.mean(distances)))

    print("avg goal distance", np.mean(distances))

    if args.real_hw:
        print("avg real sim distance", np.mean(sim_real_errors))
        f.write("avg real sim distance {}\n".format(np.mean(sim_real_errors)))

    fig, axes = plt.subplots(3, 3, sharex=True, figsize=[30, 30])

    cup_names = np.unique(np.array(data[:, 0], str))

    for i, cup_name in enumerate(cup_names):

        ax = axes[int(i/3)][i%3]
        cup_indices = np.array([cup_name in s for s in data[:, 0]])

        goal_poses = cup_poses[cup_indices]
        pred_poses = end_poses[cup_indices]

        print(cup_name, 'avg goal error:', np.linalg.norm(goal_poses - pred_poses, axis=1).mean())
        f.write("{} avg goal error {}\n".format(cup_name, np.linalg.norm(goal_poses - pred_poses, axis=1).mean()))

        ax.scatter(goal_poses[:, 0], goal_poses[:, 1], c='r', label='real')
        ax.scatter(pred_poses[:, 0], pred_poses[:, 1], c='b', label='pred')

        if args.real_hw:
            hw_poses = end_effector_poses[cup_indices]
            print(cup_name, 'avg hw real error:', np.linalg.norm(hw_poses - pred_poses, axis=1).mean())
            f.write("{} avg hw real error {}\n".format(cup_name, np.linalg.norm(hw_poses - pred_poses, axis=1).mean()))
            ax.scatter(hw_poses[:, 0], hw_poses[:, 1], c='g', label='hw')

        ax.set_title(cup_name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

    plt.savefig(os.path.join(save_path, 'simulation_result_t_{}_w_{}_crops.png'.format(args.top_crop, args.width_crop)))

    if args.real_hw:
        np.save(os.path.join(save_path, 'results.npy'), (cup_poses, end_poses, hw_poses))
    else:
        np.save(os.path.join(save_path, 'results.npy'), (cup_poses, end_poses))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Experiment gathered Kinect images with a different setup (policy, trajectory vae, vaed, image cropping)')

    parse_vaed_arguments(parser)
    parse_traj_arguments(parser)
    parse_policy_arguments(parser)

    parser.add_argument('--real-hw', dest='real_hw', action='store_true', help='Compare performance with results that were obtained while gathering the experiments' )
    parser.set_defaults(real_hw=False)
    parser.add_argument('--log-name', default='kinect_example', type=str, help='Kinect experiment folder')
    parser.add_argument('--top-crop', default=64, type=int)
    parser.add_argument('--width-crop', default=0, type=int)

    args = parser.parse_args()

    main(args)
