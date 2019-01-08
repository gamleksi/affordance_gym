#! /usr/bin/env python
from behavioural_vae.ros_monitor import ROSTrajectoryVAE, RosTrajectoryConvVAE
from behavioural_vae.visual import TrajectoryVisualizer
from behavioural_vae.utils import smooth_trajectory, MIN_ANGLE, MAX_ANGLE
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.monitor import TrajectoryEnv
from motion_planning.utils import parse_arguments, BEHAVIOUR_ROOT
import numpy as np
import os
import pickle
from tqdm import tqdm


def main(args):

    model_name = args.vae_name
    latent_dim = args.latent_dim
    duration = args.duration
    num_actions = args.num_actions
    num_joints = args.num_joints

    models_folder = os.path.join(BEHAVIOUR_ROOT, 'log', model_name, 'models')

    assert(os.path.exists(models_folder))

    listdir = os.listdir(models_folder)

    num_models = len(listdir)

    simulation_interface = SimulationInterface('lumi_arm')

    save_path = os.path.join(BEHAVIOUR_ROOT, 'log', model_name, 'reconstruction_results')

    if (not(os.path.exists(save_path))):
        os.makedirs(save_path)

    trajectories, end_poses = load_data(args.dataset_name, args.num_joints, args.num_actions)

    for model_idx in range(num_models):

        if args.conv:
            behaviour_model = RosTrajectoryConvVAE(model_name, latent_dim, num_actions, args.kernel_row, args.conv_channel)
        else:

            behaviour_model = ROSTrajectoryVAE(model_name, latent_dim, num_actions, model_index=model_idx, num_joints=num_joints,  root_path=BEHAVIOUR_ROOT)

        env = TrajectoryEnv(behaviour_model, simulation_interface, num_actions, num_joints=num_joints, trajectory_duration=duration)
        env.reset_environment(3.0)

        pose_results = []
        latents = []
        recons = []

        num_trajectories = trajectories.shape[0]

        for idx in tqdm(range(num_trajectories)):

            trajectory = trajectories[idx]

            end_pose, recon, latent = env.imitate_trajectory(trajectory)
            env.reset_environment(1.0)
            end_pose = (end_pose.pose.position.x, end_pose.pose.position.y, end_pose.pose.position.z)

            pose_results.append(end_pose)
            latents.append(latent)
            recons.append(recon)

            # TODO Visualization of end poses to get knowledge of the neigborhood properties
            # TODO MSE end poses to get knowledge of the reconstruction accuracy and distribution
            # TODO Policy debugging network!

        pose_results = np.array(pose_results)
        latents = np.array(latents)
        print("norm", np.linalg.norm(end_poses - pose_results))

        f = open(os.path.join(save_path, 'reconstructions_model_{}.pkl'.format(model_idx)), 'wb')

        pickle.dump([
            end_poses, pose_results,
            latents, trajectories, recons
        ], f)

        f.close()


def load_data(file_path, num_joints, num_actions):

    time_steps_raw, positions_raw, _, _, end_poses = np.load(os.path.join("../trajectory_data", file_path, 'trajectories.pkl'))
    num_samples = len(time_steps_raw)

    positions = np.zeros([num_samples, num_joints, num_actions])

    for i in range(num_samples):
        _, smooth_positions, _, _ = smooth_trajectory(time_steps_raw[i], positions_raw[i], num_actions, num_joints)
        positions[i] = smooth_positions
    positions = (positions - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)

    return positions, end_poses

if __name__ == '__main__':

    args = parse_arguments(behavioural_vae=True, debug=True)
    main(args)

#    print("Get Average Error:")
#    demo.multiple_demonstrations(10)
#    print("Random Latent Imitations")
#    print("Plottting")
#    demo.generate_multiple_images(30)
