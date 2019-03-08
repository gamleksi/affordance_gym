import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torch.nn import functional as F
import numpy as np

from affordance_gym.utils import parse_traj_arguments, parse_vaed_arguments, parse_policy_arguments, parse_policy_train_arguments, save_arguments, use_cuda
from affordance_gym.utils import plot_loss, plot_scatter, plot_latent_distributions
from affordance_gym.perception_policy import end_effector_pose, Predictor


from env_setup.env_setup import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON, LOOK_AT_EPSILON
from env_setup.env_setup import VAED_MODELS_PATH, POLICY_MODELS_PATH, TRAJ_MODELS_PATH

from TrajectoryVAE.utils import MIN_ANGLE, MAX_ANGLE
from TrajectoryVAE.ros_monitor import ROSTrajectoryVAE


'''

Trains the affordance policy in a supervised manner with previously generated training data (generate_perception_data.py). 

ROS is not needed.

The final position of the end effector is solved by solving the forward kinematics of the larg joint pose (the end_effector_pose function).

The forward kinematics solver is included to the PyTorch computation graph. 

'''


def load_dataset(perception_name, fixed_camera, debug):

    data_path = os.path.join(VAED_MODELS_PATH, perception_name, 'mujoco_latents')
    data_files = os.listdir(data_path)

    if debug:
        data_files = [data_files[0], data_files[-2]]

    print("Loading: ", data_files)
    # Multiple data packages exist
    latents = []
    camera_distances = []
    azimuths = []
    elevations = []
    target_coords = []
    cup_ids = []
    lookats = []

    for file in data_files:
        print(file)
        dataset = np.load(os.path.join(data_path, file))

        latents.append(dataset[0][:, 0, :]) # Bug fix
        lookats.append(dataset[1][:, :2])
        camera_distances.append(dataset[2])
        azimuths.append(dataset[3])
        elevations.append(dataset[4])
        cup_ids.append(dataset[5])
        target_coords.append(dataset[6])

    # Arrays to numpy
    latents = np.concatenate(latents)
    lookats = np.concatenate(lookats)
    camera_distances = np.concatenate(camera_distances)
    azimuths = np.concatenate(azimuths)
    elevations = np.concatenate(elevations)
    target_coords = np.concatenate(target_coords)
    cup_ids = np.concatenate(cup_ids)

    lookats = (lookats - (np.array(LOOK_AT[:2]) - LOOK_AT_EPSILON)) / (LOOK_AT_EPSILON * 2)
    camera_distances = (camera_distances - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
    azimuths = (azimuths - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)
    elevations = (elevations - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)

    if fixed_camera:

        # The first camera params
        distance = camera_distances[0]
        azimuth = azimuths[0]
        elevation = elevations[0]
        lookat = lookats[0]

        fixed_indices = camera_distances == distance
        fixed_camera = fixed_camera * (lookats == lookat)
        fixed_indices = fixed_indices * (elevations == azimuth)
        fixed_indices = fixed_indices * (azimuths == elevation)

        inputs = latents[fixed_indices]
        target_coords = target_coords[fixed_indices]

    else:
        inputs = np.concatenate([latents, lookats, camera_distances[:, None], azimuths[:, None], elevations[:, None]], axis=1)

    print(inputs.shape)

    if debug:
        indices = np.random.random_integers(0, inputs.shape[0], 100)
        inputs = inputs[indices]
        target_coords = target_coords[indices]

    # To tensor
    inputs = torch.Tensor(inputs)
    target_coord = torch.Tensor(target_coords)
    return data.TensorDataset(inputs, target_coord)


def main(args):

    save_path = os.path.join(POLICY_MODELS_PATH, args.policy_name)
    save_arguments(args, save_path)

    device = use_cuda()

    assert(args.model_index > 0)

    action_vae = ROSTrajectoryVAE(os.path.join(TRAJ_MODELS_PATH, args.vae_name), args.latent_dim, args.num_actions,
                                       model_index=args.model_index, num_joints=args.num_joints)

    # Trajectory generator
    traj_decoder = action_vae.model.decoder
    traj_decoder.eval()
    traj_decoder.to(device)

    # Load data
    dataset = load_dataset(args.g_name, args.fixed_camera, args.debug)

    # Policy
    if args.fixed_camera:
        policy = Predictor(args.g_latent, args.latent_dim, args.num_params)
    else:
        policy = Predictor(args.g_latent + 5, args.latent_dim, args.num_params)

    policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    optimizer.zero_grad()

    print("Dataset size", dataset.__len__())
    train_size = int(dataset.__len__() * 0.7)
    test_size = dataset.__len__() - train_size

    trainset, testset = data.random_split(dataset, (train_size, test_size))

    train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_processes)
    test_loader = data.DataLoader(testset, batch_size=10000)

    best_val = np.inf

    avg_train_losses = []
    avg_val_losses = []


    for epoch in range(args.num_epoch):

        print("Epoch {}".format(epoch + 1))

        policy.train()
        # Training
        train_losses = []
        end_poses = []
        target_poses = []
        latents = []

        for input, target_pose in train_loader:

            # latent1 -> latent2
            latent_1, target_pose = input.to(device), target_pose.to(device)

            latent_2 = policy(Variable(latent_1))

            # latent2 -> trajectory
            trajectories = traj_decoder(latent_2)

            # Reshape to trajectories
            trajectories = action_vae.model.to_trajectory(trajectories)

            # Get the last joint pose
            end_joint_pose = trajectories[:, :, -1]
            # Unnormalize
            end_joint_pose = (MAX_ANGLE - MIN_ANGLE) * end_joint_pose + MIN_ANGLE

            # joint pose -> cartesian
            end_pose = end_effector_pose(end_joint_pose, device)

            loss = F.mse_loss(end_pose, target_pose)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            end_poses.append(end_pose.detach().cpu().numpy())
            target_poses.append(target_pose.cpu().numpy())
            latents.append(latent_2.detach().cpu().numpy())

        avg_loss = np.mean(train_losses)
        avg_train_losses.append(avg_loss)
        print("Average error distance (training) {}".format(np.sqrt(avg_loss)))

        train_poses = np.concatenate(end_poses)
        train_targets = np.concatenate(target_poses)

        # Validation

        policy.eval()
        val_losses = []
        end_poses = []
        target_poses = []

        for latent_1, target_pose in test_loader:

            # latent1 -> latent2
            latent_1, target_pose = latent_1.to(device), target_pose.to(device)
            latent_2 = policy(Variable(latent_1))

            # latent2 -> trajectory
            trajectories = traj_decoder(latent_2)

            # Reshape to trajectories
            trajectories = action_vae.model.to_trajectory(trajectories)

            # Get the last joint pose
            end_joint_pose = trajectories[:, :, -1]

            # Unnormalize
            end_joint_pose = (MAX_ANGLE - MIN_ANGLE) * end_joint_pose + MIN_ANGLE

            # joint pose -> cartesian
            end_pose = end_effector_pose(end_joint_pose, device)

            loss = F.mse_loss(end_pose, target_pose)

            end_poses.append(end_pose.detach().cpu().numpy())

            val_losses.append(loss.item())
            end_poses.append(end_pose.detach().cpu().numpy())
            target_poses.append(target_pose.cpu().numpy())
            latents.append(latent_2.detach().cpu().numpy())

        avg_loss = np.mean(val_losses)
        val_poses = np.concatenate(end_poses)
        val_targets = np.concatenate(target_poses)
        latents = np.concatenate(latents)

        avg_val_losses.append(avg_loss)
        print("Average error distance (validation) {}".format(np.sqrt(avg_loss)))

        if avg_loss < best_val:
            best_val = avg_loss
            torch.save(policy.state_dict(), os.path.join(save_path, '{}_model.pth.tar'.format(epoch)))

        plot_scatter(train_poses, train_targets, os.path.join(save_path, 'train_scatter.png'))
        plot_scatter(val_poses, val_targets, os.path.join(save_path, 'val_scatter.png'))
        poses = np.concatenate([train_poses, val_poses])
        targets = np.concatenate([train_targets, val_targets])
        plot_scatter(poses, targets, os.path.join(save_path, 'full_scatter.png'))
        plot_latent_distributions(latents, os.path.join(save_path, 'latents_distribution.png'))

        plot_loss(avg_train_losses, avg_val_losses, 'Avg mse', os.path.join(save_path, 'avg_mse.png'))
        plot_loss(np.log(avg_train_losses), np.log(avg_val_losses), 'Avg mse in log scale', os.path.join(save_path, 'avg_log_mse.png'))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Train a perception policy in a supervised manner. Training data should be generated for a used vaed')

    parse_traj_arguments(parser)
    parse_vaed_arguments(parser)
    parse_policy_arguments(parser)
    parse_policy_train_arguments(parser)

    parser.add_argument('--debug', dest='debug', action='store_true', help='Uses only small set of data for training')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    main(args)
