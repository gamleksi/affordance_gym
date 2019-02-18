# coding=utf-8
import os
import argparse
import tf
import geometry_msgs.msg
import torch
import numpy as np
from gibson.tools import affordance_to_array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def create_pose(x_p, y_p, z_p, x_o, y_o, z_o, w_o):
    """Creates a pose using quaternions

    Creates a pose for use with MoveIt! using XYZ coordinates and XYZW
    quaternion values

    :param x_p: The X-coordinate for the pose
    :param y_p: The Y-coordinate for the pose
    :param z_p: The Z-coordinate for the pose
    :param x_o: The X-value for the orientation
    :param y_o: The Y-value for the orientation
    :param z_o: The Z-value for the orientation
    :param w_o: The W-value for the orientation
    :type x_p: float
    :type y_p: float
    :type z_p: float
    :type x_o: float
    :type y_o: float
    :type z_o: float
    :type w_o: float
    :returns: Pose
    :rtype: PoseStamped
    """
    pose_target = geometry_msgs.msg.Pose()
    pose_target.position.x = x_p
    pose_target.position.y = y_p
    pose_target.position.z = z_p
    pose_target.orientation.x = x_o
    pose_target.orientation.y = y_o
    pose_target.orientation.z = z_o
    pose_target.orientation.w = w_o
    return pose_target


def create_pose_euler(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad):

    """Creates a pose using euler angles

    Creates a pose for use with MoveIt! using XYZ coordinates and RPY
    orientation in radians

    :param x_p: The X-coordinate for the pose
    :param y_p: The Y-coordinate for the pose
    :param z_p: The Z-coordinate for the pose
    :param roll_rad: The roll angle for the pose
    :param pitch_rad: The pitch angle for the pose
    :param yaw_rad: The yaw angle for the pose
    :type x_p: float
    :type y_p: float
    :type z_p: float
    :type roll_rad: float
    :type pitch_rad: float
    :type yaw_rad: float
    :returns: Pose
    :rtype: PoseStamped
    """

    quaternion = tf.transformations.quaternion_from_euler(
            roll_rad, pitch_rad, yaw_rad)
    return create_pose(
            x_p, y_p, z_p,
            quaternion[0], quaternion[1],
            quaternion[2], quaternion[3])

def print_pose(pose, tag='Pose'):
    print("{}: x: {}, y: {}, z: {}".format(tag, pose[0], pose[1], pose[2]))


def load_parameters(model, load_path, model_name):
    model_path = os.path.join(load_path, "{}.pth.tar".format(model_name))
    model.load_state_dict(torch.load(model_path))
    model.eval()


def parse_arguments(behavioural_vae=False, policy=False, gibson=False, debug=False, feedforward=False, policy_eval=False, kinect=False):

    parser = argparse.ArgumentParser(description='MOTION PLANNER')

    if behavioural_vae:

        parser.add_argument('--vae-name', default='model_v1', type=str, help='')

        parser.add_argument('--latent-dim', default=5, type=int, help='') # VAE model determines
        parser.add_argument('--num_joints', default=7, type=int, help='')
        parser.add_argument('--num_actions', default=24, type=int, help='Smoothed trajectory') # VAE model determines
        parser.add_argument('--duration', default=0.5, type=float, help='Duration of generated trajectory')

        parser.add_argument('--conv', dest='conv', action='store_true')
        parser.add_argument('--no-conv', dest='conv', action='store_false')
        parser.set_defaults(conv=False)

        parser.add_argument('--conv-channel', default=2, type=int, help='1D conv out channel')
        parser.add_argument('--kernel-row', default=4, type=int, help='Size of Kernel window in 1D')
        parser.add_argument('--model-index', default=-1, type=int)

    if policy:

        parser.add_argument('--policy-name', default='model_v1', type=str, help='')

        parser.add_argument('--num-epoch', default=1000, type=int)
        parser.add_argument('--num-params', default=128, type=int)

        parser.add_argument('--batch-size', default=124, type=int)

        parser.add_argument('--lr', default=1.e-3, type=float)

        parser.add_argument('--test', dest='train', action='store_false')
        parser.set_defaults(train=True)

        parser.add_argument('--num-processes', default=16, type=int)

        parser.add_argument('--fixed-camera', dest='fixed_camera', action='store_true')
        parser.set_defaults(fixed_camera=False)

    if gibson:

        parser.add_argument('--g-name', default='rgb_model_v1', type=str)
        parser.add_argument('--g-latent', default=10, type=int)
        parser.add_argument('--cup-id', default=1, type=int)
        parser.add_argument('--clutter-env', dest='clutter_env', action='store_true')
        parser.set_defaults(clutter_env=False)

#    if debug:
#        parser.add_argument('--dataset-name', default='lumi_rtt_star_v2')

#    if feedforward:
#
#       parser.add_argument('--random-goal', dest='random_goal', action='store_true')
#       parser.set_defaults(random_goal=False)
#       parser.add_argument('--policy-name', type=str, default='example')
#       parser.add_argument('--num-steps', default=10, type=int)

    if policy_eval:
        parser.add_argument('--randomize-all', dest='randomize_all', action='store_true')
        parser.set_defaults(randomize_all=False)

    if kinect:
        parser.add_argument('--log', dest='log', action='store_true')
        parser.set_defaults(log=False)
        parser.add_argument('--log-name', default='kinect_example', type=str)


    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    return args


def sample_visualize(image, affordance, sample_path, id):

    image = np.transpose(image, (1, 2, 0))

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    affordance = affordance_to_array(affordance).transpose((1, 2, 0)) / 255.

    samples = np.stack((image, affordance))

    fig, axeslist = plt.subplots(ncols=2, nrows=1, figsize=(30, 30))

    for idx in range(samples.shape[0]):
        axeslist.ravel()[idx].imshow(samples[idx], cmap=plt.jet())
        axeslist.ravel()[idx].set_axis_off()

    plt.savefig(os.path.join(sample_path, 'sample_{}.png'.format(id)))
    plt.close(fig)


#    print("Give the position of a cup:")
#    x = float(raw_input("Enter x: "))
#    y = float(raw_input("Enter y: "))
#  ('camera_pose', [0.729703198019277, 0.9904542035333381, 0.5861775350680969])
#  ('Kinect lookat', array([0.71616937, -0.03126261, 0.]))
#  ('distance', 1.1780036104266332)
#  ('azimuth', -90.75890510585465)
#   ('kinect_elevation', -29.841508670508976)


def use_cuda():

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    return torch.device('cuda' if use_cuda else 'cpu')

def save_arguments(args, save_path):

    args = vars(args)

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    file = open(os.path.join(save_path, "arguments.txt"), 'w')
    lines = [item[0] + " " + str(item[1]) + "\n" for item in args.items()]
    file.writelines(lines)


def plot_loss(train, val, title, save_to):

    steps = range(1, train.__len__() + 1)

    fig = plt.figure()

    plt.plot(steps, train, 'r', label='Train')
    plt.plot(steps, val, 'b', label='Validation')

    plt.title(title)
    plt.legend()
    plt.savefig(save_to)
    plt.close()


def plot_scatter(constructed, targets, save_to):
    fig = plt.figure()
    plt.scatter(targets[:, 0], targets[:, 1], label='targets', c='r')
    plt.scatter(constructed[:, 0], constructed[:, 1], label='constructed', c='b')
    plt.legend()
    plt.savefig(save_to)
    plt.close()

def plot_latent_distributions(latents, save_to):

    fig, axes = plt.subplots(latents.shape[1], 1, sharex=True, figsize=[30, 30])

    for i in range(latents.shape[1]):
        ax = axes[i]
        batch = latents[:, i]
        ax.hist(batch, bins=100)
        ax.set_title('Latent {}'.format(i + 1))
        ax.set_xlabel('x')
        ax.set_ylabel('frequency')
    plt.savefig(save_to)
    plt.close()


# These values work only in the regular robot env (base link different in th robot_table env)
LUMI_X_LIM = [0.4, 0.75]
LUMI_Y_LIM = [-0.20, 0.20]
LUMI_Z_LIM = [.3, .3]

NO_CUP_SHOWN_POSE = [0.42, -0.18]

BEHAVIOUR_ROOT = '/home/aleksi/catkin_ws/src/motion_planning/src/behavioural_vae/models'
POLICY_ROOT = '/home/aleksi/catkin_ws/src/motion_planning/policy_models'
GIBSON_ROOT = '/home/aleksi/catkin_ws/src/motion_planning/src/gibson/perception_results'

# Perception parameters
LOOK_AT = [0.70, 0.0, 0.0]
DISTANCE = 1.16
AZIMUTH = -90.
ELEVATION = -30

LOOK_AT_EPSILON = 0.05
DISTANCE_EPSILON = 0.05
ELEVATION_EPSILON = 2.
AZIMUTH_EPSILON = 2.

CUP_X_LIM = [0.4, 0.75]
CUP_Y_LIM = [-0.20, 0.20]
