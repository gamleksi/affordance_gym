import argparse
import tf
import geometry_msgs.msg
import torch
# arm_group: 'panda_arm'
# reset joint values for panda [0.0, 0.0, 0.0, 0.0, 0.0, 3.1, np.pi/2]


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


def load_parameters(policy, load_path):
    model_path = os.path.join(load_path, 'rl_model.pth.tar')
    policy.load_state_dict(torch.load(model_path))
    policy.eval()


def parse_arguments(behavioural_vae, policy):

    parser = argparse.ArgumentParser(description='MOTION PLANNER')

    if behavioural_vae:

        parser.add_argument('--vae-name', default='lumi_1DConv_b-1_l-5_a-32_ch-8', type=str, help='')

        parser.add_argument('--latent-dim', default=5, type=int, help='') # VAE model determines
        parser.add_argument('--num_joints', default=7, type=int, help='')
        parser.add_argument('--num_actions', default=32, type=int, help='Smoothed trajectory') # VAE model determines
        parser.add_argument('--duration', default=0.5, type=float, help='Duration of generated trajectory')

        parser.add_argument('--conv', dest='conv', action='store_true')
        parser.add_argument('--no-conv', dest='conv', action='store_false')
        parser.set_defaults(conv=True)

        parser.add_argument('--conv-channel', default=2, type=int, help='1D conv out channel')
        parser.add_argument('--kernel-row', default=4, type=int, help='Size of Kernel window in 1D')

    if policy:

        parser.add_argument('--folder-name', default='example', type=str, help='')

        parser.add_argument('--iterations', default=1000, type=int)

        parser.add_argument('--batch-size', default=5, type=int)

        parser.add_argument('--lr', default=1.e-3, type=float)

        parser.add_argument('--debug', dest='debug', action='store_true')
        parser.set_defaults(debug=False)
        parser.add_argument('--random-goal', dest='random_goal', action='store_true')
        parser.set_defaults(random_goal=False)
        parser.add_argument('--test', dest='train', action='store_false')
        parser.set_defaults(train=True)

    args = parser.parse_args()
    return args

LUMI_X_LIM = [0.3, 0.55]
LUMI_Y_LIM = [-0.4, 0.4]
LUMI_Z_LIM = [.1, .1]

BEHAVIOUR_ROOT = '/home/aleksi/hacks/behavioural_ws/behaviroural_vae/behavioural_vae'
POLICY_ROOT = '/home/aleksi/mujoco_ws/src/motion_planning/rl_log'
