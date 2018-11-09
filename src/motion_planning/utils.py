import argparse
import tf
import geometry_msgs.msg
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


def parse_arguments():

    parser = argparse.ArgumentParser(description='ROS trajectory parser')
    parser.add_argument('--save-path', default='/home/aleksi/catkin_ws/src/motion_planning/rl_log', type=str, help='')
    parser.add_argument('--model-name', default='simple_full_b-5', type=str, help='')
    parser.add_argument('--num_samples', default=20000, type=int, help='Number of samples')
    parser.add_argument('--num_joints', default=7, type=int, help='')
    parser.add_argument('--num_actions', default=20, type=int, help='')
    parser.add_argument('--latent_dim', default=5, type=int, help='')
    parser.add_argument('--duration', default=4, type=int, help='Duratoin of generated trajectory')
    parser.add_argument('--model_name', default='no_normalization_model_v2', type=str, help='')

    parser.add_argument('--normalized', dest='normalized', action='store_true')
    parser.add_argument('--no-normalized', dest='normalized', action='store_false')
    parser.set_defaults(normalized=False)

    args = parser.parse_args()
    return args

KUKA_X_LIM = [0.46, 0.76]
KUKA_Y_LIM = [-0.4, 0.4]
KUKA_z_LIM = [.4, .4]
KUKA_RESET_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
KUKA_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
