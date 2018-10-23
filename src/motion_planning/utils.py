import argparse

# arm_group: 'panda_arm'
# reset joint values for panda [0.0, 0.0, 0.0, 0.0, 0.0, 3.1, np.pi/2]

def parse_arguments():

        parser = argparse.ArgumentParser(description='ROS trajectory parser')
        parser.add_argument('--save-path', default='/home/aleksi/catkin_ws/src/motion_planning/rl_log', type=str, help='')
        parser.add_argument('--model-name', default='simple_trajectories', type=str, help='')
        parser.add_argument('--num_samples', default=20000, type=int, help='Number of samples')
        parser.add_argument('--num_joints', default=7, type=int, help='')
        parser.add_argument('--num_actions', default=20, type=int, help='')
        parser.add_argument('--latent_dim', default=10, type=int, help='')
        parser.add_argument('--duration', default=4, type=int, help='Duratoin of generated trajectory')
        parser.add_argument('--model_name', default='no_normalization_model_v2', type=str, help='')

        parser.add_argument('--normalized', dest='normalized', action='store_true')
        parser.add_argument('--no-normalized', dest='normalized', action='store_false')
        parser.set_defaults(normalized=False)

        args = parser.parse_args()
        return args

KUKA_X_LIM = [0.46, 0.76]
KUKA_Y_LIM = [-0.4, 0.4]
KUKA_z_LIM = [1.4, 1.4]
KUKA_RESET_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
