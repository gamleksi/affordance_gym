import os
# import csv
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from motion_planning.simulation_interface import SimulationInterface
from motion_planning.perception_policy import Predictor, end_effector_pose
from behavioural_vae.ros_monitor import ROSTrajectoryVAE
from gibson.ros_monitor import RosPerceptionVAE
# from tf.transformations import euler_from_quaternion, quaternion_matrix
# from gibson.tools import affordance_to_array

from motion_planning.utils import parse_arguments, GIBSON_ROOT, load_parameters, BEHAVIOUR_ROOT, POLICY_ROOT, use_cuda
from motion_planning.utils import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, LOOK_AT_EPSILON
from motion_planning.utils import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON
from behavioural_vae.utils import MIN_ANGLE, MAX_ANGLE
from motion_planning.monitor import TrajectoryEnv
# from PyInquirer import prompt, style_from_dict, Token
import pandas as pd
import rospy


#    print("Give the position of a cup:")
#    x = float(raw_input("Enter x: "))
#    y = float(raw_input("Enter y: "))
#    ('camera_pose', [0.729703198019277, 0.9904542035333381, 0.5861775350680969])
#    ('Kinect lookat', array([0.71616937, -0.03126261, 0.]))
#    ('distance', 1.1780036104266332)
#    ('azimuth', -90.75890510585465)
#    ('kinect_elevation', -29.841508670508976)



# OLD Camera params (these values were used when training old policy)
# LOOK_AT = [0.70, 0.0, 0.0]
# DISTANCE = 1.2
# AZIMUTH = -90.
# ELEVATION = -30
# ELEVATION_EPSILON = 0.5
# AZIMUTH_EPSILON = 0.5
# DISTANCE_EPSILON = 0.05

# Camera values when samples were gathered
KINECT_LOOKAT = [0.71616937, -0.03126261, 0.]
KINECT_DISTANCE = 1.1780036104266332
KINECT_AZIMUTH = -90.75890510585465
KINECT_ELEVATION = -29.841508670508976

DEBUG_SAMPLE_PATH = '/home/aleksi/mujoco_ws/src/motion_planning/kinect_experiments/full_experiments/inputs'
DEBUG_DATA_PATH = '/home/aleksi/mujoco_ws/src/motion_planning/kinect_experiments/full_experiments/cup_log.csv'

CUP_NAMES = ['rocket', 'karol', 'gray', 'can', 'blue', 'subway', 'yellow', 'mirror', 'red']


def take_num(elem):
    elem = elem.split('_')[-1]
    elem = elem.split('.')[0]
    val = int(elem)
    return val

def main(args):

    device = use_cuda()

    assert(args.model_index > -1)

    bahavior_model_path = os.path.join(BEHAVIOUR_ROOT, args.vae_name)
    action_vae = ROSTrajectoryVAE(bahavior_model_path, args.latent_dim, args.num_actions,
                                  model_index=args.model_index, num_joints=args.num_joints)

    # Trajectory generator
    traj_decoder = action_vae.model.decoder

    gibson_model_path = os.path.join(GIBSON_ROOT, args.g_name)
    perception = RosPerceptionVAE(gibson_model_path, args.g_latent)

    # Policy
    policy = Predictor(args.g_latent + 5, args.latent_dim)
    policy.to(device)
    policy_path = os.path.join(POLICY_ROOT, args.policy_name)
    load_parameters(policy, policy_path, 'model')

    # Simulation interface
    # sim = SimulationInterface(arm_name='lumi_arm')
    # sim.reset(duration=2.0)

    # env = TrajectoryEnv(action_vae, sim, args.num_actions, num_joints=args.num_joints, trajectory_duration=0.5)

    # Getting object poses
    data = pd.read_csv(DEBUG_DATA_PATH)
    data = data.values
    cup_poses = np.array(data[:, 1:3], dtype=np.float32)

    debug_images = os.listdir(DEBUG_SAMPLE_PATH)
    debug_images.sort(key=take_num)

    # Camera param normalization
    n_lookat = (np.array(KINECT_LOOKAT[:2]) - (np.array(LOOK_AT[:2]) - LOOK_AT_EPSILON)) / (LOOK_AT_EPSILON * 2)
    n_camera_distance = (KINECT_DISTANCE - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
    n_azimuth = (KINECT_AZIMUTH - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)
    n_elevation = (KINECT_ELEVATION - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
    camera_params = [n_lookat[0], n_lookat[1], n_camera_distance, n_azimuth, n_elevation]

    end_poses = []
    distances = []

    print(debug_images)

    for i, sample in enumerate(zip(cup_poses, debug_images)):

        image_path = os.path.join(DEBUG_SAMPLE_PATH, sample[1])

        image = Image.open(image_path)

        width, height = image.size
        left = 0
        top = 44
        right = width - 28
        bottom = height
        image = image.crop((left, top, right, bottom))

        # Image -> Latent1
        latent1 = perception.get_latent(image)

        camera_input = Variable(torch.Tensor(camera_params).to(device))
        camera_input = camera_input.unsqueeze(0)
        latent1 = torch.cat([latent1, camera_input], 1)

        # latent and camera params -> latent2

        latent2 = policy(latent1)
        trajectories = traj_decoder(latent2)

        # Reshape to trajectories
        trajectories = action_vae.model.to_trajectory(trajectories)

        end_joint_pose = trajectories[:, :, -1]

        end_joint_pose = (MAX_ANGLE - MIN_ANGLE) * end_joint_pose + MIN_ANGLE
        # joint pose -> cartesian
        end_pose = end_effector_pose(end_joint_pose, device)
        end_pose = end_pose.cpu().detach().numpy()[0]

        # Latent2 -> trajectory (mujoco)
        # _, end_pose = env.do_latent_imitation(latent2[0])
#         end_pose = np.array((
#             end_pose.pose.position.x,
#             end_pose.pose.position.y))

        end_poses.append(end_pose)
        print("END POSE", end_pose)
        print("CUP POSE", sample[0])

        distance = np.linalg.norm(end_pose - sample[0])
        distances.append(distance)

        print("distance", distance)

        # sim.reset(duration=2.0)

    end_poses = np.array(end_poses)
    print("avg distance", np.mean(distances))

    fig, axes = plt.subplots(3, 3, sharex=True, figsize=[30, 30])

    for i, cup_name in enumerate(CUP_NAMES):

        ax = axes[int(i/3)][i%3]
        cup_indices = np.array([cup_name in s for s in data[:, 0]])

        real_poses = cup_poses[cup_indices]
        pred_poses = end_poses[cup_indices]

        print(cup_name, 'avg_mean:', np.linalg.norm(real_poses - pred_poses, axis=1).mean())

        ax.scatter(real_poses[:, 0], real_poses[:, 1], c='r', label='real')
        ax.scatter(pred_poses[:, 0], pred_poses[:, 1], c='b', label='pred')
        ax.set_title(cup_name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

    plt.savefig('large_ds.png')

    np.save('results.npy', (cup_poses, end_poses))

if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, gibson=True, policy=True, policy_eval=True, kinect=True)
    main(args)
