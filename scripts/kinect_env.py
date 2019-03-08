import os
import csv
import numpy as np
import argparse
from PIL import Image
import torch
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt

from affordance_gym.simulation_interface import SimulationInterface
from affordance_gym.remote_interface import RemoteMCInterface
from affordance_gym.perception_policy import Predictor

from affordance_gym.utils import parse_kinect_arguments, parse_policy_arguments, parse_traj_arguments, parse_vaed_arguments, parse_moveit_arguments, sample_visualize, load_parameters, use_cuda
from affordance_gym.monitor import TrajectoryEnv

from env_setup.env_setup import DISTANCE, AZIMUTH, ELEVATION,  LOOK_AT, LOOK_AT_EPSILON, CUP_NAMES, KINECT_EXPERIMENTS_PATH
from env_setup.env_setup import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON, VAED_MODELS_PATH, TRAJ_MODELS_PATH, POLICY_MODELS_PATH

from TrajectoryVAE.ros_monitor import ROSTrajectoryVAE
from AffordanceVAED.ros_monitor import RosPerceptionVAE

from tf.transformations import euler_from_quaternion, quaternion_matrix

from PyInquirer import prompt, style_from_dict, Token
import rospy


style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})


'''

Setup for doing real experiments

'''


def change_camera_pose(sim, real_hw, debug):
    # Sim object reference

    camera_found = False

    camera_questions = [
        {
            'type': 'confirm',
            'name': 'camera_found',
            'message': '',
            'default': False
        }
    ]

    while not(camera_found):

        if debug:
            cam_position = np.zeros(3)
            pitch = 0.0
            yaw = 0.0
            roll = np.pi
            kinect_lookat = LOOK_AT
            kinect_distance = DISTANCE
            kinect_azimuth = AZIMUTH
            kinect_elevation = ELEVATION

        else:

            cam_position, quaternions = sim.kinect_camera_pose()

            if (cam_position is None):
                continue

            (roll, pitch, yaw) = euler_from_quaternion(quaternion=quaternions)
            R = quaternion_matrix(quaternions)
            v = R[:3, 0] # view direction

            kinect_lookat = np.zeros(3)

            t = -cam_position[2] / v[2]
            kinect_lookat[0] = t * v[0] + cam_position[0]
            kinect_lookat[1] = t * v[1] + cam_position[1]

            kinect_distance = np.linalg.norm(kinect_lookat - cam_position)
            kinect_azimuth = (yaw / np.pi) * 180
            kinect_elevation = (-pitch / np.pi) * 180

        if not(real_hw):
            sim.change_camere_params(kinect_lookat, kinect_distance, kinect_azimuth, kinect_elevation)

        print("*****")
        print("Camera Position", cam_position, "roll", roll, "pitch", pitch, "yaw", yaw)
        print("*****")
        print("LOOKA_POINTS:")
        print("Current", kinect_lookat[0], kinect_lookat[1])
        print("Lookat values while training", LOOK_AT[0], LOOK_AT[1], "Epsilon", LOOK_AT_EPSILON)
        print("Current ERROR", np.abs(kinect_lookat[0] - LOOK_AT[0]), np.abs(kinect_lookat[1] - LOOK_AT[1]))
        print("*****")
        print("DISTANCE")
        print("current", kinect_distance)
        print("distance values while training", DISTANCE, "Epsilon", DISTANCE_EPSILON)
        print("Current ERROR", np.abs(kinect_distance - DISTANCE))
        print("*****")
        print("AZIMUTH")
        print("current", kinect_azimuth)
        print("azimuth values while training", AZIMUTH, "Epsilon", AZIMUTH_EPSILON)
        print("Current ERROR", np.abs(kinect_azimuth - AZIMUTH))
        print("*****")
        print("ELEVATION")
        print("current", kinect_elevation)
        print("elevation values while training", ELEVATION, "Epsilon", ELEVATION_EPSILON)
        print("Current ERROR", np.abs(kinect_elevation - ELEVATION))
        print("*****")
        print("LOOKAT PASSED X", np.abs(kinect_lookat[0] - LOOK_AT[0]) < LOOK_AT_EPSILON)
        print("LOOKAT PASSED Y", np.abs(kinect_lookat[1] - LOOK_AT[1]) < LOOK_AT_EPSILON)
        print("Distance Passed", np.abs(kinect_distance - DISTANCE) < DISTANCE_EPSILON)
        print("Azimuth Passed", np.abs(kinect_azimuth - AZIMUTH) < AZIMUTH_EPSILON)
        print("Elevation Passed", np.abs(kinect_elevation - ELEVATION) < ELEVATION_EPSILON)
        answer = prompt(camera_questions, style=style)
        camera_found = answer.get('camera_found')

    # Normalized Camera Params
    n_lookat = (kinect_lookat[:2] - (np.array(LOOK_AT[:2]) - LOOK_AT_EPSILON)) / (LOOK_AT_EPSILON * 2)
    n_camera_distance = (kinect_distance - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
    n_azimuth = (kinect_azimuth - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)
    n_elevation = (kinect_elevation - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
    camera_params = [n_lookat[0], n_lookat[1], n_camera_distance, n_azimuth, n_elevation]

    return camera_params, (kinect_lookat[0], kinect_lookat[1], kinect_distance, kinect_azimuth, kinect_elevation, roll)

def crop_top(image, top_crop, width_crop):
    width, height = image.size
    left = 0
    top = top_crop
    right = width - width_crop
    bottom = height
    return image.crop((left, top, right, bottom))

def main(args):

    rospy.init_node('kinect_env', anonymous=True)
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
    policy = Predictor(args.g_latent + 5, args.latent_dim, args.num_params)
    policy.to(device)
    policy_path = os.path.join(POLICY_MODELS_PATH, args.policy_name)
    load_parameters(policy, policy_path, 'model')

    # Simulation interface
    if args.real_hw:

        assert(args.duration > 5) # this ensures that nothing dangerous happens :D

        planning_interface = RemoteMCInterface()
        planning_interface.reset(0)
    else:
        # planning_interface = SimulationInterface(arm_name='lumi_arm', gripper_name='lumi_hand')

        planning_interface = SimulationInterface(arm_name='lumi_arm')
        planning_interface.reset(duration=2.0)

    camera_params, log_cam_params = change_camera_pose(planning_interface, args.real_hw, args.debug)

    env = TrajectoryEnv(action_vae, planning_interface, args.num_actions, num_joints=args.num_joints, trajectory_duration=args.duration)

    app_questions = [
        {
            'type': 'list',
            'name': 'run',
            'message': 'Do you want to run a new experiment (y: yes, yc: yes and a change camera pose, n: no)?',
            'choices': ['y', 'yc', 'r', 'n']
        }
    ]

    save_path = os.path.join(KINECT_EXPERIMENTS_PATH, args.log_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Running...")
    run_experiment = True
    change_camera = False

    i = 0

    kinect_service = '/camera/rgb/image_raw'

    cup_log = {}
    for cup_name in CUP_NAMES:
        cup_log[cup_name] = [[], [], [], [], []] # Target Pose, End Effector Pose, error sum

    log_path = os.path.join(save_path, 'cup_log.csv')

    if not(os.path.exists(os.path.join(save_path, 'inputs'))):
        os.makedirs(os.path.join(save_path, 'inputs'))

        f = open(log_path, 'w')
        writer = csv.writer(f)
        writer.writerow(['cup_type', 'cup_x', 'cup_y', 'end_pose_x', 'end_pose_y',
                 'kinect_lookat_x', 'kinect_lookat_y', 'kinect_distance', 'kinect_azimuth',
                         'kinect_elevation', 'roll', 'top_crop', 'width_crop', 'image_file', 'num_random_objects'])
        f.close()
    else:
        previous_data = pd.read_csv(log_path)
        previous_data = previous_data.values
        cup_types = np.array(previous_data[:, 0], str)

        for i in range(cup_types.__len__()):
            cup_type = cup_types[i]
            cup_log[cup_type][0].append(float(previous_data[i, 1]))
            cup_log[cup_type][1].append(float(previous_data[i, 2]))
            cup_log[cup_type][2].append(float(previous_data[i, 3]))
            cup_log[cup_type][3].append(float(previous_data[i, 4]))

        i = len(os.listdir(os.path.join(save_path, 'inputs')))

    cup_questions = []

    if args.x_pose is None:
       cup_questions.append({
               'type': 'input',
               'name': 'x_pose',
               'message': 'X position of the cup'
           })


    if args.y_pose is None:
       cup_questions.append({
               'type': 'input',
               'name': 'y_pose',
               'message': 'Y position of the cup'
           })

    if args.cup_type is None:
       cup_questions.append({
               'type': 'list',
               'name': 'cup_type',
               'message': 'What is the cup type?',
               'choices': CUP_NAMES
           })

    if args.random_objs is None:

       cup_questions.append(
           {
               'type': 'input',
               'name': 'random_objs',
               'message': 'Number of random objects?'
           }
       )

    # env.gripper_open()
    while run_experiment:

        if change_camera:
            camera_params, log_cam_params = change_camera_pose(planning_interface, args.real_hw, args.debug)

        cup_answers = prompt(cup_questions)
        # env.gripper_close()

        cup_x = float(cup_answers.get('x_pose')) if (args.x_pose is None) else args.x_pose
        cup_y = float(cup_answers.get('y_pose'))  if (args.y_pose is None) else args.y_pose
        cup_type = str(cup_answers.get('cup_type'))  if (args.cup_type is None) else args.cup_type

        num_random_objects = int(cup_answers.get('random_objs'))  if (args.random_objs is None) else args.random_objs

        image_arr = planning_interface.capture_image(kinect_service)
        image = Image.fromarray(image_arr)

        # For visualization
        affordance, sample = perception.reconstruct(crop_top(image, args.top_crop, args.width_crop)) # TODO sample visualize

        # Image -> Latent1
        latent1 = perception.get_latent(crop_top(image, args.top_crop, args.width_crop))

        camera_input = Variable(torch.Tensor(camera_params).to(device))
        camera_input = camera_input.unsqueeze(0)
        latent1 = torch.cat([latent1, camera_input], 1)

        # latent and camera params -> latent2
        latent2 = policy(latent1)
        latent2 = latent2.detach().cpu().numpy() # TODO fix this!

        # Latent2 -> trajectory (mujoco)
        _, end_pose = env.do_latent_imitation(latent2[0])
        end_pose = np.array(end_pose[:2])

        print('distance error', np.linalg.norm(np.array([cup_x, cup_y]) - end_pose))
        sample_visualize(sample, affordance, os.path.join(save_path, 'kinect_results'), i)

        print("end_pose", end_pose)
        # env.gripper_open()

        answers = prompt(app_questions, style=style)
        env.reset_environment()

        if answers.get("run") != "r":
            i += 1

            image.save(os.path.join(save_path, 'inputs', 'sample_{}.png'.format(i)))

            cup_log[cup_type][0].append(cup_x)
            cup_log[cup_type][1].append(cup_y)
            cup_log[cup_type][2].append(end_pose[0])
            cup_log[cup_type][3].append(end_pose[1])
            cup_log[cup_type][4].append(num_random_objects)

            plt.figure(figsize=(5, 10))
            plt.scatter(cup_log[cup_type][0], cup_log[cup_type][1], label='cup pos', c='r')
            plt.scatter(cup_log[cup_type][2], cup_log[cup_type][3], label='end pos', c='b')
            plt.legend()
            plt.savefig(os.path.join(save_path, '{}_results.png'.format(cup_type)))
            plt.close()

            f = open(log_path, 'a')
            writer = csv.writer(f)
            writer.writerow([cup_type, cup_x, cup_y, end_pose[0], end_pose[1],
               log_cam_params[0], log_cam_params[1], log_cam_params[2], log_cam_params[3],
                             log_cam_params[4], log_cam_params[5], args.top_crop, args.width_crop, 'sample_{}.png'.format(i), num_random_objects])
            f.close()

            if answers.get("run") == "y":
                run_experiment = True
                change_camera = False
            elif answers.get("run") == "yc":
                run_experiment = True
                change_camera = True
            else:
                run_experiment = False
                change_camera = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Do Affordance Leaning Experiments')

    parse_vaed_arguments(parser)
    parse_traj_arguments(parser)
    parse_policy_arguments(parser)
    parse_kinect_arguments(parser)
    parse_moveit_arguments(parser)

    parser.add_argument('--debug', dest='debug', action='store_true', help='run a fake setup')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    main(args)
