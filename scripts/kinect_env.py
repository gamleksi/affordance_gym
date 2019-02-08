import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from motion_planning.simulation_interface import SimulationInterface
from motion_planning.perception_policy import Predictor
from behavioural_vae.ros_monitor import ROSTrajectoryVAE
from gibson.ros_monitor import RosPerceptionVAE
from tf.transformations import euler_from_quaternion, quaternion_matrix

from motion_planning.utils import parse_arguments, GIBSON_ROOT, load_parameters, BEHAVIOUR_ROOT, POLICY_ROOT, use_cuda
from motion_planning.utils import DISTANCE, AZIMUTH, ELEVATION, sample_visualize
from motion_planning.utils import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON
from motion_planning.monitor import TrajectoryEnv
from PyInquirer import prompt, style_from_dict, Token
import rospy



style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})


def change_camera_pose(sim):
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

        # Kinect parameters TODO
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

        sim.change_camere_params(kinect_lookat, kinect_distance, kinect_azimuth, kinect_elevation)

        kinect_service = '/camera/rgb/image_raw'

        print("camera_pose", cam_position)
        print("Kinect lookat", kinect_lookat)
        print("distance", kinect_distance)
        print("azimuth", kinect_azimuth)
        print("kinect_elevation", kinect_elevation)
        answer = prompt(camera_questions, style=style)
        camera_found = answer.get('camera_found')

    # Normalized Camera Params
    n_camera_distance = (kinect_distance - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
    n_azimuth = (kinect_azimuth - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)
    n_elevation = (kinect_elevation - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
    camera_params = [n_camera_distance, n_azimuth, n_elevation]

    return camera_params, (kinect_distance, kinect_azimuth, kinect_elevation, roll, kinect_lookat)


def crop_top(image):
    width, height = image.size
    left = 0
    top = 21
    right = width
    bottom = height
    return image.crop((left, top, right, bottom))


def main(args):

    rospy.init_node('talker', anonymous=True)
    device = use_cuda()

   # Trajectory generator
   # TODO Currently includes both encoder and decoder to GPU even though only encoder is used.
    assert(args.model_index > -1)

    bahavior_model_path = os.path.join(BEHAVIOUR_ROOT, args.vae_name)
    action_vae = ROSTrajectoryVAE(bahavior_model_path, args.latent_dim, args.num_actions,
                                  model_index=args.model_index, num_joints=args.num_joints)
   # pereception
   # TODO Currently includes both encoder and decoder to GPU even though only encoder is used.
    gibson_model_path = os.path.join(GIBSON_ROOT, args.g_name)
    perception = RosPerceptionVAE(gibson_model_path, args.g_latent)

    # Policy
    policy = Predictor(args.g_latent + 3, args.latent_dim)
    policy.to(device)
    policy_path = os.path.join(POLICY_ROOT, args.policy_name)
    load_parameters(policy, policy_path, 'model')

    # Simulation interface
    sim = SimulationInterface(arm_name='lumi_arm')
    sim.reset(duration=2.0)

    camera_params, log_cam_params = change_camera_pose(sim)

    env = TrajectoryEnv(action_vae, sim, args.num_actions, num_joints=args.num_joints, trajectory_duration=0.5)

    app_questions = [
        {
            'type': 'list',
            'name': 'run',
            'message': 'Do you want to run a new experiment (y: yes, yc: yes and a change camera pose, n: no)?',
            'choices': ['y', 'yc', 'n']
        }
    ]

    save_path = os.path.join('../kinect_experiments', args.log_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Running...")
    run_experiment = True
    change_camera = False

    i = 0
    kinect_service = '/camera/rgb/image_raw' # TODO rectified service?

    if args.log:

        log_path = os.path.join(save_path, 'cup_log.csv')

        if not(os.path.exists(os.path.join(save_path, 'inputs'))):
            os.makedirs(os.path.join(save_path, 'inputs'))

            f = open(log_path, 'w')
            writer = csv.writer(f)
            writer.writerow(['cup_type', 'cup_x', 'cup_y', 'end_pose_x', 'end_pose_y',
                     'distance', 'azimuth', 'elevation', 'roll',
                     'kinect_lookat_x', 'kinect_lookat_y', 'kinect_lookat_z'])
            f.close()
        else:
            i = len(os.listdir(os.path.join(save_path, 'inputs')))

    while run_experiment:

        if change_camera:
            camera_params, log_cam_params = change_camera_pose()

        if args.log:
            cup_questions = [
                {
                    'type': 'input',
                    'name': 'cup_type',
                    'message': 'What is the cup type?'
                },
                {
                    'type': 'input',
                    'name': 'x_pose',
                    'message': 'X pose of the cup?'
                },
                {
                    'type': 'input',
                    'name': 'y_pose',
                    'message': 'Y pose of the cup?'
                }
            ]
            cup_answers = prompt(cup_questions)

        image_arr = sim.capture_image(kinect_service)
        image = Image.fromarray(image_arr)

        # For visualization
        affordance, sample = perception.reconstruct(crop_top(image)) # TODO sample visualize


        # Image -> Latent1
        latent1 = perception.get_latent(crop_top(image))

        camera_input = Variable(torch.Tensor(camera_params).to(device))
        camera_input = camera_input.unsqueeze(0)
        latent1 = torch.cat([latent1, camera_input], 1)

        # latent and camera params -> latent2
        latent2 = policy(latent1)
        latent2 = latent2.detach().cpu().numpy() # TODO fix this!

        # Latent2 -> trajectory (mujoco)
        _, end_pose = env.do_latent_imitation(latent2[0])
        end_pose = np.array((
            end_pose.pose.position.x,
            end_pose.pose.position.y))

        print("end_pose", end_pose)
        sim.reset_table(end_pose[0], end_pose[1], 0.0, 'box2')
        i += 1

        if args.log:

            cup_x = float(cup_answers.get('x_pose'))
            cup_y = float(cup_answers.get('y_pose'))
            cup_type = str(cup_answers.get('cup_type'))

            print('distance error', np.linalg.norm(np.array([cup_x, cup_y]) - end_pose))

            f = open(log_path, 'a')
            writer = csv.writer(f)
            writer.writerow([cup_type, cup_x, cup_y, end_pose[0], end_pose[1],
               log_cam_params[0], log_cam_params[1], log_cam_params[2], log_cam_params[3],
               log_cam_params[4][0], log_cam_params[4][1], log_cam_params[4][2]])
            f.close()

            image.save(os.path.join(save_path, 'inputs', 'sample_{}.png'.format(i)))

        sample_visualize(sample, affordance, os.path.join(save_path, 'kinect_results'), i)
        answers = prompt(app_questions, style=style)
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
    args = parse_arguments(behavioural_vae=True, gibson=True, policy=True, policy_eval=True, kinect=True)
    main(args)
