import os
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
from gibson.tools import affordance_to_array

from motion_planning.utils import parse_arguments, GIBSON_ROOT, load_parameters, BEHAVIOUR_ROOT, POLICY_ROOT, use_cuda
from motion_planning.utils import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM
from motion_planning.utils import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON
from motion_planning.monitor import TrajectoryEnv
from PyInquirer import prompt, style_from_dict, Token
import rospy


def sample_visualize(image, affordance, model_path, id):

    image = np.transpose(image, (1, 2, 0))

    sample_path = os.path.join(model_path,'mujoco_samples')
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


def change_camera_pose(sim):
    # Sim object reference

    camera_found = 0

    while (camera_found < 1):

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
        camera_found = input("Keep camera 1 pose, change camera pose 0")

    # Normalized Camera Params
    n_camera_distance = (kinect_distance - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
    n_azimuth = (kinect_azimuth - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)
    n_elevation = (kinect_elevation - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
    camera_params = [n_camera_distance, n_azimuth, n_elevation]

    return camera_params


style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})


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

    camera_params = change_camera_pose(sim)

    env = TrajectoryEnv(action_vae, sim, args.num_actions, num_joints=args.num_joints, trajectory_duration=0.5)

    app_questions = [
        {
            'type': 'list',
            'name': 'run',
            'message': 'Do you want to run a new experiment (y: yes, yc: yes and a change camera pose, n: no)?',
            'default': ['y', 'yc', 'n']
        }
    ]

    print("Running...")
    run_experiment = True
    change_camera = False

    i = 0
    kinect_service = '/camera/rgb/image_raw'

    while run_experiment:

        if change_camera:
            camera_params = change_camera_pose()

        image_arr = sim.capture_image(kinect_service)
        image = Image.fromarray(image_arr)

        # For visualization
        affordance, sample = perception.reconstruct(image) # TODO sample visualize

        sample_visualize(sample, affordance, 'kinect_samples', i)

        # Image -> Latent1
        latent1 = perception.get_latent(image)

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


#    print("Give the position of a cup:")
#    x = float(raw_input("Enter x: "))
#    y = float(raw_input("Enter y: "))
#  ('camera_pose', [0.729703198019277, 0.9904542035333381, 0.5861775350680969])
#  ('Kinect lookat', array([0.71616937, -0.03126261, 0.]))
#  ('distance', 1.1780036104266332)
#  ('azimuth', -90.75890510585465)
#   ('kinect_elevation', -29.841508670508976)



if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, gibson=True, policy=True, policy_eval=True)
    main(args)
