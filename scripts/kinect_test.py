import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable

from motion_planning.simulation_interface import SimulationInterface
from motion_planning.perception_policy import Predictor
from behavioural_vae.ros_monitor import ROSTrajectoryVAE
from gibson.ros_monitor import RosPerceptionVAE
from tf.transformations import euler_from_quaternion, quaternion_matrix

from motion_planning.utils import parse_arguments, GIBSON_ROOT, load_parameters, BEHAVIOUR_ROOT, POLICY_ROOT, use_cuda
from motion_planning.utils import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM
from motion_planning.utils import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON
from motion_planning.monitor import TrajectoryEnv


#def transformationMatrix(camera_position, look_at, tmp = np.array([0, 1, 0])):
#
#    forward = (camera_position - look_at) / np.linalg.norm(camera_position - look_at)
#    right = np.cross(tmp, forward)
#    up = np.cross(forward, right)
#    camToWorld = np.zeros([4,4])
#
#    camToWorld[0][0] = right[0]
#    camToWorld[0][1] = right[1]
#    camToWorld[0][2] = right[2]
#    camToWorld[1][0] = up[0]
#    camToWorld[1][1] = up[1]
#    camToWorld[1][2] = up[2]
#    camToWorld[2][0] = forward[0]
#    camToWorld[2][1] = forward[1]
#    camToWorld[2][2] = forward[2]
#
#    camToWorld[3][0] = camera_position[0]
#    camToWorld[3][1] = camera_position[1]
#    camToWorld[3][2] = camera_position[2]

#    return camToWorld


#def cameraCartesianCoords(distance, azimuth, elevation):
#
#    phi = (azimuth) * np.pi / 180.
#    theta = (elevation) * np.pi / 180.
#    coords = np.zeros(3)
#    coords[0] = distance * np.sin(phi) * np.cos(theta)
#    coords[1] = distance * np.cos(phi)
#    coords[2] = distance * np.sin(phi) * np.sin(theta)

#    return coords


def main(args):

#    device = use_cuda()

    # Trajectory generator
    # TODO Currently includes both encoder and decoder to GPU even though only encoder is used.

#    assert(args.model_index > -1)

#    bahavior_model_path = os.path.join(BEHAVIOUR_ROOT, args.vae_name)
#    action_vae = ROSTrajectoryVAE(bahavior_model_path, args.latent_dim, args.num_actions,
#                                  model_index=args.model_index, num_joints=args.num_joints)
    # pereception
    # TODO Currently includes both encoder and decoder to GPU even though only encoder is used.
#    gibson_model_path = os.path.join(GIBSON_ROOT, args.g_name)
#    perception = RosPerceptionVAE(gibson_model_path, args.g_latent)
#
#    # Policy
#    policy = Predictor(args.g_latent + 3, args.latent_dim)
#    policy.to(device)
#    policy_path = os.path.join(POLICY_ROOT, args.policy_name)
    #load_parameters(policy, policy_path, 'model')

    import rospy

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.2)  # 10hz
    while not rospy.is_shutdown():

        # Simulation interface
        sim = SimulationInterface(arm_name='lumi_arm')

        # Kinect parameters TODO
        cam_position, quaternions = sim.kinect_camera_pose()

        if cam_position is None:
            continue

#        if (cam_position is None):
#            return 0

        (roll, pitch, yaw) = euler_from_quaternion(quaternion=quaternions)
        print("roll", 180 * roll / np.pi)
        print("pitch", 180 * pitch / np.pi)
        print("yaw", 180 * yaw / np.pi)

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

#    env = TrajectoryEnv(action_vae, sim, args.num_actions, num_joints=args.num_joints)

        print('kinect_lookat', kinect_lookat)
        print('kinect distance', kinect_distance)
        print('kinect azimuth', kinect_azimuth)
        print('kinect elevation', kinect_elevation)
        rate.sleep()

#    coords = cameraCartesianCoords(kinect_distance, kinect_azimuth, kinect_elevation)
#    print('Computed position ', coords)
#    print('Computed distance ', np.linalg.norm(coords))


    kinect_service = '/camera/rgb/image_raw'

    # Normalized Camera Params
#    n_camera_distance = (kinect_distance - (DISTANCE - DISTANCE_EPSILON)) / (DISTANCE_EPSILON * 2)
#    n_azimuth = (kinect_azimuth - (AZIMUTH - AZIMUTH_EPSILON)) / (AZIMUTH_EPSILON * 2)
#    n_elevation = (kinect_elevation - (ELEVATION - ELEVATION_EPSILON)) / (ELEVATION_EPSILON * 2)
#    camera_params = [n_camera_distance, n_azimuth, n_elevation]
#
#    # run_program = input("enter 1 to run or 0 to stop")
#
#    image_arr = sim.capture_image(kinect_service)
#    image = Image.fromarray(image_arr)
#    image.show()

#    while run_program > 0:
#
#        print("Give the position of a cup:")
#        x = float(raw_input("Enter x: "))
#        y = float(raw_input("Enter y: "))
#        print("Running...")
#
#        image_arr = sim.capture_image(kinect_service)
#        image = Image.fromarray(image_arr)
#
#        # affordance, sample = perception.reconstruct(image) TODO sample visualize
#
#        # Image -> Latent1
#        latent1 = perception.get_latent(image)
#
#        camera_input = Variable(torch.Tensor().to(device))
#        camera_input = camera_params.unsqueeze(0)
#        latent1 = torch.cat([latent1, camera_input], 1)
#
#        # latent and camera params -> latent2
#        latent2 = policy(latent1)
#        latent2 = latent2.detach().cpu().numpy() # TODO fix this!
#
#        # Latent2 -> trajectory (mujoco)
#        _, end_pose = env.do_latent_imitation(latent2[0])
#        end_pose = np.array((
#            end_pose.pose.position.x,
#            end_pose.pose.position.y))
#
#        reward = np.linalg.norm(np.array([x, y]) - end_pose)
#
#        print('Distance error: {}'.format(reward))
#        print("goal", x, y)
#        print("end_pose", end_pose)
#
#        env.reset_environment(duration=5.0)
#
#        x = float(raw_input("Enter x: "))
#        y = float(raw_input("Enter y: "))
#
#        run_program = input("enter 1 to continue or 0 to stop")

    # print("AVG: ", np.mean(rewards), " VAR: ", np.var(rewards))




if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, gibson=True, policy=True, policy_eval=True)
    main(args)
