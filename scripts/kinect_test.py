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
from motion_planning.utils import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON, LOOK_AT_EPSILON
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

    import rospy

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.2)  # 10hz
    sim = SimulationInterface(arm_name='panda_arm')

    # kinect_service = '/camera/rgb/image_raw'

    while not rospy.is_shutdown():

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

        print("*****")
        print("Camera Position", cam_position, "roll", (roll / np.pi) * 180)
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

        rate.sleep()



if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, gibson=True, policy=True, policy_eval=True)
    main(args)
