import numpy as np

from affordance_gym.simulation_interface import SimulationInterface

from affordance_gym.utils import parse_arguments
from env_setup.env_setup import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION
from env_setup.env_setup import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON, LOOK_AT_EPSILON

from tf.transformations import euler_from_quaternion, quaternion_matrix

import rospy

'''

Calibrate camera to be within the training area

'''


def main(args):


    rospy.init_node('kinect calibrate', anonymous=True)

    rate = rospy.Rate(0.2)  # 10hz
    sim = SimulationInterface(arm_name='panda_arm')

    while not rospy.is_shutdown():

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
    main()
