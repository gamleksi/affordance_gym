#! /usr/bin/env python
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.utils import LUMI_X_LIM, LUMI_Y_LIM, LUMI_Z_LIM
from motion_planning.trajectory_parser import TrajectoryParser
import argparse
import os
import numpy as np
import rospy


parser = argparse.ArgumentParser(description='Trajectory Generator Arguments')
parser.add_argument('--save-folder', default='example', type=str, help='')
parser.add_argument('--num-samples', default=20000, type=int, help='Number of samples')
parser.add_argument('--num-joints', default=7, type=int, help='')

args = parser.parse_args()

ROOT_PATH = '../trajectory_data'
SAVE_PATH = os.path.join(ROOT_PATH, args.save_folder)

NUM_SAMPLES = args.num_samples
NUM_JOINTS = args.num_joints


if __name__ == '__main__':

    gripper_name = None
    simulation_interface = SimulationInterface('lumi_arm', gripper_name)
    simulation_interface.reset(1)
    trajectory_saver = TrajectoryParser(SAVE_PATH, 'trajectories', NUM_JOINTS)

    rospy.on_shutdown(trajectory_saver.save)

    steps = np.int(np.sqrt(NUM_SAMPLES))

    i = 0

    for x in np.linspace(LUMI_X_LIM[0], LUMI_X_LIM[1], steps):
        for y in np.linspace(LUMI_Y_LIM[0], LUMI_Y_LIM[1], steps):
            plan = simulation_interface.plan_end_effector_to_position(x_p=x, y_p=y, z_p=LUMI_Z_LIM[0])
            if plan:
                trajectory_saver.add_trajectory(plan)
                i += 1
            else:
                print(x, y)


                if i % 5000 == 0:
                    trajectory_saver.save()
                    print('Generated and Saved: {}'.format(i))

    trajectory_saver.save()
