#! /usr/bin/env python
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.trajectory_parser import TrajectoryParser
import argparse
import os


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
    simulation_interface.reset()
    trajectory_saver = TrajectoryParser(SAVE_PATH, 'trajectories', NUM_JOINTS)

    for i in range(NUM_SAMPLES):
        plan = simulation_interface.random_plan()
        trajectory_saver.add_trajectory(plan)

        if i % 100 == 0:
            trajectory_saver.save()
            print('Generated and Saved: {}'.format(i))

    trajectory_saver.save()