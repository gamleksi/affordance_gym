#! /usr/bin/env python
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.utils import LUMI_X_LIM, LUMI_Y_LIM, LUMI_Z_LIM
from motion_planning.trajectory_parser import TrajectoryParser, parse_trajectory
from behavioural_vae.utils import smooth_trajectory, MAX_ANGLE, MIN_ANGLE
import os
import numpy as np
import rospy
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Trajectory Generator Arguments')
parser.add_argument('--save-folder', default='example', type=str, help='')
parser.add_argument('--num-samples', default=20000, type=int, help='Number of samples')
parser.add_argument('--num-joints', default=7, type=int, help='')

args = parser.parse_args()

ROOT_PATH = '../trajectory_data'
SAVE_PATH = os.path.join(ROOT_PATH, args.save_folder)

NUM_SAMPLES = args.num_samples
NUM_JOINTS = args.num_joints


def plot_trajectory(trajectories, image, path):

    fig = plt.figure(figsize=(30, 30))
    columns = 1
    rows = trajectories.shape[1]

    steps = range(1, trajectories.shape[2] + 1)
    for i in range(rows):

        fig.add_subplot(rows, columns, i + 1)

        for traj_idx in range(trajectories.shape[0]):

            plt.plot(steps, trajectories[traj_idx, i, :], label='Trajectory {}'.format(traj_idx + 1))

        plt.legend()

    plt.savefig(os.path.join(path, image))
    plt.close()


if __name__ == '__main__':

    # planning_id = "RRTstar"
    planning_id = "RRT"

    gripper_name = None
    simulation_interface = SimulationInterface('lumi_arm', gripper_name, planning_id=planning_id)
    simulation_interface.reset(1)
    trajectory_saver = TrajectoryParser(SAVE_PATH, 'trajectories', NUM_JOINTS)

    i = 0

#    steps = np.int(np.sqrt(NUM_SAMPLES))
#    if not(os.path.exists(args.save_folder)):
#        os.makedirs(args.save_folder)
#
#    trajectory_vars = []
#    most_var_joints = []
#
#    epsilon = 0.04
#    log_file = open(os.path.join(args.save_folder, "results.txt"), 'w')
#    log_file.write("Planning ID: {}, Epsilon {} '\n'".format(planning_id, epsilon))
#
#
#    for x in np.linspace(LUMI_X_LIM[0], LUMI_X_LIM[1], steps):
#
#      for y in np.linspace(LUMI_Y_LIM[0], LUMI_Y_LIM[1], steps):
#
#        trajectories = []
#
#        plan = simulation_interface.plan_end_effector_to_position(x_p=x, y_p=y, z_p=LUMI_Z_LIM[0])
#        time_steps_raw, positions_raw, _, _ = parse_trajectory(plan)
#        _, trajectory, _, _ = smooth_trajectory(time_steps_raw, positions_raw, 24, NUM_JOINTS)
#        trajectories.append(trajectory)
#
#        neighbors = 5
#
#        for j in range(1, neighbors):
#
#            x_p = x + epsilon * np.cos(j * np.pi * 2 / neighbors)
#            y_p = y + epsilon * np.sin(j * np.pi * 2 / neighbors)
#
#            plan = simulation_interface.plan_end_effector_to_position(x_p=x_p, y_p=y_p, z_p=LUMI_Z_LIM[0])
#            time_steps_raw, positions_raw, _, _ = parse_trajectory(plan)
#            _, trajectory, _, _ = smooth_trajectory(time_steps_raw, positions_raw, 24, NUM_JOINTS)
#            trajectories.append(trajectory)
#
#        print("Trajectory", i)
#        trajectories = np.array(trajectories)

#        var_trajectories = trajectories.var(0)
#        trajectory_vars.append(var_trajectories.sum())
#        most_var_joint = np.argmax(var_trajectories.var(1))
#        most_var_joints.append(most_var_joint)
#
#        log_file.write("X: {}, Y: {}, VAR: {}, MAX VAR JOINT: {} \n".format(x, y, trajectory_vars[i], most_var_joints[i]))
#
#        normalized_trajectories = (trajectories - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)
#        plot_trajectory(trajectories, '{}_trajectories.png'.format(i), args.save_folder)
#        plot_trajectory(trajectories, '{}_normalized.png'.format(i), args.save_folder)
#        i += 1
#
#    log_file.write("Mean of VAR: {} \n".format(np.mean(trajectory_vars)))
#    log_file.write("MAX COUNTS of JOINTS: {} \n".format(np.bincount(most_var_joints)))
#    log_file.close()

    rospy.on_shutdown(trajectory_saver.save)

    for x in np.linspace(LUMI_X_LIM[0], LUMI_X_LIM[1], 50):
      for y in np.linspace(LUMI_Y_LIM[0], LUMI_Y_LIM[1], 160):
          plan = simulation_interface.plan_end_effector_to_position(x_p=x, y_p=y, z_p=LUMI_Z_LIM[0])
          if plan:
              trajectory_saver.add_trajectory(plan)
              i += 1
          else:
              print(x, y)

#          if i % int(i / 100) == 0:
#              trajectory_saver.save()
#              print('Generated and Saved: {}'.format(i))

    trajectory_saver.save()
