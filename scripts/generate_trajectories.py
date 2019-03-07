#! /usr/bin/env python
import os
import numpy as np
import rospy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

from affordance_gym.simulation_interface import SimulationInterface
from affordance_gym.trajectory_parser import TrajectoryParser, parse_trajectory
from TrajectoryVAE.utils import smooth_trajectory, MAX_ANGLE, MIN_ANGLE

from env_setup.env_setup import LUMI_X_LIM, LUMI_Y_LIM, LUMI_Z_LIM

parser = argparse.ArgumentParser(description='Trajectory Generator Arguments')

parser.add_argument('--save-folder', default='example', type=str, help='folder name')
parser.add_argument('--save-root', default='../trajectory_data', type=str, help='data path')

parser.add_argument('--num-samples', default=64, type=int, help='Number of samples')
parser.add_argument('--num-joints', default=7, type=int, help='The Number of robot joints')

parser.add_argument("--debug", action="store_true", help='Explore manipulator results')
parser.add_argument('--epsilon', default=0.04, type=float, help='For debugging the stochasticity of the manipulation algorithm')
parser.add_argument("--rtt-star", help="Use RTTStar", action="store_true")
parser.add_argument()

args = parser.parse_args()


'''

Generates training data for the trajectory VAE.
Only MoveIt should be running simultaneously

'''


def plot_trajectory(trajectories, image, path, normalized):

    fig = plt.figure(figsize=(30, 30))
    columns = 1
    rows = trajectories.shape[1]

    steps = range(1, trajectories.shape[2] + 1)
    for i in range(rows):

        fig.add_subplot(rows, columns, i + 1)

        for traj_idx in range(trajectories.shape[0]):

            plt.plot(steps, trajectories[traj_idx, i, :], label='Trajectory {}'.format(traj_idx + 1))

            if normalized:
                plt.ylim(0, 1)
            else:
                plt.ylim(-np.pi, np.pi)

        plt.legend()

    plt.savefig(os.path.join(path, image))
    plt.close()


if __name__ == '__main__':

    rospy.init_node('talker', anonymous=True)

    save_path = os.path.join(args.save_root, args.save_folder)

    if args.rtt_star:
        planning_id = "RRTstar"
    else:
        planning_id = "RRT"

    print("Planning_ID", planning_id)

    gripper_name = None
    simulation_interface = SimulationInterface('lumi_arm', gripper_name=gripper_name, planning_id=planning_id)
    simulation_interface.reset(1)

    if args.debug:

        steps = np.int(np.sqrt(args.num_samples))
        if not(os.path.exists(args.save_folder)):
            os.makedirs(args.save_folder)

        trajectory_vars = []
        most_var_joints = []

        log_file = open(os.path.join(args.save_folder, "results.txt"), 'w')
        log_file.write("Planning ID: {}, Epsilon {} '\n'".format(planning_id, args.epsilon))

        i = 0
        for x in np.linspace(LUMI_X_LIM[0], LUMI_X_LIM[1], steps):

          for y in np.linspace(LUMI_Y_LIM[0], LUMI_Y_LIM[1], steps):

            trajectories = []

            plan = simulation_interface.plan_end_effector_to_position(x_p=x, y_p=y, z_p=LUMI_Z_LIM[0])
            time_steps_raw, positions_raw, _, _ = parse_trajectory(plan)
            _, trajectory, _, _ = smooth_trajectory(time_steps_raw, positions_raw, 24, args.num_joints)
            trajectories.append(trajectory)

            neighbors = 5

            for j in range(1, neighbors):

                x_p = x + args.epsilon * np.cos(j * np.pi * 2 / neighbors)
                y_p = y + args.epsilon * np.sin(j * np.pi * 2 / neighbors)

                plan = simulation_interface.plan_end_effector_to_position(x_p=x_p, y_p=y_p, z_p=LUMI_Z_LIM[0])
                time_steps_raw, positions_raw, _, _ = parse_trajectory(plan)
                _, trajectory, _, _ = smooth_trajectory(time_steps_raw, positions_raw, 24, args.num_joints)
                trajectories.append(trajectory)

            print("Trajectory", i)
            trajectories = np.array(trajectories)

            var_trajectories = trajectories.var(0)
            trajectory_vars.append(var_trajectories.sum())
            most_var_joint = np.argmax(var_trajectories.var(1))
            most_var_joints.append(most_var_joint)

            log_file.write("X: {}, Y: {}, VAR: {}, MAX VAR JOINT: {} \n".format(x, y, trajectory_vars[i], most_var_joints[i]))

            normalized_trajectories = (trajectories - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)
            plot_trajectory(trajectories, '{}_trajectories.png'.format(i), args.save_folder, False)
            plot_trajectory(normalized_trajectories, '{}_normalized.png'.format(i), args.save_folder, True)
            i += 1

        log_file.write("Mean of VAR: {} \n".format(np.mean(trajectory_vars)))
        log_file.write("MAX COUNTS of JOINTS: {} \n".format(np.bincount(most_var_joints)))
        log_file.close()

    else:

        trajectory_saver = TrajectoryParser(save_path, 'trajectories', args.num_joints)
        rospy.on_shutdown(trajectory_saver.save)
        i = 0

        for x in np.linspace(LUMI_X_LIM[0], LUMI_X_LIM[1], 100):
          for y in np.linspace(LUMI_Y_LIM[0], LUMI_Y_LIM[1], 100):
              plan = simulation_interface.plan_end_effector_to_position(x_p=x, y_p=y, z_p=LUMI_Z_LIM[0])
              if plan:
                  trajectory_saver.add_trajectory(plan, (x, y, LUMI_Z_LIM[0]))
                  i += 1
              else:
                  print(x, y)
              print("Sample", i)

        trajectory_saver.save()
