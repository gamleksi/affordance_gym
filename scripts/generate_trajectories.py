#! /usr/bin/env python
from motion_planning.controller import RobotScene, RobotWrapper
from motion_planning.trajectory_parser import TrajectoryParser
from motion_planning.utils import KUKA_X_LIM, KUKA_Y_LIM, KUKA_z_LIM, parse_arguments, KUKA_RESET_JOINTS

args = parse_arguments()

SAVE_PATH = args.save_path
SAVE_FILE = args.save_file
NUM_SAMPLES = args.num_samples
NUM_JOINTS = args.num_joints

if __name__ == '__main__':

        robot = RobotWrapper('full_lwr', KUKA_RESET_JOINTS)
        env = RobotScene(robot, KUKA_X_LIM, KUKA_Y_LIM, KUKA_z_LIM)
        parser = TrajectoryParser(SAVE_PATH, SAVE_FILE, NUM_JOINTS)

        for i in range(NUM_SAMPLES):
                plan = env.random_plan()
                parser.add_trajectory(plan)

                if i % 200 == 0:
                        parser.save()
                        print(i)
