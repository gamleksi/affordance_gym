#! /usr/bin/env python
from behavioural_vae.utils import smooth_trajectory
from motion_planning.controller import RobotScene, RobotWrapper, RobotTrajectoryHandler
from motion_planning.trajectory_parser import TrajectoryParser, parse_trajectory
from behavioural_vae.ros_monitor import ROSTrajectoryVAE
import numpy as np
from scipy import interpolate

# arm_group: 'panda_arm'
# reset joint values for panda [0.0, 0.0, 0.0, 0.0, 0.0, 3.1, np.pi/2]
KUKA_START = [-2.214618353590936e-06, -1.0097254404151101e-05, -2.9373917476149813e-06, 3.5813285501617997e-06, 1.660007352022319e-06, -2.327471992913388e-07, 1.8815573543662367e-07]
KUKA_X_LIM = [0.46, 0.76]
KUKA_Y_LIM = [-0.4, 0.4]
KUKA_z_LIM = [1.4, 1.4]
SAVE_PATH = '/home/aleksi/hacks/behavioural_ws/trajectories'
NUM_SAMPLES = 201


if __name__ == '__main__':

        robot = RobotWrapper('full_lwr', [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0])
        env = RobotScene(robot, KUKA_X_LIM, KUKA_Y_LIM, KUKA_z_LIM)
        plan = env.random_trajectory()
        time_steps_raw, positions_raw, _, _ = parse_trajectory(plan)
        #import pdb; pdb.set_trace()
        _, positions, _, _ = smooth_trajectory(time_steps_raw, positions_raw, 20, 7)

        model = ROSTrajectoryVAE('no_normalization_model_v2', 10, 20, num_joints=7, root_path='/home/aleksi/hacks/behavioural_ws/behaviroural_vae/behavioural_vae')
        duration = 4
        handler = RobotTrajectoryHandler(duration)
        result = model.get_result(positions)

        result_plan = handler._parse_plan(result)
        robot.reset_position()
        robot.do_plan(result_plan)

#        parser = TrajectoryParser(SAVE_PATH, 7)

#        for i in range(NUM_SAMPLES):
#                plan = env.random_plan()
#                parser.add_trajectory(plan)
#
#                if i  % 200 == 0:
#                        parser.save()
#                        print(i)
