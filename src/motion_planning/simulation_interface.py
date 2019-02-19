#!/usr/bin/env python

import sys
import rospy
import numpy as np
import random
from motion_planning.moveit_commander_interface import MCInterface
from mujoco_ros_control.srv import ChangeCupPose, ChangeCameraParams
from std_srvs.srv import Empty
from motion_planning.utils import LUMI_Y_LIM, LUMI_X_LIM, LUMI_Z_LIM


GRIPPER_POSITION_LIMITS = (LUMI_X_LIM, LUMI_Y_LIM, LUMI_Z_LIM)


class SimulationInterface(MCInterface):

    def __init__(self, arm_name, gripper_name=None, planning_id='RRT', gripper_position_limit=GRIPPER_POSITION_LIMITS):

        super(SimulationInterface, self).__init__(arm_name, gripper_name=gripper_name, planning_id=planning_id)

        self.gripper_position_limit = gripper_position_limit

    def random_end_effector_pose(self):
        x_p = random.uniform(GRIPPER_POSITION_LIMITS[0][0], GRIPPER_POSITION_LIMITS[0][1])
        y_p = random.uniform(GRIPPER_POSITION_LIMITS[1][0], GRIPPER_POSITION_LIMITS[1][1])
        z_p = random.uniform(GRIPPER_POSITION_LIMITS[2][0], GRIPPER_POSITION_LIMITS[2][1])
        return x_p, y_p, z_p

    def random_trajectory(self):
        x_p, y_p, z_p = self.random_end_effector_pose()
        return self.move_arm_to_position(x_p=x_p, y_p=y_p, z_p=z_p)

    def random_plan(self):

        plan = None
        while plan is None:

            x_p, y_p, z_p = self.random_end_effector_pose()

            plan = self.plan_end_effector_to_position(x_p=x_p, y_p=y_p, z_p=z_p)

            if plan is None:
                print(x_p, y_p, z_p)
                print('Failed')

        return plan

    def reset(self, duration):

        self.arm_planner.clear_pose_targets()
        reset = rospy.ServiceProxy('lumi_mujoco/reset', Empty)
        try:
            reset()
        except rospy.ServiceException as exc:
            print("Reset did not work:" + str(exc))

        rospy.sleep(duration)

    def reset_table(self, x, y, z, object_name, duration=5.0):

       self.arm_planner.clear_pose_targets()

       reset = rospy.ServiceProxy('lumi_mujoco/reset_table', ChangeCupPose)
       try:
           reset(object_name, x, y, z)
       except rospy.ServiceException as exc:
           print("Reset did not work:" + str(exc))

       rospy.sleep(duration)

    def change_object_position(self, x, y, z, object_name, duration=0.5):

       change_pose = rospy.ServiceProxy('lumi_mujoco/change_object_pose', ChangeCupPose)
       try:
           change_pose(object_name, x, y, z)
       except rospy.ServiceException as exc:
           print("Reset did not work:" + str(exc))

       rospy.sleep(duration)

    def change_camere_params(self, look_at, distance, azimuth, elevation):

       assert(len(look_at) == 3)

       request = rospy.ServiceProxy('lumi_mujoco/change_camera', ChangeCameraParams)

       try:
           request(look_at[0], look_at[1], look_at[2], distance, azimuth, elevation)
       except rospy.ServiceException as exc:
           print("Camera Change did not work:" + str(exc))


from moveit_msgs.msg import RobotTrajectory, genpy
from motion_planning.srv import RobotTrajectoryRequest
from moveit_msgs.msg import _RobotTrajectory


class CommunicationHandler(object):

    def __init__(self, duration, initial_joints, joint_names):
        self.duration = duration
        self.joint_names = joint_names
        self.initial_joints = np.array([initial_joints])

    def build_message(self, trajectory):
        trajectory_msg = RobotTrajectoryRequest()
        trajectory_msg.joint_trajectory.joint_names = self.joint_names

        trajectory = trajectory.transpose(1, 0)

        trajectory = np.concatenate((self.initial_joints, trajectory))
        num_steps = len(trajectory + 1) # ??
        time_steps = np.linspace(0, self.duration, num_steps)

        points = []
        for i, positions in enumerate(trajectory):
            point = _RobotTrajectory.trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = genpy.Duration(time_steps[i])
            points.append(point)

            trajectory_msg.joint_trajectory.points = points

        return trajectory_msg

    def _parse_plan(self, plan):
        trajectory = []
        for p in plan.joint_trajectory.points:
            trajectory.append(p.positions)
        return self.build_message(trajectory)

    def return_positions(self, plan):
        return [point.positions for point in plan.joint_trajectory.points]

if __name__ == '__main__':
    planner = SimulationInterface(arm_name='lumi_arm')
    joints = planner.current_joint_values()
    print(joints)

