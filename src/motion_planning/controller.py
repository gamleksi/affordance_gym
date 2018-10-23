#!/usr/bin/env python

import sys
import rospy
import moveit_commander as mc
import geometry_msgs.msg
import numpy as np
import tf
import random
import warnings
warnings.filterwarnings("error")


def create_pose(x_p, y_p, z_p, x_o, y_o, z_o, w_o):
        """Creates a pose using quaternions

        Creates a pose for use with MoveIt! using XYZ coordinates and XYZW
        quaternion values

        :param x_p: The X-coordinate for the pose
        :param y_p: The Y-coordinate for the pose
        :param z_p: The Z-coordinate for the pose
        :param x_o: The X-value for the orientation
        :param y_o: The Y-value for the orientation
        :param z_o: The Z-value for the orientation
        :param w_o: The W-value for the orientation
        :type x_p: float
        :type y_p: float
        :type z_p: float
        :type x_o: float
        :type y_o: float
        :type z_o: float
        :type w_o: float
        :returns: Pose
        :rtype: PoseStamped
        """
        pose_target = geometry_msgs.msg.Pose()
        pose_target.position.x = x_p
        pose_target.position.y = y_p
        pose_target.position.z = z_p
        pose_target.orientation.x = x_o
        pose_target.orientation.y = y_o
        pose_target.orientation.z = z_o
        pose_target.orientation.w = w_o
        return pose_target


def create_pose_euler(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad):
        """Creates a pose using euler angles

        Creates a pose for use with MoveIt! using XYZ coordinates and RPY
        orientation in radians

        :param x_p: The X-coordinate for the pose
        :param y_p: The Y-coordinate for the pose
        :param z_p: The Z-coordinate for the pose
        :param roll_rad: The roll angle for the pose
        :param pitch_rad: The pitch angle for the pose
        :param yaw_rad: The yaw angle for the pose
        :type x_p: float
        :type y_p: float
        :type z_p: float
        :type roll_rad: float
        :type pitch_rad: float
        :type yaw_rad: float
        :returns: Pose
        :rtype: PoseStamped
        """
        quaternion = tf.transformations.quaternion_from_euler(
                        roll_rad, pitch_rad, yaw_rad)
        return create_pose(
                        x_p, y_p, z_p,
                        quaternion[0], quaternion[1],
                        quaternion[2], quaternion[3])


class RobotWrapper(object):

        def __init__(
                        self, arm_group, reset_joint_values,
                        gripper_group=None, gripper_open_values=[0.04, 0.04]
                        ):
                self.robot = mc.RobotCommander()
                self.reset_joint_values = reset_joint_values
                self.arm = self.init_group(arm_group)

                self.gripper_open_values = gripper_open_values

                if gripper_group is not None:
                        self.gripper = self.init_group(gripper_group)
                else:
                        self.gripper = None

                self.reset_position()

        def init_group(self, name):

                arm = mc.MoveGroupCommander(name)
                arm.set_planner_id("ESTkConfigDefault")
                arm.allow_replanning(False)
                arm.set_goal_position_tolerance(0.0005)
                arm.set_goal_orientation_tolerance(0.005)
                return arm

        def get_planning_frame(self):
                return self.robot.get_planning_frame()

        def gripper_open(self):

                if self.gripper is not None:
                        self.gripper.set_joint_value_target(
                                        self.gripper_open_values)
                        self.gripper.plan()
                        self.gripper.go(wait=True)
                        rospy.sleep(1)

        def gripper_close(self):

                if self.gripper is not None:
                        self.gripper.set_joint_value_target([0., 0.])
                        self.gripper.plan()
                        self.gripper.go(wait=True)
                        rospy.sleep(1)

        def reset_position(self):

                self.arm.set_joint_value_target(self.reset_joint_values)
                self.arm.plan()
                self.arm.go(wait=True)
                rospy.sleep(1)
                self.gripper_close()

        def plan_path_global(
                        self, x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad):

                self.arm.clear_pose_targets()

                pose = create_pose_euler(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad)

                self.arm.set_pose_target(pose)

                plan = self.arm.plan()
                if len(plan.joint_trajectory.points) < 1:
                        plan = None
                return plan

        def plan_and_move(self, x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad):
                """Plans and moves an arm to target
                """
                self.reset_position()
                plan = self.plan_path_global(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad)
                if plan is not None:
                        self.arm.go(wait=True)
                        rospy.sleep(1)
                return plan


class RobotScene(object):

        def __init__(self, robot, x_lim, y_lim, z_lim):
                mc.roscpp_initialize(sys.argv)
                rospy.init_node('motion_planning', anonymous=True)
                self.robot = robot
                self.scene = mc.PlanningSceneInterface()
                self.remove_object('table')
                self.add_table(0.6, 0, 0.1, x_width=1, y_widht=1, z_width=0.2)
                self.x_lim = x_lim
                self.y_lim = y_lim
                self.z_lim = z_lim

        def init_pos_msg(self):

                p = mc.PoseStamped()
                p.header.frame_id = self.robot.get_planning_frame()

                return p

        def add_table(self, x, y, z, x_width=1, y_widht=1, z_width=1):
                p = self.init_pos_msg()
                p.pose.position.x = x
                p.pose.position.y = y
                p.pose.position.z = z
                self.scene.add_box('table', p, (x_width, y_widht, z_width))

        def remove_object(self, name):
                self.scene.remove_world_object(name)

        def open_gripper(self):
                self.robot.gripper_open()

        def close_gripper(self):
                self.robot.gripper_close()

        def reset_position(self):
                self.robot.reset_position()

        def print_current_pose(self):
                pose = self.robot.arm.get_current_pose()
                print('Current Pose:')
                print(pose)
                rpy = self.robot.arm.get_current_rpy()

        def print_current_joint_states(self):
                print('Current Joint Values:')
                print(self.robot.arm.get_current_joint_values())

        def random_trajectory(self):
                x_p = random.uniform(self.x_lim[0], self.x_lim[1])
                y_p = random.uniform(self.y_lim[0], self.y_lim[1])
                z_p = random.uniform(self.z_lim[0], self.z_lim[1])
                return self.move_arm_to_position(x_p=x_p, y_p=y_p, z_p=z_p)

        def random_plan(self):

                while True:

                        x_p = random.uniform(self.x_lim[0], self.x_lim[1])
                        y_p = random.uniform(self.y_lim[0], self.y_lim[1])
                        z_p = random.uniform(self.z_lim[0], self.z_lim[1])
                        plan = self.plan_trajecory_to_position(x_p=x_p, y_p=y_p, z_p=z_p)

                        if (plan is not None):
                                break
                        else:
                                print(x_p, y_p, z_p)
                                print('Failed')

                return plan

        def move_arm_to_position(self, x_p=0.5, y_p=0, z_p=0.5, roll_rad=np.pi/2, pitch_rad=np.pi/4, yaw_rad=np.pi/2):
                return self.robot.plan_and_move(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad)

        def plan_trajecory_to_position(self, x_p=0.5, y_p=0, z_p=0.5, roll_rad=np.pi/2, pitch_rad=np.pi/4, yaw_rad=np.pi/2):
                return self.robot.plan_path_global(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad)

        def do_plan(self, plan):
                succeed = self.robot.arm.execute(plan)
                return succeed

from moveit_msgs.msg import RobotTrajectory, genpy
from moveit_msgs.msg import _RobotTrajectory

class RobotTrajectoryHandler(object):

        def __init__(self, duration, initial_joints, joint_names = ['lwr_a1_joint', 'lwr_a2_joint', 'lwr_e1_joint', 'lwr_a3_joint', 'lwr_a4_joint', 'lwr_a5_joint', 'lwr_a6_joint']):
                self.duration = duration
                self.joint_names = joint_names
                self.num_joints = len(joint_names)
                self.initial_joints = np.array([initial_joints])

        def build_message(self, trajectory):
                trajectory_msg = RobotTrajectory()
                trajectory_msg.joint_trajectory.joint_names = self.joint_names

                trajectory = np.concatenate((self.initial_joints, trajectory))
                num_steps = len(trajectory + 1)

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
        robot_env = RobotScene()

