#!/usr/bin/env python

import sys
import rospy
import moveit_commander as mc
import numpy as np
import random
from motion_planning.utils import LUMI_Y_LIM, LUMI_X_LIM, LUMI_Z_LIM
from motion_planning.utils import create_pose_euler
from std_srvs.srv import Empty

# Adding obstacles:

# self.moveit_environment_interface = mc.PlanningSceneInterface()
# self.robot_interface = mc.RobotCommander()

#    def init_pos_msg(self):
#
#        p = mc.PoseStamped()
#        p.header.frame_id = self.robot.get_planning_frame()
#
#        return p
#
#    def add_table(self, x, y, z, x_width=1, y_widht=1, z_width=1):
#        p = self.init_pos_msg()
#        p.pose.position.x = x
#        p.pose.position.y = y
#        p.pose.position.z = z
#   # self.scene.add_box('table', p, (x_width, y_widht, z_width))
#
#    def remove_object(self, name):
#        self.scene.remove_world_object(name)


GRIPPER_POSITION_LIMITS = (LUMI_X_LIM, LUMI_Y_LIM, LUMI_Z_LIM)
GRIPPER_OPEN_VALUES = (0.04, 0.04)

class SimulationInterface(object):

    def __init__(self, arm_name, gripper_name=None, gripper_open_values=GRIPPER_OPEN_VALUES,
                 gripper_position_limit=GRIPPER_POSITION_LIMITS):

        # Initialize Moveit Node interface
        mc.roscpp_initialize(sys.argv)
        rospy.init_node('motion_planning', anonymous=True)

        self.arm_planner = self.build_planning_interface(arm_name)

        if gripper_name is not None:
            self.gripper_planner = self.build_planning_interface(gripper_name)
            self.gripper_open_values = gripper_open_values
        else:
            self.gripper_planner = None

        self.gripper_position_limit = gripper_position_limit


    def build_planning_interface(self, name):

        arm = mc.MoveGroupCommander(name)
        arm.set_planner_id("ESTkConfigDefault")
        arm.allow_replanning(False)
        arm.set_goal_position_tolerance(0.0005)
        arm.set_goal_orientation_tolerance(0.005)

        return arm

    def gripper_open(self):

        if self.gripper_planner is not None:
            self.gripper_planner.set_joint_value_target(
                            self.gripper_open_values)
            self.gripper_planner.plan()
            self.gripper_planner.go(wait=True)
            rospy.sleep(1)

    def current_joint_values(self):
        return self.arm_planner.get_current_joint_values()

    def joint_names(self):
        return self.arm_planner.get_joints()

    def current_pose(self):
        return self.arm_planner.get_current_pose()

    def gripper_close(self):
        if self.gripper_planner is not None:
            self.gripper_planner.set_joint_value_target([0., 0.])
            self.gripper_planner.plan()
            self.gripper_planner.go(wait=True)
            rospy.sleep(1)

    def print_current_pose(self):
        pose = self.arm_planner.get_current_pose()
        print('Current Pose:')
        print(pose)
        print(self.arm_planner.get_current_rpy())

    def print_current_joint_states(self):
        print('Current Joint Values:')
        print(self.arm_planner.get_current_joint_values())

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

# For grasping roll_rad=np.pi/2, pitch_rad=np.pi/4, yaw_rad=np.pi/2):
    def plan_end_effector_to_position(self, x_p=0.5, y_p=0, z_p=0.5,  roll_rad=0, pitch_rad=np.pi, yaw_rad=np.pi):

        self.arm_planner.clear_pose_targets()
        pose = create_pose_euler(x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad)
        self.arm_planner.set_pose_target(pose)
        plan = self.arm_planner.plan()

        if len(plan.joint_trajectory.points) < 1:
                plan = None

        return plan

    def move_arm_to_position(
            self, x_p=0.5, y_p=0, z_p=0.5,  roll_rad=0, pitch_rad=np.pi, yaw_rad=np.pi):
        plan = self.plan_end_effector_to_position(
                x_p, y_p, z_p, roll_rad, pitch_rad, yaw_rad)

        if plan is not None:
            self.arm_planner.go(wait=True)
        else:
            print("Failed to move the arm")

        return plan

    def do_plan(self, plan):
        succeed = self.arm_planner.execute(plan)
        return succeed

    def reset(self):

        self.arm_planner.clear_pose_targets()
        reset = rospy.ServiceProxy('lumi_mujoco/reset', Empty)
        try:
            reset()
        except rospy.ServiceException as exc:
            print("Reset did not work:" + str(exc))

        rospy.sleep(1.0)
#        self.gripper_close()


from moveit_msgs.msg import RobotTrajectory, genpy
from moveit_msgs.msg import _RobotTrajectory


class CommunicationHandler(object):

    def __init__(self, duration, initial_joints, joint_names):
        self.duration = duration
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.initial_joints = np.array([initial_joints])

    def build_message(self, trajectory):
        trajectory_msg = RobotTrajectory()
        trajectory_msg.joint_trajectory.joint_names = self.joint_names

        trajectory = trajectory.transpose(1, 0)

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
    planner = SimulationInterface(arm_name='lumi_arm')

