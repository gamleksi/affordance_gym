from __future__ import with_statement
from threading import Lock
import rospy
from affordance_gym.hardware_interface import HardwareInterface
from std_srvs import srv
from affordance_gym.srv import ChangePose, ChangePoseResponse, JointNames, JointNamesResponse, CurrentPose, CurrentPoseResponse
from affordance_gym.srv import RobotTrajectory, RobotTrajectoryResponse
from affordance_gym.srv import JointValues, JointValuesResponse

import argparse

'''

This process should run in the same server, as the main roscore.  
This setup was introduced, because there was a different ros version in the robot's computer than in the 'policy' computer.    

'''

parser = argparse.ArgumentParser(description='MC interface')
parser.add_argument('--velocity-factor', default=0.1, type=float)
parser.add_argument('--arm-name', default='lumi_arm', type=str)
parser.add_argument('--gripper-name', default=None, type=str)

args = parser.parse_args()


if __name__ == '__main__':

    rospy.init_node('mc_interface')
    planner = HardwareInterface(args.arm_name, args.gripper_name, args.velocity_factor)

    lock = Lock()

    def reset(req):
        with lock:
            planner.reset(None)
        return srv.EmptyResponse()

    def move_arm_to_position(req):

        with lock:
            plan = planner.move_arm_to_position(req.x, req.y, req.z)

        if plan is None:
            return ChangePoseResponse(False)
        else:
            return ChangePoseResponse(True)

    def gripper_close(req):
        with lock:
            planner.close_gripper()
        return srv.EmptyResponse()

    def gripper_open(req):
        with lock:
            planner.open_gripper()
        return srv.EmptyResponse()

    def do_plan(req):
        with lock:
            succeed = planner.do_plan(req)
        return RobotTrajectoryResponse(succeed)

    def joint_names(req):
        with lock:
            names = planner.joint_names()
        return JointNamesResponse(names)

    def current_joint_values(req):
        with lock:
            joint_values = planner.current_joint_values()
        return JointValuesResponse(joint_values)

    def current_pose(req):
        with lock:
            current_pose_values = planner.current_pose()
            print("current_pose_values", current_pose_values)
        return CurrentPoseResponse(current_pose_values)

    reset_service = rospy.Service('reset', srv.Empty, reset)
    move_arm_service = rospy.Service('move_arm', ChangePose, move_arm_to_position)
    do_plan_service = rospy.Service('do_plan', RobotTrajectory, do_plan)
    joint_names_service = rospy.Service('joint_names', JointNames, joint_names)
    current_joint_values_service = rospy.Service('joint_values', JointValues, current_joint_values)
    current_pose_service = rospy.Service('current_pose', CurrentPose,  current_pose)
    current_pose_service = rospy.Service('close_gripper', srv.Empty,  gripper_close)
    current_pose_service = rospy.Service('open_gripper', srv.Empty,  gripper_open)

    rospy.spin()
