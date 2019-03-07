import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image
import tf
import geometry_msgs
import cv_bridge
import moveit_commander as mc


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


GRIPPER_OPEN_VALUES = (0.04, 0.04)


class MCInterface(object):

    def __init__(self, arm_name, gripper_name=None, planning_id='RRT'):

        mc.roscpp_initialize(sys.argv)

        self.robot = mc.RobotCommander()

        self.arm_planner = self.build_planning_interface(arm_name, planning_id)

        if gripper_name is not None:
            self.gripper_planner = mc.MoveGroupCommander(gripper_name)
            self.gripper_open_values = GRIPPER_OPEN_VALUES
        else:
            self.gripper_planner = None


    def build_planning_interface(self, name, planning_id):

        arm = mc.MoveGroupCommander(name)
        arm.set_planner_id(planning_id)
        arm.allow_replanning(False)
        arm.set_goal_position_tolerance(0.0005)
        arm.set_goal_orientation_tolerance(0.005)

        return arm

    def gripper_open(self):

        if self.gripper_planner is not None:
            self.gripper_planner.set_joint_value_target(self.gripper_open_values)
            self.gripper_planner.plan()
            self.gripper_planner.go(wait=True)

    def current_joint_values(self):
        return self.arm_planner.get_current_joint_values()

    def joint_names(self):
        return self.arm_planner.get_joints()

    def current_pose(self):
        current_pose = self.arm_planner.get_current_pose()
        return [current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z]

    def gripper_close(self):
        if self.gripper_planner is not None:
            import pdb; pdb.set_trace()
            self.gripper_planner.set_joint_value_target([0.01, 0.01])
            self.gripper_planner.plan()
            self.gripper_planner.go(wait=True)

    def print_current_pose(self):
        pose = self.arm_planner.get_current_pose()
        print('Current Pose:')
        print(pose)
        print(self.arm_planner.get_current_rpy())

    def print_current_joint_states(self):
        print('Current Joint Values:')
        print(self.arm_planner.get_current_joint_values())

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
        self.arm_planner.stop()
        return succeed

    def reset(self, duration):
        return NotImplementedError

    def capture_image(self, topic):

        try:
            image_msg = rospy.wait_for_message(topic, Image)
            img = cv_bridge.CvBridge().imgmsg_to_cv2(image_msg, "rgb8")
            img_arr = np.uint8(img)
            return img_arr
        except rospy.exceptions.ROSException as e:
            print(e)
            return None

    def kinect_camera_pose(self):

        listener = tf.TransformListener()
        try:
            listener.waitForTransform('/base_link', '/camera_rgb_frame', rospy.Time(0), rospy.Duration(5))
            (trans, rot) = listener.lookupTransform('/base_link', '/camera_rgb_frame', rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print(e)
            return None, None


if __name__ == '__main__':
    import rospy
    rospy.init_node('talker', anonymous=True)
    planner = MCInterface('panda_arm', 'hand')
    planner.gripper_close()
    planner.gripper_open()
