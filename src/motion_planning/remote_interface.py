import rospy
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image
import tf
from std_srvs import srv
from motion_planning.srv import RobotTrajectory, ChangePose, JointValues, JointNames, CurrentPose

class RemoteMCInterface(object):

   def gripper_open(self):
       return NotImplementedError

   def current_joint_values(self):

       rospy.wait_for_service('joint_values')

       try:
           joint_names_service = rospy.ServiceProxy('joint_values', JointValues)
           response = joint_names_service()
           values = response.values
       except rospy.ServiceException as exc:
           print("current_joint_values failed:" + str(exc))
           values = None

       return values

   def joint_names(self):

       rospy.wait_for_service('joint_names')

       try:
           joint_names_service = rospy.ServiceProxy('joint_names', JointNames)
           response = joint_names_service()
           names = response.names
       except rospy.ServiceException as exc:
           print("joint_names failed:" + str(exc))
           names = None

       return names

   def current_pose(self):

       rospy.wait_for_service('current_pose')
       try:
           current_pose_values = rospy.ServiceProxy('current_pose', CurrentPose)
           response = current_pose_values()
           pose = response.pose
       except rospy.ServiceException as exc:
           print("Current pose failed:" + str(exc))
           pose = None
       return pose

   def gripper_close(self):
       return NotImplementedError

   def print_current_pose(self):
       return NotImplementedError

   def print_current_joint_states(self):
       return NotImplementedError

   def plan_end_effector_to_position(self, x_p=0.5, y_p=0, z_p=0.5,  roll_rad=0, pitch_rad=np.pi, yaw_rad=np.pi):
       return NotImplementedError

   def move_arm_to_position(
           self, x_p=0.5, y_p=0, z_p=0.5,  roll_rad=0, pitch_rad=np.pi, yaw_rad=np.pi):
       rospy.wait_for_service('move_arm')
       try:
           do_plan_service = rospy.ServiceProxy('move_arm', ChangePose)
           response = do_plan_service(x_p, y_p, z_p)
           success = response.success
       except rospy.ServiceException as exc:
           success = False
           print("Move arm failed:" + str(exc))
       return success


   def do_plan(self, plan):
       rospy.wait_for_service('do_plan')
       try:
           do_plan_service = rospy.ServiceProxy('do_plan', RobotTrajectory)
           response = do_plan_service(plan)
           success = response.success
       except rospy.ServiceException as exc:
           success = False
           print("Plan failed:" + str(exc))

       return success


   def reset(self, duration):
       rospy.wait_for_service('reset')
       try:
           reset_service = rospy.ServiceProxy('reset', srv.Empty)
           reset_service()
       except rospy.ServiceException as exc:
           print("Reset did not work:" + str(exc))
       rospy.sleep(duration)

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


if __name__  == '__main__':
    remote = RemoteMCInterface()
    remote.move_arm_to_position(0.4, 0.0, 0.3)
    print(remote.reset(0))
    print(remote.joint_names())
    print(remote.current_joint_values())
