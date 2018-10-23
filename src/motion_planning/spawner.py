import sys
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState


"""
This file is not used.
"""


MODEL_XML =  "/home/aleksi/catkin_ws/src/kuka-lwr/single_lwr_example/single_lwr_robot/robot/single_lwr_robot.urdf"


def set_initial_state():

    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        pause()
    except rospy.ServiceException as e:
        print('/gazebo/pause_physics: {}'.format(e))
        sys.exit(2)

    set_link = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
    rospy.wait_for_service('/gazebo/set_link_state')

    try:
        set_link(LinkState(link_name='lwr_base_link'))
        set_link(LinkState(link_name='lwr_base_link'))
    except rospy.ServiceException as e:
        print('/gazebo/set_link_state: {}'.format(e))
        sys.exit(2)

    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause()
    except rospy.ServiceException as e:
        print('/gazebo/unpause_physics: {}'.format(e))
        sys.exit(2)


#    req = SetModelStateRequest()
#    req.model_state.pose = model_state.pose
#    req.model_state.model_name = model_name
#    req.model_state.reference_frame = ''
#
#    try:
#        srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
#        resp = srv(req)
#    except rospy.ServiceException as e:
#        print("   Service call failed: %s" % e)
#        sys.exit(1)
#    if resp.success:
#        print(resp.status_message)
#        return 0
#    else:
#        print(resp.status_message)
#        return 1


def spawn(model_name="single_lwr_robot", namespace=""):
    req = SpawnModelRequest()
    req.model_name = model_name
    f = open(MODEL_XML)
    model_urdf = f.read()
    req.model_xml = model_urdf
    req.robot_namespace = namespace
    req.initial_pose.position.x = 0.0
    req.initial_pose.position.y = 0.0
    req.initial_pose.position.z = 0.0
    req.initial_pose.orientation.x = 0.0
    req.initial_pose.orientation.y = 0.0
    req.initial_pose.orientation.z = 0.0
    req.initial_pose.orientation.w = 1.0
    req.reference_frame = ''

    try:
        srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp = srv(req)
    except rospy.ServiceException as e:
        print("   Service call failed: %s" % e)
        sys.exit(1)
    if resp.success:
        print(resp.status_message)
        return 0
    else:
        print(resp.status_message)
        return 1

def delete(model_name="single_lwr_robot", namespace=""):
    try:
        srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        req = DeleteModelRequest()
        req.model_name = model_name
        res = srv(req)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        sys.exit(2)
    if res.success:
        print(res.status_message)
    else:
        print(res.status_message)

        return 1

if __name__ == '__main__':
    set_initial_state()
