import numpy as np
from affordance_gym.moveit_commander_interface import MCInterface



'''

A Moveit interface for a Panda Franka 

'''


class HardwareInterface(MCInterface):

    def __init__(self, arm_name, gripper_name, velocity_scaling_factor=0.2):
        super(HardwareInterface, self).__init__(arm_name, gripper_name=gripper_name)
        self.arm_planner.set_max_velocity_scaling_factor(velocity_scaling_factor)
        self.reset(11)

    def reset(self, duration):

        self.arm_planner.clear_pose_targets()

        joint_goal = self.current_joint_values()
        joint_goal[0] = 0.
        joint_goal[1] = 0.
        joint_goal[2] = 0.0
        joint_goal[3] = -np.pi / 2
        joint_goal[4] = 0.0
        joint_goal[5] = np.pi / 2
        joint_goal[6] = 0

        self.arm_planner.go(joint_goal, wait=True)
        self.arm_planner.stop()

        plan = self.arm_planner.plan()
        succeed = self.arm_planner.execute(plan, wait=True)
        return succeed

if __name__ == '__main__':

    planner = HardwareInterface('lumi_arm', 'lumi_hand')
    print(planner.reset(11))
