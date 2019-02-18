import numpy as np
from motion_planning.simulation_interface import MCInterface

class HardwareInterface(MCInterface):

    def __init__(self, arm_name):
        super(HardwareInterface, self).__init__(arm_name)
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
    planner = HardwareInterface(arm_name='lumi_arm')
    print(planner.reset(11))
