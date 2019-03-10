import numpy as np
from env_setup.env_setup import LUMI_X_LIM, LUMI_Y_LIM


'''

This file is only used in the latent predictor eval experiment (for debugging)

'''


class SimpleEnvironment(object):

    def __init__(self, trajectory_model, random_goal, device):
        self.trajectory_model = trajectory_model
        self.random_goal = random_goal

    def get_state(self):

        if self.random_goal:
            x = np.random.uniform(LUMI_X_LIM[0] + 0.1, LUMI_X_LIM[1])
            y = np.random.uniform(LUMI_Y_LIM[0], LUMI_Y_LIM[1])
        else:
            x = (LUMI_X_LIM[0] + 0.1 + LUMI_X_LIM[1]) / 2.
            y = (LUMI_Y_LIM[0] + LUMI_Y_LIM[1]) / 2.

        return np.array([x, y]), np.array([x, y])

    def get_reward(self, goal, end_pose, train=False):

        if train:
            norm_denumerator = np.array([1. / (LUMI_X_LIM[1] - LUMI_X_LIM[0]), 1. / (LUMI_Y_LIM[1] - LUMI_Y_LIM[0])])
            diff = (goal[:2] - end_pose[:2]) * norm_denumerator
            return - np.sum((diff) ** 2)
        else:
            return np.linalg.norm(goal[:2] - end_pose[:2])

    def do_action(self, action):
        _, achieved_pose = self.trajectory_model.do_latent_imitation(
            action)
        end_pose = np.array((
            achieved_pose.pose.position.x,
            achieved_pose.pose.position.y,
            achieved_pose.pose.position.z))
        return end_pose

    def reset(self):
        self.trajectory_model.reset_environment()
