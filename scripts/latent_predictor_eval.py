import os
import torch
from TrajectoryVAE.ros_monitor import ROSTrajectoryVAE
from TrajectoryVAE.latent_predicor import Predictor

from affordance_gym.simulation_interface import SimulationInterface
from affordance_gym.utils import load_parameters, use_cuda
from affordance_gym.monitor import TrajectoryEnv
from affordance_gym.rl_env import SimpleEnvironment

from env_setup.env_setup import TRAJ_MODELS_PATH, LUMI_X_LIM, LUMI_Y_LIM

import numpy as np
import rospy
from torch.autograd import Variable
np.random.seed(10)


'''
For debugging: how well the trajecory VAE (the latent space and decoder) works without the perception part. 

Task of the policy: a goal end effector position -> a latent action -> a trajectory 

Trained: in TrajecotyVAE's latent_predictor.py with a generated trajectory dataset. 

'''


def main(args):

    device = use_cuda()

    rospy.init_node('talker', anonymous=True)

    behaviour_model =  ROSTrajectoryVAE(os.path.join(TRAJ_MODELS_PATH, args.vae_name), args.latent_dim, args.num_actions, model_index=args.model_index, num_joints=args.num_joints)

    simulation_interface = SimulationInterface('lumi_arm')
    trajectory_model = TrajectoryEnv(behaviour_model, simulation_interface, args.num_actions, num_joints=args.num_joints, trajectory_duration=args.duration)
    trajectory_model.reset_environment(3)

    env = SimpleEnvironment(trajectory_model, args.random_goal, device)

    policy = Predictor(args.latent_dim)
    policy.to(device)
    load_parameters(policy, os.path.join(TRAJ_MODELS_PATH, 'pred_log', args.policy_name), 'model')

    rewards = np.zeros(args.num_steps)

    def do_action(state):

        latent_action = policy(Variable(state))
        latent_action = latent_action.data[0].cpu().numpy()
        end_pose = env.do_action(latent_action)
        return end_pose

    if(args.random_goal):
        for i in range(args.num_steps):
            _, goal = env.get_state()

            state = torch.from_numpy(goal).float().unsqueeze(0).to(device)
            end_pose = do_action(state)
            reward = env.get_reward(goal, end_pose - np.array([-0.4, 0.15, 0.0]), train=False)
            rewards[i] = reward
            print('Reward: {}'.format(reward))
            print("goal", goal)
            print("end_pose", end_pose - np.array([-0.4, 0.15, 0.0]))
            env.reset()

        print("AVG: ", rewards.mean(), " VAR: ", rewards.var())

    if args.debug:

        goals = [(LUMI_X_LIM[0] + 0.1, LUMI_Y_LIM[0] + 0.1), (LUMI_X_LIM[1], LUMI_Y_LIM[0] + 0.1),
                 (LUMI_X_LIM[0] + 0.1, LUMI_Y_LIM[1] - 0.15), (LUMI_X_LIM[1], LUMI_Y_LIM[1] - 0.15)]

        for goal in goals:

            state = torch.Tensor(goal).unsqueeze(0).to(device)
            end_pose = do_action(state)
            reward = env.get_reward(goal, end_pose - np.array([-0.4, 0.15, 0.0]), train=False)
            print("goal", goal)
            print("end_pose", end_pose - np.array([-0.4, 0.15, 0.0]))
            print('Reward: {}'.format(reward))
            env.reset()


if __name__ == '__main__':

  #  args = parse_arguments(behavioural_vae=True, feedforward=True)
  #  main(args)

  print("This does not work")