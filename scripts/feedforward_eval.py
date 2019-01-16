import os
import torch
from behavioural_vae.ros_monitor import ROSTrajectoryVAE, RosTrajectoryConvVAE
from behavioural_vae.latent_predicor import Predictor
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.utils import parse_arguments, BEHAVIOUR_ROOT, load_parameters, use_cuda, LUMI_X_LIM, LUMI_Y_LIM
from motion_planning.monitor import TrajectoryEnv
from motion_planning.rl_env import SimpleEnvironment
from motion_planning.policy_gradient import PolicyGradient
import numpy as np
from torch.autograd import Variable
np.random.seed(10)

def main(args):

    device = use_cuda()

    model_index = args.model_index

    if args.conv:
        behaviour_model = RosTrajectoryConvVAE(args.vae_name, args.latent_dim, args.num_actions, args.kernel_row, args.conv_channel)
    else:

        behaviour_model = ROSTrajectoryVAE(args.vae_name, args.latent_dim, args.num_actions,
                                           model_index=model_index, num_joints=args.num_joints,  root_path=BEHAVIOUR_ROOT)

    simulation_interface = SimulationInterface('lumi_arm')
    trajectory_model = TrajectoryEnv(behaviour_model, simulation_interface, args.num_actions, num_joints=args.num_joints, trajectory_duration=args.duration)
    trajectory_model.reset_environment(3)

    env = SimpleEnvironment(trajectory_model, args.random_goal, device)

    policy = Predictor(args.latent_dim)
    policy.to(device)
    load_parameters(policy, os.path.join(BEHAVIOUR_ROOT, 'pred_log', args.policy_name), 'model')

   # algo = PolicyGradient(env, policy, 1e-3, device)

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
            # print("end_pose abs", np.abs(end_pose[:2] - goal[:2]))

if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, feedforward=True)
    main(args)