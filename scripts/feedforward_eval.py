import os
import torch
from behavioural_vae.ros_monitor import ROSTrajectoryVAE, RosTrajectoryConvVAE
from behavioural_vae.latent_predicor import Predictor
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.utils import parse_arguments, BEHAVIOUR_ROOT, load_parameters, use_cuda
from motion_planning.monitor import TrajectoryEnv
from motion_planning.rl_env import SimpleEnvironment
from motion_planning.policy_gradient import PolicyGradient
import numpy as np


def main(args):

    random_goal = args.random_goal
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

    env = SimpleEnvironment(trajectory_model, random_goal, device)

    policy = Predictor(args.latent_dim)
    policy.to(device)
    load_parameters(policy, os.path.join(BEHAVIOUR_ROOT, 'pred_log', args.policy_name), 'model')

    algo = PolicyGradient(env, policy, 1e-3, device)

    rewards = np.zeros(args.num_steps)
    for i in range(args.num_steps):
        rewards[i] = algo.eval()
    print("AVG: ", rewards.mean(), " VAR: ", rewards.var())

if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, feedforward=True)
    main(args)