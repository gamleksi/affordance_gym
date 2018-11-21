#! /usr/bin/env python
import torch
import torch.nn as nn
import random
import os
import csv

from behavioural_vae.ros_monitor import ROSTrajectoryVAE, RosTrajectoryConvVAE
from behavioural_vae.visual import TrajectoryVisualizer

from motion_planning.policy_gradient import Policy, PolicyGradient
from motion_planning.rl_logger import Logger
from motion_planning.rl_env import SimpleEnvironment

from motion_planning.simulation_interface import SimulationInterface
from motion_planning.monitor import TrajectoryDemonstrator
from motion_planning.utils import parse_arguments, BEHAVIOUR_ROOT, POLICY_ROOT, load_parameters


def use_cuda():

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    return torch.device('cuda' if use_cuda else 'cpu')


def save_arguments(args):
    save_path = os.path.join('log', args.folder_name)
    args = vars(args)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    w = csv.writer(open(os.path.join(save_path, "arguments.csv"), "w"))
    for key, val in args.items():
        w.writerow([key, val])


def gauss_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def get_trajectory_model(args):

    model_name = args.vae_name
    latent_dim = args.latent_dim
    duration = args.duration
    num_actions = args.num_actions
    num_joints = args.num_joints
    simulation_interface = SimulationInterface('lumi_arm')
    if args.conv:
        behaviour_model = RosTrajectoryConvVAE(model_name, latent_dim, num_actions, args.kernel_row, args.conv_channel,
                                               num_joints=num_joints,  root_path=BEHAVIOUR_ROOT)
    else:
        behaviour_model = ROSTrajectoryVAE(model_name, latent_dim, num_actions,
                                           num_joints=num_joints,  root_path=BEHAVIOUR_ROOT)
    visualizer = TrajectoryVisualizer(os.path.join(BEHAVIOUR_ROOT, 'log', model_name))

    demo = TrajectoryDemonstrator(behaviour_model, latent_dim, simulation_interface, num_joints,
                 num_actions, duration, visualizer)
    demo.reset_environment()
    return demo


def main(args):
    save_arguments(args)
    # Trajectory Interface
    trajectory_model = get_trajectory_model(args)
    random_goal = args.random_goal
    env = SimpleEnvironment(trajectory_model, random_goal)

    # Policy
    save_path = os.path.join(POLICY_ROOT, args.folder_name)

    policy = Policy()

    if args.train:
        random.seed(333)
        torch.manual_seed(333)
        gauss_init(policy)
    else:
        load_parameters(policy, save_path)

    logger = Logger(save_path, debug=args.debug) if args.train else None

    device = use_cuda()
    policy.to(device)

    algo = PolicyGradient(env, policy, args.lr, device, logger=logger)

    if args.train:
        algo.run(args.iterations, args.batch_size)
    else:
        algo.eval()
    return algo

if __name__ == '__main__':
    args = parse_arguments(True, True)
    algo = main(args)

