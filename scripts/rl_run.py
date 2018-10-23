#! /usr/bin/env python
from motion_planning.monitor import TrajectoryDemonstrator
from motion_planning.utils import KUKA_X_LIM, KUKA_Y_LIM, KUKA_z_LIM
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal
import random
import os
import csv
import argparse
import matplotlib.pyplot as plt


class Logger(object):

    def __init__(self, model_name, root_path, debug=False):

        self.rewards = []
        self.losses = []
        self.prev_reward = -np.inf

        self.log_path = os.path.join(root_path, model_name)

        if not(debug):
            assert(not(os.path.exists(self.log_path)))
            os.makedirs(self.log_path)
        elif not(os.path.exists(self.log_path)):
            os.makedirs(self.log_path)

    def visualize_rewards(self):
        plt.figure()
        plt.plot(self.rewards)
        plt.savefig(os.path.join(self.log_path, 'rewards.png'))
        plt.close()

    def visualize_losses(self):
        plt.figure()
        plt.plot(self.losses)
        plt.savefig(os.path.join(self.log_path, 'loss.png'))
        plt.close()

    def update_rewards(self, reward):

        self.rewards.append(reward)
        self.visualize_rewards()

        row = {'rewards': reward}

        csv_path = os.path.join(self.log_path, 'rewards.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.DictWriter(
                    f,
                    delimiter=',',
                    fieldnames=['rewards'])

            if not(file_exists):
                writer.writeheader()

            writer.writerow(row)

    def update_losses(self, loss):

        self.losses.append(loss)
        self.visualize_losses()

        row = {'losses': loss}

        csv_path = os.path.join(self.log_path, 'losses.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.DictWriter(
                    f,
                    delimiter=',',
                    fieldnames=['losses'])

            if not(file_exists):
                writer.writeheader()

            writer.writerow(row)

    def update_actions(self, locs, scales):

        row = {'locs': locs, 'scales': scales}

        csv_path = os.path.join(self.log_path, 'actions.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.DictWriter(
                    f,
                    delimiter=',',
                    fieldnames=['locs', 'scales'])

            if not(file_exists):
                writer.writeheader()

            writer.writerow(row)

    def update_model(self, reward, model):
        if self.prev_reward < reward:
            self.prev_reward = reward
            torch.save(
                    model.state_dict(),
                    os.path.join(self.log_path, 'rl_model.pth.tar')
                    )


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(3, 32)
        self.fc2_loc = nn.Linear(32, 5)
        self.fc2_scale = nn.Linear(32, 5)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):

        x = self.relu(self.fc1(x))

        loc = self.fc2_loc(x)
        scale = self.softplus(self.fc2_scale(x))

        return loc, scale


def gauss_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(0.0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def load_parameters(
        policy, model_name,
        root_path='/home/aleksi/catkin_ws/src/motion_planning/rl_log'
        ):
    model_path = os.path.join(root_path, model_name, 'rl_model.pth.tar')
    policy.load_state_dict(torch.load(model_path))
    policy.eval()


def print_pose(pose, tag='Pose'):
    print(
            "{}: x: {}, y: {}, z: {}".format(tag, pose[0], pose[1], pose[2]))


class PolicyGradient(object):

    def __init__(self, env, policy, lr, device, logger=None):
        self.env = env
        self.policy = policy
        self.logger = logger
        self.device = device

        # Optimization
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.initialize_iteration()

    def initialize_iteration(self):
        self.action_probs = []
        self.rewards = []

    def select_action(self, state, train=True):

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        locs, scales = self.policy.forward(Variable(state))

        if train:
            self.logger.update_actions(locs, scales)
            distribution = Normal(locs, scales)
            action = distribution.sample()
            probs = distribution.log_prob(action)
            self.action_probs.append(-torch.sum(probs))
        else:
            action = locs

        return action.data[0].cpu().numpy()

    def review_iteration(self):

        rewards = torch.FloatTensor(self.rewards).to(self.device)
        rewards_mean = torch.mean(rewards)
        rewards_std = torch.std(rewards)
        normalized_rewards = (rewards - rewards_mean) / (rewards_std + 1.e-7)

        action_neg_log_probs = self.action_probs
        loss = 0.

        for r, p in zip(normalized_rewards, action_neg_log_probs):
            loss += r * p

        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

    def step(self, train=True):

        state, goal_pose = self.env.get_state()
        action = self.select_action(state, train=train)
        end_pose = self.env.do_action(action)
        reward = self.env.get_reward(goal_pose, end_pose)
        return reward, end_pose, goal_pose

    def run(self, num_iterations, batch_size):

        for i in range(1, num_iterations + 1):
            print("===========================")
            self.initialize_iteration()
            iter_reward = 0.
            for s in range(batch_size):
                reward, end_pose, goal_pose = self.step()
                self.rewards.append(reward)
                iter_reward += reward
                print('Reward: {}'.format(reward))
                print_pose(goal_pose, tag='GOAL')
                print_pose(end_pose, tag='Result')

            iter_reward = iter_reward / batch_size
            loss = self.review_iteration() / batch_size
            if self.logger:
                self.logger.update_rewards(iter_reward)
                self.logger.update_losses(loss)
                self.logger.update_model(iter_reward, self.policy)
            print('Episode: {}, loss: {}, reward: {}' .format(
                i, loss, iter_reward))

    def eval(self):
        reward, end_pose, goal_pose = self.step(train=False)
        print('Reward: {}'.format(reward))
        print_pose(goal_pose, tag='GOAL')
        print_pose(end_pose, tag='Result')


class SimpleEnvironment(object):

    def __init__(self, trajectory_model, random_goal):
        self.trajectory_model = trajectory_model
        self.random_goal = random_goal

    def get_state(self):

        if self.random_goal:
            x = random.uniform(KUKA_X_LIM[0], KUKA_X_LIM[1])
            y = random.uniform(KUKA_Y_LIM[0], KUKA_Y_LIM[1])
            z = random.uniform(KUKA_z_LIM[0], KUKA_z_LIM[1])
        else:
            x = (KUKA_X_LIM[0] + KUKA_X_LIM[1]) / 2.
            y = (KUKA_Y_LIM[0] + KUKA_Y_LIM[1]) / 2.
            z = (KUKA_z_LIM[0] + KUKA_z_LIM[1]) / 2.

        return np.array([x, y, z]), np.array([x, y, z])

    def get_reward(self, goal, end_pose):
        reward = -np.linalg.norm(goal - end_pose)
        return reward

    def do_action(self, action):
        _, achieved_pose = self.trajectory_model.do_latent_imitation(
                action)
        end_pose = np.array((
            achieved_pose.pose.position.x,
            achieved_pose.pose.position.y,
            achieved_pose.pose.position.z))
        return end_pose


parser = argparse.ArgumentParser(description='RL AGENT parser')

parser.add_argument(
        '--num-joints', default=7, type=int, help='')
parser.add_argument('--num-actions', default=20, type=int, help='')
parser.add_argument('--latent-dim', default=5, type=int, help='')
parser.add_argument(
        '--duration', default=4,
        type=int, help='Duratoin of generated trajectory')
parser.add_argument(
        '--trajectory-model',
        default='simple_full_b-5',
        type=str, help='')

# Policy Gradient
parser.add_argument(
        '--save_path',
        default='/home/aleksi/catkin_ws/src/motion_planning/rl_log',
        type=str, help='root path'
        )
parser.add_argument('--rl-model', default='rl_model-v1', type=str, help='')
parser.add_argument(
        '--iterations',
        default=1000,
        type=int
        )
parser.add_argument(
        '--batch-size',
        default=5,
        type=int
        )

parser.add_argument(
        '--lr',
        default=1.e-3,
        type=float
        )

parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)
parser.add_argument('--random-goal', dest='random_goal', action='store_true')
parser.set_defaults(random_goal=False)
parser.add_argument('--test', dest='train', action='store_false')
parser.set_defaults(train=True)
args = parser.parse_args()


def main(args):
    trajectory_model = args.trajectory_model
    latent_dim = args.latent_dim
    num_joints = args.num_joints
    num_actions = args.num_actions
    duration = args.duration
    # robot_id = 1

    trajectory_model = TrajectoryDemonstrator(
            trajectory_model, latent_dim,
            num_joints=num_joints,
            num_actions=num_actions, trajectory_duration=duration)

    random_goal = args.random_goal
    env = SimpleEnvironment(trajectory_model, random_goal)

    train = args.train
    root_path = args.save_path
    rl_name = args.rl_model

    use_cuda = torch.cuda.is_available()

    policy = Policy()
    if train:
        random.seed(333)
        torch.manual_seed(333)
        gauss_init(policy)
        debug = args.debug
        logger = Logger(rl_name, root_path, debug=debug)

    else:
        load_parameters(policy, rl_name, root_path)

    device = torch.device('cuda' if use_cuda else 'cpu')
    policy.to(device)

    algo = PolicyGradient(
            env, policy, args.lr, device, logger=logger if train else None)

    if train:
        algo.run(args.iterations, args.batch_size)
    else:
        algo.eval()
    return algo

if __name__ == '__main__':
    algo = main(args)

