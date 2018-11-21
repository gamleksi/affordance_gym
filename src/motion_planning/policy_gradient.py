import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal
from motion_planning.utils import print_pose


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
                self.env.reset()

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
