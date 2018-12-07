import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import pandas as pd


class Logger(object):

    def __init__(self, log_path):

        self.rewards = []
        self.losses = []
        self.prev_reward = -np.inf

        self.log_path = log_path

    def visualize_rewards(self, window=50):

        # https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
        fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
        rolling_mean = pd.Series(self.rewards).rolling(window).mean()
        std = pd.Series(self.rewards).rolling(window).std()
        ax1.plot(rolling_mean)
        ax1.fill_between(range(len(self.rewards)), rolling_mean - std, rolling_mean + std, color='orange',
                         alpha=0.2)
        ax1.set_title('Negative Distance to the Goal Coordinate Moving Average ({}-episode window)'.format(window))
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Distance')

        ax2.plot(self.rewards)
        ax2.set_title('Negative distance to the goal coordinate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Distance')
        fig.tight_layout(pad=2)
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
