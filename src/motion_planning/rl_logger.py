import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch


class Logger(object):

    def __init__(self, log_path, debug=False):

        self.rewards = []
        self.losses = []
        self.prev_reward = -np.inf

        self.log_path = log_path

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
