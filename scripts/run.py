#! /usr/bin/env python
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.monitor import TrajectoryDemonstrator
import argparse
import matplotlib.pyplot as plt

def generate_image(original, reconstructed, num_actions):
    fig = plt.figure(figsize=(30, 30))
    columns = 1
    rows = 7
    steps = range(1, num_actions + 1)
    for i in range(rows):
        fig.add_subplot(rows, columns, i + 1)
        plt.plot(steps, original[:, i], 'ro', steps, reconstructed[:, i], 'bo')
        plt.legend('Original', 'Reconstructed')
    plt.show()


parser = argparse.ArgumentParser(description='Behavioral VAE demonstration')
parser.add_argument('--vae-name', default='simple_full_b-5', type=str, help='')
parser.add_argument('--simple-model', default='simple_full_b-5', type=str, help='')

parser.add_argument('--latent-dim', default=4, type=int, help='') # VAE model determines
parser.add_argument('--num_joints', default=7, type=int, help='')
parser.add_argument('--num_actions', default=20, type=int, help='Smoothed trajectory') # VAE model determines
parser.add_argument('--duration', default=4, type=int, help='Duration of generated trajectory')

args = parser.parse_args()

MODEL_NAME = args.vae_name
LATENT_DIM = args.latent_dim
DURATION = args.duration
NUM_ACTIONS = args.num_actions
NUM_JOINTS = args.num_joints

def experiment():

    _, _, positions, results = demo.demonstrate(visualize=True)
    generate_image(positions, results, NUM_ACTIONS)

if __name__ == '__main__':
    simulation_interface = SimulationInterface('lumi_arm')
    demo = TrajectoryDemonstrator(MODEL_NAME, LATENT_DIM, simulation_interface, NUM_JOINTS,
                                  NUM_ACTIONS, DURATION)
    demo.reset_environment()




#    print("Get Average Error:")
#    demo.multiple_demonstrations(10)
#    print("Random Latent Imitations")
#    demo.generate_random_imitations(5)
#    print("Plottting")
#    demo.generate_multiple_images(30)
