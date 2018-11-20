#! /usr/bin/env python
from behavioural_vae.ros_monitor import ROSTrajectoryVAE, RosTrajectoryConvVAE
from behavioural_vae.visual import TrajectoryVisualizer
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.monitor import TrajectoryDemonstrator
import argparse
import matplotlib.pyplot as plt
import os

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

parser.add_argument('--latent-dim', default=5, type=int, help='') # VAE model determines
parser.add_argument('--num_joints', default=7, type=int, help='')
parser.add_argument('--num_actions', default=20, type=int, help='Smoothed trajectory') # VAE model determines
parser.add_argument('--duration', default=4, type=int, help='Duration of generated trajectory')

parser.add_argument('--conv', dest='conv', action='store_true')
parser.add_argument('--no-conv', dest='conv', action='store_false')
parser.set_defaults(conv=False)

parser.add_argument('--conv-channel', default=2, type=int, help='1D conv out channel')
parser.add_argument('--kernel-row', default=4, type=int, help='Size of Kernel window in 1D')

args = parser.parse_args()

model_name = args.vae_name
latent_dim = args.latent_dim
duration = args.duration
num_actions = args.num_actions
num_joints = args.num_joints

MODEL_ROOT = '/home/aleksi/hacks/behavioural_ws/behaviroural_vae/behavioural_vae'

def experiment():

    _, _, positions, results = demo.demonstrate(visualize=True)
    generate_image(positions, results, num_actions)

if __name__ == '__main__':
    simulation_interface = SimulationInterface('lumi_arm')
    if args.conv:
        behaviour_model = RosTrajectoryConvVAE(model_name, latent_dim, num_actions, args.kernel_row, args.conv_channel,
                                               num_joints=num_joints,  root_path=MODEL_ROOT)
    else:
        behaviour_model = ROSTrajectoryVAE(model_name, latent_dim, num_actions,
                                           num_joints=num_joints,  root_path=MODEL_ROOT)
    visualizer = TrajectoryVisualizer(os.path.join(MODEL_ROOT, 'log', model_name))

    demo = TrajectoryDemonstrator(behaviour_model, latent_dim, simulation_interface, num_joints,
                 num_actions, duration, visualizer)
    demo.reset_environment()


#    print("Get Average Error:")
#    demo.multiple_demonstrations(10)
#    print("Random Latent Imitations")
#    demo.generate_random_imitations(5)
#    print("Plottting")
#    demo.generate_multiple_images(30)
