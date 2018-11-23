#! /usr/bin/env python
from behavioural_vae.ros_monitor import ROSTrajectoryVAE, RosTrajectoryConvVAE
from behavioural_vae.visual import TrajectoryVisualizer
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.monitor import TrajectoryDemonstrator
from motion_planning.utils import parse_arguments, BEHAVIOUR_ROOT
import os

def main(args):

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
    demo.reset_environment(3.0)
    return demo

if __name__ == '__main__':

    args = parse_arguments(True, False)
    demo = main(args)
    demo.generate_random_imitations(5)

#    print("Get Average Error:")
#    demo.multiple_demonstrations(10)
#    print("Random Latent Imitations")
#    print("Plottting")
#    demo.generate_multiple_images(30)
