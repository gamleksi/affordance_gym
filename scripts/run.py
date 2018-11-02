#! /usr/bin/env python
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.monitor import TrajectoryDemonstrator
from motion_planning.utils import parse_arguments

args = parse_arguments()

MODEL_NAME = args.model_name
LATENT_DIM = args.latent_dim
NUM_JOINTS = args.num_joints
NUM_ACTIONS = args.num_actions
DURATION = args.duration

if __name__ == '__main__':
    simulation_interface = SimulationInterface('arm')
    demo = TrajectoryDemonstrator(MODEL_NAME, LATENT_DIM, simulation_interface, NUM_JOINTS,
                                  NUM_ACTIONS, DURATION)

#    demo.generate_multiple_images(40)
#    print("Random_plan 1")
#    demo.reset_environment()
#    demo.do_random_plan()
#    print("Random_plan 2")
#    demo.reset_environment()
#    demo.do_random_plan()
#    print("Random_plan 3")
#    demo.do_random_plan()
#    print("Random_plan 4")
#    demo.reset_environment()
#    demo.do_random_plan()


#    print("Demonstrattion 1")
#    demo.demonstrate()
#    print("Demonstrattion 2")
#    demo.demonstrate()
#    print("Demonstrattion 3")
#    demo.demonstrate()
#    print("Demonstrattion 4")
#    demo.demonstrate()
