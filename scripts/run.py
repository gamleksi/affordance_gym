#! /usr/bin/env python
from motion_planning.monitor import TrajectoryDemonstrator
from motion_planning.utils import parse_arguments

args = parse_arguments()

MODEL_NAME = args.model_name
LATENT_DIM = args.latent_dim
NUM_JOINTS = args.num_joints
NUM_ACTIONS = args.num_actions
DURATION = args.duration

if __name__ == '__main__':
        demo = TrajectoryDemonstrator(MODEL_NAME, LATENT_DIM, num_joints=NUM_JOINTS,
                        num_actions=NUM_ACTIONS, trajectory_duration=DURATION)
