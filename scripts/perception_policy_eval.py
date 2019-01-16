import os
import numpy as np
from motion_planning.simulation_interface import SimulationInterface
from motion_planning.perception_policy import Predictor
from behavioural_vae.ros_monitor import ROSTrajectoryVAE
from gibson.ros_monitor import RosPerceptionVAE

from motion_planning.utils import parse_arguments, GIBSON_ROOT, load_parameters, BEHAVIOUR_ROOT, use_cuda
from motion_planning.utils import LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM
from motion_planning.monitor import TrajectoryEnv


def main(args):

    device = use_cuda()

    # Trajectory generator
    # TODO insserst both encoder and decoder to GPU
    action_vae = ROSTrajectoryVAE(args.vae_name, args.latent_dim, args.num_actions,
                                       model_index=args.model_index, num_joints=args.num_joints,  root_path=BEHAVIOUR_ROOT)
    # pereception
    # TODO insserst both encoder and decoder to GPU
    perception = RosPerceptionVAE(args.g_name, args.g_latent, root_path=GIBSON_ROOT)

    # Policy
    policy = Predictor(args.g_latent, args.latent_dim)
    policy.to(device)
    policy_path = os.path.join('../policy_log', args.policy_name)
    load_parameters(policy, policy_path, 'model')

    # Simulation interface
    sim = SimulationInterface(arm_name='lumi_arm')
    sim.change_camere_params(LOOK_AT, DISTANCE, AZIMUTH, ELEVATION)
    env = TrajectoryEnv(action_vae, sim, args.num_actions, num_joints=args.num_joints)

    rewards = []

    for idx in range(10):
        x = np.random.uniform(CUP_X_LIM[0], CUP_X_LIM[1])
        y = np.random.uniform(CUP_Y_LIM[0], CUP_Y_LIM[1])

        env.change_cup_position(x, y)
        image = env.get_image()
        # affordance, sample = perception.reconstruct(image) TODO sample visualize
        latent1 = perception.get_latent(image)
        latent2 = policy(latent1)
        latent2 = latent2.detach().cpu().numpy() # TODO fix this!
        _, end_pose = env.do_latent_imitation(latent2[0])
        end_pose = np.array((
            end_pose.pose.position.x,
            end_pose.pose.position.y))
        reward = np.linalg.norm(np.array([x, y]) - end_pose)
        rewards.append(reward)
        env.reset_environment(duration=2.0)

        print('Reward: {}'.format(reward))
        print("goal", x, y)
        print("end_pose", end_pose)

    print("AVG: ", np.mean(rewards), " VAR: ", np.var(rewards))


if __name__ == '__main__':
    args = parse_arguments(behavioural_vae=True, gibson=True, feedforward=True)
    main(args)
