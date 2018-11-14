#! /usr/bin/env python
from behavioural_vae.utils import smooth_trajectory, MIN_ANGLE, MAX_ANGLE
from motion_planning.trajectory_parser import parse_trajectory
from motion_planning.simulation_interface import CommunicationHandler
from behavioural_vae.ros_monitor import ROSTrajectoryVAE

import matplotlib.pyplot as plt
import numpy as np
import os

MODEL_ROOT_PATH = '/home/aleksi/hacks/behavioural_ws/behaviroural_vae/behavioural_vae'


class Visualizer(object):

    def __init__(self, sample_root, model_name):
        self.sample_path = os.path.join(sample_root, model_name)
        if not(os.path.isdir(self.sample_path)):
            os.mkdir(self.sample_path)

    def generate_image(self, original, reconstructed, file_name=None):
        # w = original.shape[0]
        # h = original.shape[1]
        fig = plt.figure(figsize=(30, 30))
        columns = 4
        rows = 1
        images = (original, reconstructed, np.abs(
            original - reconstructed))
        for i in range(1, 4):
            fig.add_subplot(rows, columns, i)
            im = plt.imshow(images[i-1],
                    cmap='gray', vmin=0, vmax=1)
        plt.colorbar(im, fig.add_subplot(rows, columns, 4))
        if file_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(self.sample_path, '{}.png'.format(file_name)))
            plt.close()


class TrajectoryEnv(object):

    def __init__(
            self, model_name,
            latent_dim, env_interface,
            num_joints=7, num_actions=20,
            model_root_path=MODEL_ROOT_PATH, trajectory_duration=4
            ):

        self.env_interface = env_interface
        self.behaviour_model = ROSTrajectoryVAE(
                model_name, latent_dim, num_actions,
                num_joints=num_joints, root_path=model_root_path
                )

        self.msg_handler = CommunicationHandler(trajectory_duration, self.env_interface.current_joint_values(), self.env_interface.joint_names())
        self.num_actions = num_actions
        self.num_joints = num_joints

    def reset_environment(self):

        # Reset Simulation
        # Clear Controller target

        self.env_interface.reset()

    def process_plan(self, plan):

        # Smooth plan to N steps
        # Normalize positions of the smoothed trajectory
        # Returns the normalized trajectory positions which can be feed to a behavioural model

        time_steps_raw, positions_raw, _, _ = parse_trajectory(plan)
        _, positions, _, _ = smooth_trajectory(time_steps_raw, positions_raw, self.num_actions, self.num_joints)
        positions = (positions - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)

        return np.array(positions)

    def unnormalize_positions(self, positions):

        return (MAX_ANGLE - MIN_ANGLE) * positions + MIN_ANGLE

    def smooth_plan(self, plan):

        # Smooth a moveit generated plan to N step plan
        # Return new plan with N step

        positions = self.process_plan(plan)
        smoothed_plan = self.msg_handler.build_message(self.unnormalize_positions(positions),
                                                       )
        return smoothed_plan, positions


    def get_imitation(self, plan):
        # Gets a moveit generated plan
        # Parses normalized smoothed positions of the plan
        # a behavioural vae returns a imitation of the smoothed plan
        # Returns normalized raw trajectory

        positions = self.process_plan(plan)
        result = self.behaviour_model.get_result(positions)
        return result

    def imitate_plan(self, plan):
        # Gets a moveit generated plan
        # Obtain an imitation of the position trajectory
        # does the imitation

        result = self.get_imitation(plan)
        result_plan = self.msg_handler.build_message(self.unnormalize_positions(result))
        self.env_interface.do_plan(result_plan)

        return result, self.env_interface.current_joint_values()

    def get_latent_imitation(self, latent):
        # Returns a normalized position trajectory of a given latent
        return self.behaviour_model.decode(latent)

    def do_latent_imitation(self, latent):
        # Executes a trajectory based on a given latent

        normalized_positions = self.get_latent_imitation(latent)
        plan = self.msg_handler.build_message(self.unnormalize_positions(normalized_positions))
        self.env_interface.do_plan(plan)
        return normalized_positions, self.env_interface.get_current_pose()

    def generate_random_plan(self):
        # Generates a random smoothed trajectory
        # returns a smoothed and normalized position trajectory, and
        # the smoothed plan

        plan = self.env_interface.random_plan()
        positions = self.process_plan(plan)
        smoothed_plan = self.msg_handler.build_message(self.unnormalize_positions(positions))
        return positions, smoothed_plan

    def do_random_plan(self):
        # Executes a random smoothed plan

        positions, smoothed_plan = self.generate_random_plan()
        self.env_interface.do_plan(smoothed_plan)
        end_model_pose = self.env_interface.current_joint_values()

        return positions, smoothed_plan, end_model_pose

    def do_random_raw_plan(self):
        # Executes a random plan

        plan = self.env_interface.random_plan()
        self.env_interface.do_plan(plan)
        end_model_pose = self.env_interface.current_joint_values()

        return end_model_pose


VIS_ROOT = '/home/aleksi/hacks/behavioural_ws/result_samples'
MODEL_ROOT = '/home/aleksi/hacks/behavioural_ws/behaviroural_vae/behavioural_vae'

class TrajectoryDemonstrator(TrajectoryEnv):

    def __init__(self, model_name, latent_dim, env_interface, num_joints,
                 num_actions, trajectory_duration, model_root_path=MODEL_ROOT, vis_root=VIS_ROOT):

        super(TrajectoryDemonstrator, self).__init__(
            model_name, latent_dim,
            env_interface, num_joints, num_actions,
            model_root_path, trajectory_duration
            )

        self.visualizer = Visualizer(vis_root, model_name)
        self.latent_dim = latent_dim

    def log_imitation(self, file_name=None):

        self.reset_environment()
        positions, smoothed_plan = self.generate_random_plan()
        result = self.get_imitation(smoothed_plan)
        self.visualizer.generate_image(positions, result, file_name=file_name)

    def demonstrate(self, visualize=False):

        self.reset_environment()

        # Random smoothed trajectory
        positions, smoothed_plan, end_model_pose = self.do_random_plan()

        self.reset_environment()

        # Generated trajectory
        result, end_gen_pose = self.imitate_plan(smoothed_plan)

        if visualize:
            self.visualizer.generate_image(positions, result)

        return end_model_pose, end_gen_pose

    def multiple_demonstrations(self, num_samples):

        losses = np.zeros(num_samples)

        for i in range(num_samples):
            end_model_pose, end_gen_pose = self.demonstrate(visualize=False)
            loss = np.linalg.norm(np.array(end_model_pose) - np.array(end_gen_pose))
            print(loss)
            losses[i] = loss

        print('AVG LOSS:', np.mean(losses))

    def generate_multiple_images(self, num_samples):

        for i in range(num_samples):
            self.log_imitation('sample_{}'.format(i))

    def generate_random_imitations(self, num_samples):

        for i in range(num_samples):
            self.reset_environment()
            random_latent = np.random.randn(self.latent_dim)
            self.do_latent_imitation(random_latent)
