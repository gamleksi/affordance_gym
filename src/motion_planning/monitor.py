#! /usr/bin/env python
from behavioural_vae.utils import smooth_trajectory, MIN_ANGLE, MAX_ANGLE
from motion_planning.controller import RobotScene, RobotWrapper
from motion_planning.controller import RobotTrajectoryHandler
from motion_planning.trajectory_parser import parse_trajectory
from motion_planning.utils import KUKA_RESET_JOINTS, KUKA_X_LIM
from motion_planning.utils import KUKA_Y_LIM, KUKA_z_LIM


from behavioural_vae.ros_monitor import ROSTrajectoryVAE
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT_PATH = '/home/aleksi/hacks/behavioural_ws/behaviroural_vae/behavioural_vae'


class Visualizer(object):

    def __init__(self, sample_root):
        self.sample_root = sample_root

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
            plt.savefig(os.path.join('/home/aleksi/hacks/behavioural_ws/result_samples', '{}.png'.format(file_name)))
            plt.close()


class ROSMonitor(object):

    def __init__(
            self, model_name,
            latent_dim, arm_group='full_lwr',
            num_joints=7, num_actions=20,
            model_root_path=ROOT_PATH, trajectory_duration=4
            ):

        robot, env, model, handler = self.build_environment(
                 model_name, latent_dim, arm_group, num_joints,
                 num_actions, model_root_path, trajectory_duration
                 )
        self.robot = robot
        self.env = env
        self.model = model
        self.handler = handler

    def build_environment(
            self, model_name, latent_dim,
            arm_group, num_joints, num_actions,
            model_root_path, trajectory_duration):
        robot = RobotWrapper(arm_group, KUKA_RESET_JOINTS)
        env = RobotScene(robot, KUKA_X_LIM, KUKA_Y_LIM, KUKA_z_LIM)
        model = ROSTrajectoryVAE(
                model_name, latent_dim, num_actions,
                num_joints=num_joints, root_path=model_root_path
                )
        handler = RobotTrajectoryHandler(trajectory_duration, KUKA_RESET_JOINTS)
        return robot, env, model, handler

    def process_plan(self, plan):
        # return smoothed trajectories
        time_steps_raw, positions_raw, _, _ = parse_trajectory(plan)
        _, positions, _, _ = smooth_trajectory(time_steps_raw, positions_raw, NUM_ACTIONS, NUM_JOINTS)
        positions = (positions - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)
        return np.array(positions)

    def unnormalize_positions(self, positions):
        return (MAX_ANGLE - MIN_ANGLE) * positions + MIN_ANGLE

    def smooth_plan(self, plan):
        positions = self.process_plan(plan)
        smoothed_plan = self.handler.build_message(self.unnormalize_positions(positions))
        return smoothed_plan, positions

    # Behavioural model trajectories

    def get_imitation(self, plan):
        positions = self.process_plan(plan)
        result = self.model.get_result(positions)
        return result

    def imitate_plan(self, plan):
        result = self.get_imitation(plan)
        result_plan = self.handler.build_message(self.unnormalize_positions(result))
        self.robot.reset_position()
        self.env.do_plan(result_plan)
        return result, self.env.robot.arm.get_current_joint_values()

    def get_latent_imitation(self, latent):
        result = self.model.decode(latent)
        return result

    def do_latent_imitation(self, latent):
        result = self.get_latent_imitation(latent)
        result_plan = self.handler.build_message(self.unnormalize_positions(result))
        self.robot.reset_position()
        self.env.do_plan(result_plan)
        return result, self.env.robot.arm.get_current_pose()

    # MOVEIT Random Trajectories

    def generate_random_plan(self):
        # generate random smoothed trajectory
        # return smoothed trajectory and plan and end pose
        self.env.reset_position()
        plan = self.env.random_plan()
        positions = self.process_plan(plan)
        smoothed_plan = self.handler.build_message(self.unnormalize_positions(positions))
        return positions, smoothed_plan

    def do_random_plan(self):
        # Act random smoothed trajectory
        # return smoothed trajectory, plan and end pose
        positions, smoothed_plan = self.generate_random_plan()
        self.env.do_plan(smoothed_plan)
        end_model_pose = self.env.robot.arm.get_current_joint_values()
        self.env.reset_position()
        return positions, smoothed_plan, end_model_pose


VIS_ROOT = '/home/aleksi/hacks/behavioural_ws/result_samples'
MODEL_ROOT = '/home/aleksi/hacks/behavioural_ws/behaviroural_vae/behavioural_vae'

class TrajectoryDemonstrator(ROSMonitor):

    def __init__(
            self, model_name, latent_dim, viz_root=VIS_ROOT,
            arm_group='full_lwr', num_joints=7, num_actions=20,
            model_root_path=MODEL_ROOT, trajectory_duration=4):

        super(TrajectoryDemonstrator, self).__init__(
            model_name, latent_dim, arm_group,
            num_joints, num_actions, model_root_path, trajectory_duration
            )

        self.visualizer = Visualizer(viz_root)
        self.latent_dim = latent_dim

    def log_imitation(self, file_name=None):
        positions, smoothed_plan = self.generate_random_plan()
        result = self.get_imitation(smoothed_plan)
        self.visualizer.generate_image(positions, result, file_name=file_name)

    def demonstrate(self, visualize=False):

        # Random smoothed trajectory
        positions, smoothed_plan, end_model_pose = self.do_random_plan()

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
            self.log_imitation('sample_{i}')

    def generate_random_imitations(self, num_samples):

        for i in range(num_samples):
            random_latent = np.random.randn(self.latent_dim)
            self.do_latent_imitation(random_latent)

