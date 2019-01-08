import numpy as np
import os
import pickle

def parse_trajectory(trajectory):

    trajectory = trajectory.joint_trajectory.points

    time_steps_raw = np.array([motion.time_from_start.to_sec() for motion in trajectory])
    positions_raw = np.stack(np.array(motion.positions) for motion in trajectory)
    velocities_raw = np.stack(np.array(motion.velocities) for motion in trajectory)
    accelerations_raw = np.stack(np.array(motion.accelerations) for motion in trajectory)

    return time_steps_raw, positions_raw, velocities_raw, accelerations_raw

class TrajectoryParser(object):

    def __init__(self, save_path, save_file, num_joints):

        self.save_path = save_path

        if not(os.path.exists(self.save_path)):
            os.makedirs(self.save_path)

        self.save_file = save_file
        self.num_joints = num_joints
        self.time_steps_raw = []
        self.positions_raw = []
        self.velocities_raw = []
        self.accelerations_raw = []
        self.end_poses = []

    def add_trajectory(self, trajectory, end_pose):

        time_steps_raw, positions_raw, velocities_raw, accelerations_raw = parse_trajectory(trajectory)
        self.time_steps_raw.append(time_steps_raw)
        self.positions_raw.append(positions_raw)
        self.velocities_raw.append(velocities_raw)
        self.accelerations_raw.append(accelerations_raw)
        self.end_poses.append(end_pose)

    def save(self):
        f = open(os.path.join(self.save_path, '{}.pkl'.format(self.save_file)), 'wb')
        pickle.dump([
            np.array(self.time_steps_raw), np.array(self.positions_raw),
            np.array(self.velocities_raw), np.array(self.accelerations_raw), np.array(self.end_poses)
        ], f)
        f.close()
