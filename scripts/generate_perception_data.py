import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

from motion_planning.utils import parse_arguments, GIBSON_ROOT, LUMI_X_LIM, LUMI_Y_LIM, LUMI_Z_LIM
from motion_planning.simulation_interface import SimulationInterface
from gibson.tools import affordance_to_array
from gibson.ros_monitor import RosPerceptionVAE


def sample_visualize(image, affordance, model_path, id):

    image = np.transpose(image, (1, 2, 0))

    sample_path = os.path.join(model_path,'mujoco_samples')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    affordance = affordance_to_array(affordance).transpose((1, 2, 0)) / 255.

    samples = np.stack((image, affordance))

    fig, axeslist = plt.subplots(ncols=2, nrows=1, figsize=(30, 30))

    for idx in range(samples.shape[0]):
        axeslist.ravel()[idx].imshow(samples[idx], cmap=plt.jet())
        axeslist.ravel()[idx].set_axis_off()

    plt.savefig(os.path.join(sample_path, 'sample_{}.png'.format(id)))
    plt.close(fig)


class Experiment:

    def __init__(self, model, planner):
        self.model = model
        self.planner = planner
        self.iter = 0

    def do(self, x, y):
        planner.reset_table(x, y, 0)
        image_arr = self.planner.capture_image()
        image = Image.fromarray(image_arr)
        model.get_latent(image)
        affordance, sample = self.model.reconstruct(image)
        sample_visualize(sample, affordance, 'gibson_test', self.iter)
        self.iter += 1


def end_effector_pose(thetas):

    thetas.append(0)
    alphas = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
    ds = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
    rs = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]

    T = np.eye(4)
    T[0, 3] = -0.4
    T[1, 3] = 0.15
    T[2, 3] = 0

    for idx in range(0, len(alphas)):
        T_i = DH(thetas[idx], ds[idx], rs[idx], alphas[idx])
        T = np.dot(T, T_i)

    return (T[0, 3], T[1, 3], T[2, 3])


if __name__  == '__main__':

    args = parse_arguments(gibson=True)
    model = RosPerceptionVAE(args.g_name, args.g_latent, root_path=GIBSON_ROOT)
    planner = SimulationInterface(arm_name='lumi_arm')
    planner.reset(2)

    look_at = [.45, 4.1, -2.8]
    distance = 6.
    azimuth = 90.
    elevation = -35.

    planner.change_camere_params(look_at, distance, azimuth, elevation)
    worker = Experiment(model, planner)

    cup_positions = []
    images = []

    for x in np.linspace(0.0, 0.15, 20):

        for y in np.linspace(-0.20, 0.40, 70):

            planner.reset_table(x, y, 0)
            image_arr = planner.capture_image()
            image = Image.fromarray(image_arr)
            images.append(image)
            cup_positions.append((x, y))

    latents = []
    for image in images:
        latent = model.get_latent(image)
        latent = latent.detach().cpu().numpy()
        latents.append(latent)

    model_path = os.path.join(GIBSON_ROOT, 'log', args.g_name)

    save_path = os.path.join(model_path, 'mujoco_latents')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, 'latents.pkl')
    f = open(save_path, 'wb')
    pickle.dump([np.array(latents), np.array(cup_positions)], f)
    f.close()

    for idx in range(len(images)):
        affordance, sample = model.reconstruct(images[idx])
        sample_visualize(sample, affordance, model_path, idx)

#    worker.do(x, y)
#    planner.move_arm_to_position(x_p=x, y_p=y, z_p=0.4)
#    real_coord = planner.current_pose()
#    real_coord = [real_coord.pose.position.x, real_coord.pose.position.y, real_coord.pose.position.z]
#    coord = end_effector_pose(planner.current_joint_values())
