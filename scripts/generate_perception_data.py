import os
import numpy as np
import rospy
import matplotlib.pyplot as plt
from PIL import Image
import pickle

from motion_planning.utils import parse_arguments, GIBSON_ROOT, LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM
from motion_planning.utils import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON, POLICY_ROOT
from motion_planning.simulation_interface import SimulationInterface
from gibson.tools import affordance_to_array
from gibson.ros_monitor import RosPerceptionVAE
import itertools


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


if __name__  == '__main__':

    rospy.init_node('generate_perception', anonymous=True)

    args = parse_arguments(gibson=True)
    model = RosPerceptionVAE(os.path.join(GIBSON_ROOT, args.g_name), args.g_latent)

    if args.debug:
        model_path = os.path.join(POLICY_ROOT, 'debug', 'perception', args.g_name)
        x_steps = 2
        y_steps = 2
    else:
        model_path = os.path.join(GIBSON_ROOT, args.g_name)
        x_steps = 10
        y_steps = 10

    planner = SimulationInterface(arm_name='lumi_arm')
    planner.reset(2)

    planner.change_camere_params(LOOK_AT, DISTANCE, AZIMUTH, ELEVATION)

    if (args.debug):
        steps = 2
        cup_id_steps = 1
    else:
        steps = 5
        cup_id_steps = 10

    lookat_x_values = LOOK_AT[0] + np.linspace(-0.05, 0.05, steps)
    lookat_y_values = LOOK_AT[1] + np.linspace(-0.05, 0.05, steps)

    distances = DISTANCE + np.linspace(-DISTANCE_EPSILON, DISTANCE_EPSILON, steps)
    elevations = ELEVATION + np.linspace(-ELEVATION_EPSILON, ELEVATION_EPSILON, steps)
    azimuths = AZIMUTH + np.linspace(-AZIMUTH_EPSILON, AZIMUTH_EPSILON, steps)

    # camera_params_combinations = itertools.product([LOOK_AT[0]], [LOOK_AT[1]], [LOOK_AT[2]], distances, azimuths, elevations)
    camera_params_combinations = itertools.product(lookat_x_values, lookat_y_values, [LOOK_AT[2]], distances, azimuths, elevations)

    idx = 0
    inputs = []
    cup_positions = []
    latents = []
    container_elevations = []
    container_azimuths = []
    container_distances = []
    cup_ids = []
    lookat_points = []

    for camera_params in camera_params_combinations:

        lookat = [camera_params[0], camera_params[1], camera_params[2]]
        distance = camera_params[3]
        azimuth = camera_params[4]
        elevation = camera_params[5]

        print(lookat)
        print(distance)
        print(azimuth)
        print(elevation)

        planner.change_camere_params(lookat, distance, azimuth, elevation)

        if args.debug:
            cup_id_range = range(1, 2)
        else:
            cup_id_range = range(args.cup_id * 2 + 1, (args.cup_id + 1) * 2 + 1)

        for cup_id in cup_id_range:

           cup_name = 'cup{}'.format(cup_id)

           for x in np.linspace(CUP_X_LIM[0], CUP_X_LIM[1], x_steps):

               for y in np.linspace(CUP_Y_LIM[0], CUP_Y_LIM[1], y_steps):

                   # Change pose of the cup and get an image sample
                   planner.reset_table(x, y, 0, cup_name, duration=0.01)
                   image_arr = planner.capture_image()
                   image = Image.fromarray(image_arr)

                   # Get latent1
                   latent = model.get_latent(image)
                   latent = latent.detach().cpu().numpy()

                   # Store samples
                   cup_positions.append((x, y))
                   latents.append(latent)
                   container_distances.append(distance)
                   container_azimuths.append(azimuth)
                   container_elevations.append(elevation)
                   cup_ids.append(cup_id)
                   lookat_points.append(lookat)

                   # Visualize affordance results
                   if args.debug:
                       affordance, sample = model.reconstruct(image)
                       sample_visualize(sample, affordance, model_path, idx)

                   idx += 1
                   print("sample: {} / {}".format(idx, (steps ** 5) * x_steps * y_steps * cup_id_steps))

    # Save training samples
    save_path = os.path.join(model_path, 'mujoco_latents')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, 'latents_{}.pkl'.format(args.cup_id))
    f = open(save_path, 'wb')
    pickle.dump([np.array(latents), np.array(lookat_points), np.array(container_distances),
                 np.array(container_azimuths), np.array(container_elevations),
                 np.array(cup_ids), np.array(cup_positions)], f)
    f.close()

