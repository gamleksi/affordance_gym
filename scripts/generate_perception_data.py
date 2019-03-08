import os
import numpy as np
import rospy
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import argparse

from affordance_gym.utils import parse_vaed_arguments

from affordance_gym.simulation_interface import SimulationInterface

from env_setup.env_setup import ELEVATION_EPSILON, AZIMUTH_EPSILON, DISTANCE_EPSILON, POLICY_MODELS_PATH, LOOK_AT_EPSILON, NO_CUP_SHOWN_POSE
from env_setup.env_setup import VAED_MODELS_PATH, LOOK_AT, DISTANCE, AZIMUTH, ELEVATION, CUP_X_LIM, CUP_Y_LIM

from AffordanceVAED.tools import affordance_to_array, affordance_layers_to_array
from AffordanceVAED.ros_monitor import RosPerceptionVAE
import itertools

'''

Generates training data for a policy to map a affordance latent vector to a correct trajectory latent reprsentation.


Training limits for camera and cup position are defined in utils.


Three different dataset can be generated:

- clutter_env: only clutter objects on the table
- two_cups: two cups on the table

Training data is saved to the perception model's folder.


'''

# The following values should correspond to objects in the MuJoCo XML env.

NUM_RANDOM_OBJECTS = 15
NUM_CUPS = 9


def sample_visualize(image, affordance_arr, model_path, id):

    image = np.transpose(image, (1, 2, 0))

    sample_path = os.path.join(model_path, 'mujoco_debug')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    affordance = affordance_to_array(affordance_arr).transpose((1, 2, 0)) / 255.

    affordance_layers = affordance_layers_to_array(affordance_arr) / 255.
    affordance_layers = np.transpose(affordance_layers, (0, 2, 3, 1))
    # affordance_layers = [layer for layer in affordance_layers]

    samples = np.stack((image, affordance, affordance_layers[0], affordance_layers[1]))

    fig, axeslist = plt.subplots(ncols=4, nrows=1, figsize=(30, 30))

    for idx in range(samples.shape[0]):
        axeslist.ravel()[idx].imshow(samples[idx], cmap=plt.jet())
        axeslist.ravel()[idx].set_axis_off()

    plt.savefig(os.path.join(sample_path, 'sample_{}.png'.format(id)))
    plt.close(fig)


if __name__  == '__main__':

    parser = argparse.ArgumentParser(description='Generates perception training data for the policy part')

    parse_vaed_arguments(parser)

    parser.add_argument('--cup-id', default=1, type=int, help="cup-id lets you define two cups (cup-id, cup-id + 1) of which the samples are gathered (rendering cannot be parallellized in one roscore).")
    parser.add_argument('--clutter-env', dest='clutter_env', action='store_true', help="Only clutter objects on the table")
    parser.set_defaults(clutter_env=False)
    parser.add_argument('--two-cups', dest='two_cups', action='store_true', help="Two cups on table in each sample")
    parser.set_defaults(two_cups=False)
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Decreases the number of generated samples and visualizes the results to (POLICY_MODELS_PATH, 'debug')")

    args = parser.parse_args()

    rospy.init_node('generate_perception', anonymous=True, disable_signals=True)

    model = RosPerceptionVAE(os.path.join(VAED_MODELS_PATH, args.g_name), args.g_latent)

    if args.debug:
        x_steps = 2
        y_steps = 2
        camera_param_steps = 2
    else:
        x_steps = 10
        y_steps = 10
        camera_param_steps = 5

    save_path = os.path.join(VAED_MODELS_PATH, args.g_name)
    planner = SimulationInterface(arm_name='lumi_arm')
    planner.reset(2)

    planner.change_camere_params(LOOK_AT, DISTANCE, AZIMUTH, ELEVATION)

    lookat_x_values = LOOK_AT[0] + np.linspace(-LOOK_AT_EPSILON, LOOK_AT_EPSILON, camera_param_steps)
    lookat_y_values = LOOK_AT[1] + np.linspace(-LOOK_AT_EPSILON, LOOK_AT_EPSILON, camera_param_steps)
    distances = DISTANCE + np.linspace(-DISTANCE_EPSILON, DISTANCE_EPSILON, camera_param_steps)
    elevations = ELEVATION + np.linspace(-ELEVATION_EPSILON, ELEVATION_EPSILON, camera_param_steps)
    azimuths = AZIMUTH + np.linspace(-AZIMUTH_EPSILON, AZIMUTH_EPSILON, camera_param_steps)

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
        if (args.clutter_env):

            for i in range(10):

                num_objects = np.random.randint(2, 6)
                random_objects = np.random.choice(NUM_RANDOM_OBJECTS, num_objects) + 1

                # Set random object on the table
                for obj_id in random_objects:
                    x = np.random.uniform(CUP_X_LIM[0], CUP_X_LIM[1])
                    y = np.random.uniform(CUP_Y_LIM[0], CUP_Y_LIM[1])
                    obj_name = 'random{}'.format(obj_id)
                    planner.change_object_position(x, y, 0.0, obj_name, duration=0)

                image_arr = planner.capture_image("/lumi_mujoco/rgb")
                image = Image.fromarray(image_arr)

                # Get latent1
                latent = model.get_latent(image)
                latent = latent.detach().cpu().numpy()

                # Store samples
                cup_positions.append((NO_CUP_SHOWN_POSE[0], NO_CUP_SHOWN_POSE[1]))
                latents.append(latent)
                container_distances.append(distance)
                container_azimuths.append(azimuth)
                container_elevations.append(elevation)
                cup_ids.append(0)
                lookat_points.append(lookat)

                if args.debug:
                    affordance, sample = model.reconstruct(image)
                    sample_visualize(sample, affordance, save_path, idx)
                idx += 1

                # Remove selected objects from the table
                for obj_id in random_objects:
                    obj_name = 'random{}'.format(obj_id)
                    planner.change_object_position(10, 12 + obj_id, 0.0, obj_name, duration=0)

        elif (args.two_cups):

            if args.debug:
                cup_id_range = range(3, 6)
            else:
                cup_id_range = range(args.cup_id * 2 + 1, (args.cup_id + 1) * 2 + 1)

            for cup_id in cup_id_range:

                for x in np.linspace(CUP_X_LIM[0], CUP_X_LIM[1], x_steps):

                   for y in np.linspace(CUP_Y_LIM[0], CUP_Y_LIM[1], y_steps):

                        # First cup
                        planner.change_object_position(x, y, 0.0, 'cup{}'.format(cup_id), duration=0)

                        # choose randomly another cup. It is assumed that if the cup_id is larger than the firs one's id
                        # then the randomly drawn cup is larger

                        cup_id2 = 2 # np.random.randint(1, NUM_CUPS + 1)

                        if cup_id2 == cup_id:
                            if cup_id2 < 2:
                                cup_id2 += 1
                            else:
                                cup_id2 -= 1

                        cup1_x_area = np.array([x - 0.15, x + 0.15])
                        cup1_y_area = np.array([y - 0.15, y + 0.15])

                        # Probabilities whether the randomly drawn cup is located to left or right side of the first cup
                        left_x = max((cup1_x_area[0] - CUP_X_LIM[0]) / (CUP_X_LIM[1] - CUP_X_LIM[0]), 0.0)
                        right_x = max(1 - (cup1_x_area[1] - CUP_X_LIM[0]) / (CUP_X_LIM[1] - CUP_X_LIM[0]), 0.0)
                        # normalization between 0
                        left_x_prob = left_x / (left_x + right_x)

                        x_side_sample = np.random.uniform()
                        if x_side_sample < left_x_prob:
                            cup2_x = np.random.uniform(CUP_X_LIM[0], cup1_x_area[0])
                        else:
                            cup2_x = np.random.uniform(cup1_x_area[1], CUP_X_LIM[1])

                        left_y = max((cup1_y_area[0] - CUP_Y_LIM[0]) / (CUP_Y_LIM[1] - CUP_Y_LIM[0]), 0.0)
                        right_y = max(1 - (cup1_y_area[1] - CUP_Y_LIM[0]) / (CUP_Y_LIM[1] - CUP_Y_LIM[0]), 0.0)
                        print("left_y", left_y)
                        print("right_y", right_y)
                        # normalization between 0
                        left_y_prob = left_y / (left_y + right_y)

                        y_side_sample = np.random.uniform()
                        if y_side_sample < left_x_prob:
                            cup2_y = np.random.uniform(CUP_Y_LIM[0], cup1_y_area[0])
                        else:
                            cup2_y = np.random.uniform(cup1_y_area[1], CUP_Y_LIM[1])

                        # Add the random cup on the table
                        planner.change_object_position(cup2_x, cup2_y, 0.0, 'cup{}'.format(cup_id2), duration=0)

                        image_arr = planner.capture_image("/lumi_mujoco/rgb")
                        image = Image.fromarray(image_arr)

                        # Get latent1
                        latent = model.get_latent(image)
                        latent = latent.detach().cpu().numpy()

                        # Store samples

                        # The goal pose is the larger cup
                        if (cup_id2 < cup_id):
                            cup_positions.append((x, y))
                        else:
                            cup_positions.append((cup2_x, cup2_y))

                        latents.append(latent)
                        container_distances.append(distance)
                        container_azimuths.append(azimuth)
                        container_elevations.append(elevation)
                        cup_ids.append(cup_id)
                        lookat_points.append(lookat)

                        if args.debug:
                            affordance, sample = model.reconstruct(image)
                            sample_visualize(sample, affordance, save_path, idx)
                        idx += 1
                        planner.change_object_position(10, 12 + cup_id2, 0.0, 'cup{}'.format(cup_id2), duration=0)

                   planner.change_object_position(10, 12 + cup_id, 0.0, 'cup{}'.format(cup_id), duration=0)

        else:

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
                       image_arr = planner.capture_image("/lumi_mujoco/rgb")
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
                           sample_visualize(sample, affordance, save_path, idx)

                       idx += 1
                       print("sample: {} / {}".format(idx, (camera_param_steps ** 5) * x_steps * y_steps * 2))


    # Save training samples
    save_path = os.path.join(save_path, 'mujoco_latents')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.clutter_env:
        save_path = os.path.join(save_path, 'random_{}.pkl'.format(args.cup_id))
    elif args.two_cups:
        save_path = os.path.join(save_path, 'two_cups_latents_{}.pkl'.format(args.cup_id))
    else:
        save_path = os.path.join(save_path, 'latents_{}.pkl'.format(args.cup_id))

    f = open(save_path, 'wb')
    pickle.dump([np.array(latents), np.array(lookat_points), np.array(container_distances),
                 np.array(container_azimuths), np.array(container_elevations),
                 np.array(cup_ids), np.array(cup_positions)], f)
    f.close()
