import os
import torch
import numpy as np
from AffordanceVAED.tools import affordance_to_array, affordance_layers_to_array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def print_pose(pose, tag='Pose'):
    print("{}: x: {}, y: {}, z: {}".format(tag, pose[0], pose[1], pose[2]))


def load_parameters(model, load_path, model_name):
    model_path = os.path.join(load_path, "{}.pth.tar".format(model_name))
    model.load_state_dict(torch.load(model_path))
    model.eval()


def parse_traj_arguments(parser):

    parser.add_argument('--traj-name', default='model_v2', type=str, help='folder name of the trajectory vae')
    parser.add_argument('--traj-latent', default=5, type=int, help='Latent size of trajectory VAE')
    parser.add_argument('--num-joints', default=7, type=int, help='The number of robot joints')
    parser.add_argument('--num-actions', default=24, type=int, help='The number of actions in a trajectory')
    parser.add_argument('--model-index', default=11, type=int, help='The nth saved model in the model folder')


def parse_vaed_arguments(parser):

    parser.add_argument('--vaed-name', default='rgb_model_v1', type=str, help='folder name of variational affordance encoder decoder')
    parser.add_argument('--vaed-latent', default=10, type=int, help='Latent size of VAED')


def parse_moveit_arguments(parser):
    parser.add_argument('--duration', default=0.5, type=float, help='Duration of a generated trajectory in MuJoCo or Real HW')


def parse_kinect_arguments(parser):

    parser.add_argument('--real-hw', dest='real_hw', action='store_true')
    parser.set_defaults(real_hw=False, help='use a real harware setup')
    parser.add_argument('--log-name', default='kinect_example', type=str, help='saves real experiment results to a given folder')
    parser.add_argument('--top-crop', default=64, type=int)
    parser.add_argument('--width-crop', default=0, type=int)
    parser.add_argument('--x-pose', default=None, type=float, help='Defines a constant cup x position')
    parser.add_argument('--y-pose', default=None, type=float, help='Defines a constant cup y position')
    parser.add_argument('--cup-type', default=None, type=str, help='Defines a constant cup name')
    parser.add_argument('--random-objs', default=None, type=int, help='Defines a constant number of clutter objects on a table')


def parse_policy_arguments(parser):
    parser.add_argument('--policy-name', default='model_v1', type=str, help='folder name of the policy')
    parser.add_argument('--num-params', default=128, type=int, help='Num params in each policy layer (default 128)')
    parser.add_argument('--fixed-camera', dest='fixed_camera', action='store_true', help='Camera params are not given to policy')
    parser.set_defaults(fixed_camera=False)


def parse_policy_train_arguments(parser):

    parser.add_argument('--num-epoch', default=1000, type=int)
    parser.add_argument('--batch-size', default=124, type=int)
    parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
    parser.add_argument('--num-processes', default=16, type=int)


def sample_visualize(image, affordance_arr, sample_path, id):

    image = np.transpose(image, (1, 2, 0))

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    affordance = affordance_to_array(affordance_arr).transpose((1, 2, 0)) / 255.

    affordance_layers = affordance_layers_to_array(affordance_arr) / 255.
    affordance_layers = np.transpose(affordance_layers, (0, 2, 3, 1))

    samples = np.stack((image, affordance, affordance_layers[0], affordance_layers[1]))

    fig, axeslist = plt.subplots(ncols=4, nrows=1)

    for idx in range(samples.shape[0]):
        axeslist.ravel()[idx].imshow(samples[idx], cmap=plt.jet())
        axeslist.ravel()[idx].set_axis_off()

    plt.tight_layout()

    plt.savefig(os.path.join(sample_path, 'sample_{}.png'.format(id)))
    plt.close(fig)


def use_cuda():

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    return torch.device('cuda' if use_cuda else 'cpu')

def save_arguments(args, save_path):

    args = vars(args)

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    file = open(os.path.join(save_path, "arguments.txt"), 'w')
    lines = [item[0] + " " + str(item[1]) + "\n" for item in args.items()]
    file.writelines(lines)


def plot_loss(train, val, title, save_to):

    steps = range(1, train.__len__() + 1)

    fig = plt.figure()

    plt.plot(steps, train, 'r', label='Train')
    plt.plot(steps, val, 'b', label='Validation')

    plt.title(title)
    plt.legend()
    plt.savefig(save_to)
    plt.close()


def plot_scatter(constructed, targets, save_to):
    fig = plt.figure()
    plt.scatter(targets[:, 0], targets[:, 1], label='targets', c='r')
    plt.scatter(constructed[:, 0], constructed[:, 1], label='constructed', c='b')
    plt.legend()
    plt.savefig(save_to)
    plt.close()

def plot_latent_distributions(latents, save_to):

    fig, axes = plt.subplots(latents.shape[1], 1, sharex=True, figsize=[30, 30])

    for i in range(latents.shape[1]):
        ax = axes[i]
        batch = latents[:, i]
        ax.hist(batch, bins=100)
        ax.set_title('Latent {}'.format(i + 1))
        ax.set_xlabel('x')
        ax.set_ylabel('frequency')
    plt.savefig(save_to)
    plt.close()
