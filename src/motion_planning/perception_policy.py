import torch
import torch.nn as nn
import numpy as np


class Predictor(nn.Module):

    def __init__(self, input_size, output_size):

        super(Predictor, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, X):

        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-2)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


def DH(theta, d, r, alpha, device):

    """
     Calculates the Denavit-Hartenberg Matrix
     where
     d: offset along previous z to the common normal
     theta: angle about previous z, from old x to new x
     r: length of the common normal (aka a, but if using this notation, do not confuse with alpha). Assuming a revolute joint, this is the radius about previous z.
     alpha: angle about common normal, from old z axis to new z axis
    """

    T = torch.zeros([theta.shape[0], 4, 4]).to(device)
    T[:, :, :] = torch.eye(4).to(device)

    cTheta = torch.cos(theta)
    sTheta = torch.sin(theta)
    calpha = torch.cos(alpha)
    salpha = torch.sin(alpha)

    T[:, 0, 0] = cTheta
    T[:, 0, 1] = -sTheta
    T[:, 0, 2] = 0.0
    T[:, 0, 3] = r

    T[:, 1, 0] = sTheta * calpha
    T[:, 1, 1] = cTheta * calpha
    T[:, 1, 2] = -salpha
    T[:, 1, 3] = - d * salpha

    T[:, 2, 0] = sTheta * salpha
    T[:, 2, 1] = cTheta * salpha
    T[:, 2, 2] = calpha
    T[:, 2, 3] = d * calpha

    return T

def end_effector_pose(thetas, device):

    alphas = torch.Tensor([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]).to(device)
    ds = torch.Tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]).to(device)
    rs = torch.Tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]).to(device)

    T = torch.zeros([thetas.shape[0], 4, 4]).to(device)
    T[:, :, :] = torch.eye(4).to(device)

    # Base link coordinates
    T[:, 0, 3] = 0.0
    T[:, 1, 3] = 0.0
    T[:, 2, 3] = 0.0

    for idx in range(0, len(alphas) - 1):
        T_i = DH(thetas[:, idx], ds[idx], rs[idx], alphas[idx], device)
        T = T.bmm(T_i)

    T_i = DH(torch.zeros(thetas.shape[0]).to(device), ds[-1], rs[-1], alphas[-1], device)
    T = T.bmm(T_i)

    return torch.stack((T[:, 0, 3], T[:, 1, 3]), 1)

