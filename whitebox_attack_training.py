import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np


# Load the means and covariances
input_file = 'BLXXXgrd02_means_covs.npz'
npzfile = np.load(input_file)
p_means = npzfile['arr_0']
p_covariances = npzfile['arr_1']
q_means = npzfile['arr_2']
q_covariances = npzfile['arr_3']
mask = npzfile['arr_4']
y = npzfile['arr_5']

print("got means and covs")

# convert to tensors
p_means = torch.from_numpy(p_means).float()
p_covariances = torch.from_numpy(p_covariances).float()
q_means = torch.from_numpy(q_means).float()
q_covariances = torch.from_numpy(q_covariances).float()
mask = torch.from_numpy(mask).float()
y = torch.from_numpy(y).float()

# add small noise to all covariance matrices to ensure they are non-singular
p_covariances = p_covariances + (1e-4*torch.eye(13))
q_covariances = q_covariances + (1e-4*torch.eye(13))

# Define constants
lr = 3*1e-2
epochs = 400
bs = 450
seed = 1
torch.manual_seed(seed)
