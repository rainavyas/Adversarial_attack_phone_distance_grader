import torch
from utility import *
import numpy as np
from models import FCC

model_path = "FCC_lpron_seed1.pt"
input_file = 'BLXXXgrd02_means_covs.npz'
npzfile = np.load(input_file)
p_means = npzfile['arr_0']
p_covariances = npzfile['arr_1']
q_means = npzfile['arr_2']
q_covariances = npzfile['arr_3']
mask = npzfile['arr_4']
y = npzfile['arr_5']

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

# Load the model
model = torch.load(model_path)
model.eval()

# get predictions
y_pred = model(p_means, p_covariances, q_means, q_covariances, mask)
y_pred[y_pred>6]=6.0
y_pred[y_pred<0]=0.0
y_pred_list = y_pred.tolist()

y_list = y.tolist()

print(y_pred_list)
print(y_list)

mse = calculate_mse(y_pred_list, y_list)
pcc = calculate_pcc(y_pred, y)
less1 = calculate_less1(y_pred, y)
less05 = calculate_less05(y_pred, y)

print("mse: "+ str(mse)+"\n pcc: "+str(pcc)+"\n less than 1 away: "+ str(less1)+"\n less than 0.5 away: "+str(less05))
