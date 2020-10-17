import torch
import numpy as np
from attack_models import Spectral_attack

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

# Load the model
attack_model_path = "attack_model_seed1.pt"
attack_model = torch.load(attack_model_path)
attack_model.eval()

y_pred_no_attack = attack_model.get_preds_no_noise(p_means, p_covariances, q_means, q_covariances, mask)
no_attack_avg_grade = torch.mean(y_pred_no_attack)
y_pred_attack = attack_model(p_means, p_covariances, q_means, q_covariances, mask)
attack_avg_grade = torch.mean(y_pred_attack)

    print("No attack average grade: ", no_attack_avg_grade)
    print("Attacked average grade: ", attack_avg_grade)

    # get the noise
    noise = attack_model.get_noise()
    print("Spectral noise: ", noise)