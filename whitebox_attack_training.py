import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
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

# Define constants
lr = 3*1e-3
epochs = 100
bs = 50
seed = 1
torch.manual_seed(seed)
trained_model_path = "FCC_lpron_seed1.pt"
spectral_dim = 24
mfcc_dim = 13


# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(p_means, p_covariances, q_means, q_covariances, mask)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

attack_model = Spectral_attack(spectral_dim, mfcc_dim, trained_model_path)
print("model initialised")

optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr)

for epoch in range(epochs):
    attack_model.train()

    y_pred = attack_model(p_means, p_covariances, q_means, q_covariances, mask)
    loss = -1*torch.sum(y_pred)
    optimizer.zero_grad()
    print("backwarding")
    loss.backward()
    optimizer.step()
    

#    for pm, pc, qm, qc, m in train_dl:

        # Forward pass
#        y_pred = attack_model(pm, pc, qm, qc, m)

        # Compute loss
#        loss = -1*torch.sum(y_pred)

        # Zero gradients, backward pass, update weights
#        optimizer.zero_grad()
#        loss.backward(retain_graph=TRUE)
#        optimizer.step()

    # Check average grade prediction
    attack_model.eval()
    y_pred_no_attack = attack_model.get_preds_no_noise(p_means, p_covariances, q_means, q_covariances, mask)
    no_attack_avg_grade = torch.mean(y_pred_no_attack)
    y_pred_attack = attack_model(p_means, p_covariances, q_means, q_covariances, mask)
    attack_avg_grade = torch.mean(y_pred_attack)
    print("epoch", epoch)
    print("No attack average grade: ", no_attack_avg_grade)
    print("Attacked average grade: ", attack_avg_grade)

    # get the noise
    noise = attack_model.get_noise()
    print("Spectral noise is currently: ", noise)


# save the model
output_file = "attack_model_seed"+str(seed)+".pt"
torch.save(attack_model, output_file)
