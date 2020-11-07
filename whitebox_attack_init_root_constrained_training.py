import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from attack_models_init_root import Spectral_attack_init

def barrier(barriers, noise, r):
    '''
    A barrier function that penalises parameters nearing the barrier
    '''
    diff = barriers - noise
    penalty = torch.log(diff)
    total_penalty = torch.sum(penalty)
    barrier_loss = -1*r*total_penalty
    return barrier_loss


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
p_covariances = p_covariances + (1e-3*torch.eye(13))
q_covariances = q_covariances + (1e-3*torch.eye(13))

# Define constants
lr = 2*1e-1
epochs = 200
bs = 100
seed = 1
torch.manual_seed(seed)
trained_model_path = "FCC_lpron_seed1.pt"
spectral_dim = 24
mfcc_dim = 13
r = 0.01 # controls impact of barrier function

init_root = torch.FloatTensor([-3]*spectral_dim)
barriers = torch.FloatTensor([0.5]*spectral_dim)

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(p_means, p_covariances, q_means, q_covariances, mask)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

attack_model = Spectral_attack_init(spectral_dim, mfcc_dim, trained_model_path, init_root)
print("model initialised")

optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr)

for epoch in range(epochs):
    attack_model.train()
    for pm, pc, qm, qc, m in train_dl:

        # Forward pass
        y_pred = attack_model(pm, pc, qm, qc, m)

        # Compute loss
        loss = -1*torch.sum(y_pred) + barrier(barriers, attack_model.get_noise(), r)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    print("loss and regularisation value: ", loss.item())
    # Check average grade prediction
    attack_model.eval()
    y_pred_no_attack = attack_model.get_preds_no_noise(p_means, p_covariances, q_means, q_covariances, mask)
    no_attack_avg_grade = torch.mean(y_pred_no_attack)
    y_pred_attack = attack_model(p_means, p_covariances, q_means, q_covariances, mask)
    attack_avg_grade = torch.mean(y_pred_attack)
    print("epoch", epoch)
    print("On validation")
    print("No attack average grade: ", no_attack_avg_grade)
    print("Attacked average grade: ", attack_avg_grade)

    # get the noise
    noise = attack_model.get_noise()
    print("Spectral noise is currently: ", noise)


# save the model
output_file = "attack_model_init_root_constrained_seed"+str(seed)+".pt"
torch.save(attack_model, output_file)
