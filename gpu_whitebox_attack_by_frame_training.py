import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from attack_models_by_frame_spectral import Spectral_attack_by_frame
import math
import argparse
from pkl2pqvects import get_phones, get_vects
import pickle
import sys
import os

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')


def clip_params(model, barrier_val):
    old_params = {}

    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    for i, param in enumerate(old_params['noise_root']):
        if param > math.log(barrier_val):
            old_params['noise_root'][i] = math.log(barrier_val)

    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])


# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--barrier_val', default=1.0, type=float, help='limit on spectral attack noise')

args = commandLineParser.parse_args()
barrier_val = args.barrier_val


# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/train.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# set the device
device = get_default_device()

# Get all the p/q vects
pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
pkl = pickle.load(open(pkl_file, "rb"))
print("loaded pkl")
# get the phones
phones = get_phones()
max_len_frames = 4000
p_vects, q_vects, p_mask, q_mask, mask = get_vects(pkl, phones, max_len_frames)

num_spk = len(p_vects)

# Get output labels
y = pkl['score']
y = np.array(y)

# Convert to tensors
p_vects = torch.from_numpy(p_vects).float()
q_vects = torch.from_numpy(q_vects).float()
p_mask = torch.from_numpy(p_mask).float()
q_mask= torch.from_numpy(q_mask).float()
mask = torch.from_numpy(mask).float()
y = torch.from_numpy(y).float()

# Define constants
lr = 5*1e-1
epochs = 50
bs = 5
seed = 1
torch.manual_seed(seed)
trained_model_path = "FCC_lpron_seed1.pt"
spectral_dim = 24
mfcc_dim = 13
sch = 0.985

init_root = torch.FloatTensor([-2]*spectral_dim)
barrier_val = barrier_val
#barriers = torch.FloatTensor([barrier_val]*spectral_dim)

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(p_vects, q_vects, p_mask, q_mask, mask)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

attack_model = Spectral_attack_by_frame(spectral_dim, mfcc_dim, trained_model_path, init_root)
attack_model.to(device)
print("model initialised")

optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr, momentum = 0.9, nesterov=True)

# Scheduler for an adpative learning rate
# Every step size number of epochs, lr = lr * gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = sch)

 
for epoch in range(epochs):
    attack_model.train()
    combined_loss = 0
    for p, q, pm, qm, m in train_dl:

        p = p.to(device)
        q = q.to(device)
        pm = pm.to(device)
        qm = qm.to(device)
        m = m.to(device)

        # Forward pass
        y_pred = attack_model(p, q, pm, qm, m)

        # Compute loss
        loss = -1*torch.sum(y_pred)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Keep weights below barrier
        clip_params(attack_model, barrier_val)

        combined_loss+=loss.item()

    #print("loss: ", loss.item())
    avg = -1*combined_loss/num_spk
    print("Avg Grade: ")
    # Check average grade prediction
    #attack_model.eval()
    #y_pred_no_attack = attack_model.get_preds_no_noise(p_vects, q_vects, p_mask, q_mask, mask)
    #no_attack_avg_grade = torch.mean(y_pred_no_attack)
    #y_pred_attack = attack_model(p_vects, q_vects, p_mask, q_mask, mask)
    #attack_avg_grade = torch.mean(y_pred_attack)
    print("epoch", epoch)
    print("On validation")
    #print("No attack average grade: ", no_attack_avg_grade)
    print("Attacked average grade: ", attack_avg_grade)

    # get the noise
    noise = attack_model.get_noise()
    print("Spectral noise is currently: ", noise)

    scheduler.step()

# save the model
#output_file = "attack_model_init_root_constrained"+str(barrier_val)+"_seed"+str(seed)+".pt"
output_file = "attack_model_by_frame"+str(barrier_val)+"_seed"+str(seed)+".pt"
torch.save(attack_model, output_file)
