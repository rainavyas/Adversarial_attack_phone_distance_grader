import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models_from_mfcc_vects import LPRON
import numpy as np
from utility import calculate_mse


# Load the means and covariances
input_file = 'BLXXXgrd02_pqvects.npz'
npzfile = np.load(input_file)
p_vects = npzfile['arr_0']
q_vects = npzfile['arr_1']
p_lengths = npzfile['arr_2']
q_lengths = npzfile['arr_3']
mask = npzfile['arr_4']
y = npzfile['arr_5']

print("got means and covs")

# convert to tensors
p_vects = torch.from_numpy(p_vects).float()
q_vects = torch.from_numpy(q_vects).float()
p_lengths = torch.from_numpy(p_lengths).float()
q_lengths = torch.from_numpy(q_lengths).float()
mask = torch.from_numpy(mask).float()
y = torch.from_numpy(y).float()

# Define constants
lr = 3*1e-2
epochs = 400
bs = 450
seed = 1
torch.manual_seed(seed)

# Split into training and dev sets
num_dev = 100

p_vects_dev = p_vects[:num_dev]
q_vects_dev = q_vects[:num_dev]
p_lengths_dev = p_lengths[:num_dev]
q_lengths_dev = q_lengths[:num_dev]
mask_dev = mask[:num_dev]
y_dev = y[:num_dev]

p_vects_train = p_vects[num_dev:]
q_vects_train = q_vects[num_dev:]
p_lengths_train = p_lengths[num_dev:]
q_lengths_train = q_lengths[num_dev:]
mask_train = mask[num_dev:]
y_train = y[num_dev:]

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(p_vects_train, q_vects_train, p_lengths_train, q_lengths_train, mask_train, y_train)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

num_features = 1128
model = LPRON(num_features)
model = model.float()
print("model initialised")

criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define a scheduler for an adaptive learning rate
lambda1 = lambda epoch: 0.999**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)

for epoch in range(epochs):
    model.train()
    for pv, qv, pl, ql, m, yb in train_dl:

        # Forward pass
        y_pred = model(pv, qv, pl, ql, m)

        # Compute loss
        loss = criterion(y_pred, yb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss.item())
    model.eval()
    # Evaluate on dev set
    y_pr = model(p_vects_dev, q_vects_dev, p_lengths_dev, q_lengths_dev, mask_dev)
    y_pr[y_pr>6]=6
    y_pr[y_pr<0]=0
    dev_loss = calculate_mse(y_pr.tolist(), y_dev.tolist())
    print(epoch, dev_loss)

    scheduler.step()

# save the model
output_file = "LPRON_seed"+str(seed)+"_epochs"+str(epochs)+".pt"
torch.save(model, output_file)
