import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import FCC
import pickle


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
p_means = torch.from_numpy(p_means)
p_covariances = torch.from_numpy(p_covariances)
q_means = torch.from_numpy(q_means)
q_covariances = torch.from_numpy(q_covariances)
mask = torch.from_numpy(mask)
y = torch.from_numpy(y)


# Define constants
lr = 3*1e-2
epochs = 400
bs = 20
seed = 1
torch.manual_seed(seed)

# Split into training and dev sets
num_dev = 100

p_means_dev = p_means[:num_dev]
q_means_dev = q_means[:num_dev]
p_covariances_dev = p_covariances[:num_dev]
q_covariances_dev = p_covariances[:num_dev]
mask_dev = mask[:num_dev]
y_dev = y[:num_dev]

p_means_train = p_means[num_dev:]
q_means_train = q_means[num_dev:]
p_covariances_train = p_covariances[num_dev:]
q_covariances_train = p_covariances[num_dev:]
mask_train = mask[num_dev:]
y_train = y[num_dev:]

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(p_means_train, p_covariances_train, q_means_train, q_covariances_train, mask_train, y_train)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

model = FCC(num_features)
print("model initialised")

criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define a scheduler for an adaptive learning rate
lambda1 = lambda epoch: 0.999**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)

for epoch in range(epochs):
    model.train()
    for pm, pc, qm, qc, m, yb in train_dl:

        # Forward pass
        y_pred = model(pm, pc, qm, qc, m)

        # Compute loss
        loss = criterion(y_pred, yb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss.item())
    model.eval()
    # Evaluate on dev set
    y_pr = model(X_dev)
    y_pr[y_pr>6]=6
    y_pr[y_pr<0]=0
    #dev_loss = calculate_mse(y_pr.tolist(), y_dev.tolist())
    #print(epoch, dev_loss)

    scheduler.step()
