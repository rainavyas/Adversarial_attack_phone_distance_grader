import torch
import numpy
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import FCC
import pickle


def get_phones(alphabet='arpabet'):
    if alphabet == 'arpabet':
        vowels = ['aa', 'ae', 'eh', 'ah', 'ea', 'ao', 'ia', 'ey', 'aw', 'ay', 'ax', 'er', 'ih', 'iy', 'uh', 'oh', 'oy', 'ow', 'ua', 'uw']
        consonants = ['el', 'ch', 'en', 'ng', 'sh', 'th', 'zh', 'w', 'dh', 'hh', 'jh', 'em', 'b', 'd', 'g', 'f', 'h', 'k', 'm', 'l', 'n', 'p', 's', 'r', 't', 'v', 'y', 'z'] + ['sil']
        phones = vowels + consonants
        return phones
    if alphabet == 'graphemic':
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'] + ['sil']
        phones = vowels + consonants
        return phones
    raise ValueError('Alphabet name not recognised: ' + alphabet)


def get_pdf(obj, phones):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector

    # Define the tensors required by spectral attack model
    p_means = np.zeros((len(obj['plp']), (len(phones)-1)*(len(phones)-2)*0.5 , n))
    p_covariances = np.zeros((len(obj['plp']), (len(phones)-1)*(len(phones)-2)*0.5, n, n))
    q_means = np.zeros((len(obj['plp']), (len(phones)-1)*(len(phones)-2)*0.5 , n))
    q_covariances = np.zeros((len(obj['plp']), (len(phones)-1)*(len(phones)-2)*0.5, n, n))
    num_phones_mask = np.zeros(len(obj['plp']), (len(phones)-1)*(len(phones)-2)*0.5)


    for spk in range(len(obj['plp'])):
        SX = np.zeros((len(phones) - 1, n, 1))
        N = np.zeros(len(phones) - 1)
        SX2 = np.zeros((len(phones) - 1, n, n))
        Sig = np.zeros((len(phones) - 1, n, n))

        for utt in range(len(obj['plp'][spk])):
            for w in range(len(obj['plp'][spk][utt])):
                for ph in range(len(obj['plp'][spk][utt][w])):
                    for frame in range(len(obj['plp'][spk][utt][w][ph])):
                        N[obj['phone'][spk][utt][w][ph]] += 1
                        X = np.reshape(np.array(obj['plp'][spk][utt][w][ph][frame]), [n, 1])
                        SX[obj['phone'][spk][utt][w][ph]] += X
                        SX2[obj['phone'][spk][utt][w][ph]] += np.matmul(X, np.transpose(X))

        for ph in range(len(phones)-1):
            if N[ph] !=0:
                SX[ph] /= N[ph]
                SX2[ph] /= N[ph]
            m2 = np.matmul(SX[ph], np.transpose(SX[ph]))
            Sig[ph] = SX2[ph] - m2

        k = 0
        for i in range(len(phones) - 1):
            for j in range(i + 1, len(phones) - 1):
                if N[i] == 0 or N[j] == 0:
                    num_phones_mask[spk][k] = 0
                    # define Gaussian distribution that has 0 kl div
                    # later the mask will be used to make these features "-1"
                    p_covariances[spk][k] = np.eye(n)
                    q_covariances[spk][k] = np.eye(n)
                else:
                    num_phones_mask[spk][k] = 1
                    p_covariances[spk][k] = Sig[i]
                    q_covariances[spk][k] = Sig[j]

                p_means[spk][k] = SX[i]
                q_means[spk][k] = SX[j]

                k += 1

    return p_means, p_covariances, q_means, q_covariances, num_phones_mask


pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
pkl = pickle.load(open(pkl_file, "rb"))

print("loaded pkl")

# get the phones
phones = get_phones()

# get the means and covariances split into p and q groups (for doing kl)
p_means, p_covariances, q_means, q_covariances, mask = get_pdf(pkl, phones)

print("got means and covs")

# convert to tensors
p_means = torch.from_numpy(p_means)
p_covariances = torch.from_numpy(p_covariances)
q_means = torch.from_numpy(q_means)
q_covariances = torch.from_numpy(q_covariances)
mask = torch.from_numpy(mask)


# Define constants
lr = 3*1e-2
epochs = 400
bs = 20
seed = 1
torch.manual_seed(seed)

# Construct the output scores tensor
y = (pkl['score'])
y = torch.FloatTensor(y)

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
