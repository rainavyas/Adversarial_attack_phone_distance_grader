import numpy as np
import pickle
import torch
import torch_dct as dct
from models import FCC
from attack_models import Spectral_attack
from utility import *
import argparse

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


def spectral_attack(X, attack):
    X = torch.from_numpy(X).float()
    X_sq = X.squeeze()
    attack = torch.from_numpy(attack).float()

    # Add the attack in the spectral space
    # Pad to spectral dimension
    padding = torch.zeros(attack.size(0) - X_sq.size(0))
    padded_X = torch.cat((X_sq, padding))

    # Apply inverse dct
    log_spectral_X = dct.idct(padded_X)

    # Apply inverse log
    spectral_X = torch.exp(log_spectral_X)

    # Add the adversarial attack
    attacked_spectral_X = spectral_X + attack

    # Get back to mfcc domain
    attacked_log_spectral_X = torch.log(attacked_spectral_X)
    attacked_padded_X = dct.dct(attacked_log_spectral_X)
    X_attacked = torch.narrow(attacked_padded_X, 0, 0, X_sq.size(0))
    X_attacked = X_attacked.detach().numpy()

    return X_attacked


def get_pdf_attack(obj, phones, attack):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector
    num_spk = len(obj['plp'])
    num_spk = 50

    # Define the tensors required by spectral attack model
    p_means = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    p_covariances = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5), n, n))
    q_means = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    q_covariances = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5), n, n))
    num_phones_mask = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5)))


    for spk in range(num_spk):
        print("on speaker", spk)
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
                        X_attack = np.reshape(spectral_attack(X, attack), [n, 1])
                        SX[obj['phone'][spk][utt][w][ph]] += X_attack
                        SX2[obj['phone'][spk][utt][w][ph]] += np.matmul(X_attack, np.transpose(X_attack))

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

                p_means[spk][k] = SX[i].squeeze()
                q_means[spk][k] = SX[j].squeeze()

                k += 1

    return p_means, p_covariances, q_means, q_covariances, num_phones_mask

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--barrier_val', default=1.0, type=float, help='specify attack model to load with barrier val')

args = commandLineParser.parse_args()
barrier_val = args.barrier_val


#pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
pkl = pickle.load(open(pkl_file, "rb"))

print("loaded pkl")

# get the phones
phones = get_phones()

# get the attack spectral vector
#attack_model_path = "attack_model_seed1.pt"
#attack_model_path = "attack_model_init_root_constrained"+str(barrier_val)+"_seed1.pt"
attack_model_path = "attack_model_Taylor1_init_root_constrained"+str(barrier_val)+"_seed1.pt"
attack_model = torch.load(attack_model_path)
attack_model.eval()
attack = attack_model.get_noise()
attack = attack.detach().numpy()

#attack = np.array([1.0, 0.0966, 1.0, 0.7846, 0.3997, 0.0040, 0.1374, 0.1373, 0.0979, 1.0000, 1.0000, 0.0597, 0.0812, 0.0080, 0.0142, 0.2040, 0.0577, 0.1652, 0.0011, 0.8029, 0.2753, 0.0834, 0.0127, 1.0000])
#attack = np.array([0.5239, 0.1329, 1.0000, 0.4248, 0.3545, 0.1186, 0.2915, 0.2734, 0.2552, 0.3720, 1.0000, 0.2049, 0.2153, 0.0603, 0.1125, 0.2792, 0.1817, 0.1935, 0.0463, 0.4211, 0.3960, 0.0895, 0.1215, 0.529])
#attack = np.array([0.0506, 0.0454, 0.0447, 0.0499, 0.0497, 0.0500, 0.0492, 0.0495, 0.0496, 0.0485, 0.0470, 0.0495, 0.0494, 0.0470, 0.0485, 0.0497, 0.0497, 0.0479, 0.0483, 0.0494, 0.0492, 0.0486, 0.0488, 0.0497])
attack = np.array([0.2168, 0.0009, 0.0010, 0.0540, 0.0483, 0.0449, 0.0111, 0.0345, 0.0334, 0.0048, 0.0032, 0.0357, 0.0321, 0.0024, 0.0053, 0.0375, 0.0368, 0.0025, 0.0040, 0.0335, 0.0341, 0.0037, 0.0023, 0.0547])

# get the means and covariances split into p and q groups (for doing kl) with attack by frame in spectral space
p_means, p_covariances, q_means, q_covariances, mask = get_pdf_attack(pkl, phones, attack)

# get output labels
y = (pkl['score'])
y = np.array(y)
y = y[:50] # temp


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

# Load the model
model_path = "FCC_lpron_seed1.pt"
model = torch.load(model_path)
model.eval()

# get predictions
y_pred = model(p_means, p_covariances, q_means, q_covariances, mask)
y_pred[y_pred>6]=6.0
y_pred[y_pred<0]=0.0
y_pred_list = y_pred.tolist()
attack_avg_grade = torch.mean(y_pred)


print("Results for spectral attack by frame, with attack model: "+attack_model_path)

# get the noise
#noise = attack_model.get_noise()
noise = attack
print("Spectral noise: ", noise)

# Stats with attack
print("---------------------------------------------------------")
print("STATS with attack")

mse = calculate_mse(y_pred_list, y.tolist())
pcc = calculate_pcc(y_pred, y)
less1 = calculate_less1(y_pred, y)
less05 = calculate_less05(y_pred, y)

print("mse: "+ str(mse)+"\n pcc: "+str(pcc)+"\n less than 1 away: "+ str(less1)+"\n less than 0.5 away: "+str(less05))

print("------------------------------------------------------------")

print("Attacked average grade: ", attack_avg_grade)

# Get stats with no attack

# Load the means and covariances
input_file = 'BLXXXgrd02_means_covs.npz'
#input_file = 'BLXXXeval3_means_covs.npz'
npzfile = np.load(input_file)
p_means = npzfile['arr_0'][:50]
p_covariances = npzfile['arr_1'][:50]
q_means = npzfile['arr_2'][:50]
q_covariances = npzfile['arr_3'][:50]
mask = npzfile['arr_4'][:50]
y = npzfile['arr_5'][:50]

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



y_pred = model(p_means, p_covariances, q_means, q_covariances, mask)
y_pred[y_pred>6]=6.0
y_pred[y_pred<0]=0.0
y_pred_list = y_pred.tolist()
no_attack_avg_grade = torch.mean(y_pred)

print("Not attacked average grade: ", no_attack_avg_grade)

