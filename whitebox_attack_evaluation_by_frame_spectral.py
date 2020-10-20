import numpy as np
import pickle
import torch
import torch_dct as dct
from models import FCC
from attack_models import Spectral_attack

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
    X.squeeze()
    attack = torch.from_numpy(attack).float()

    # Add the attack in the spectral space
    # Pad to spectral dimension
    padding = torch.zeros(attack.size(0) - X.size(0))
    padded_X = torch.cat((X, padding))

    # Apply inverse dct
    log_spectral_X = dct.idct(padded_X)

    # Apply inverse log
    spectral_X = torch.exp(log_spectral_X)

    # Add the adversarial attack
    attacked_spectral_X = spectral_X + attack

    # Get back to mfcc domain
    attacked_log_spectral_X = torch.log(attacked_spectral_X)
    attacked_padded_X = dct.dct(attacked_log_spectral_X)
    X_attacked = torch.narrow(attacked_padded_X, 0, 0, X.size(0))
    X_attacked = X_attacked.numpy()

    return X_attacked


def get_pdf_attack(obj, phones, attack):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector

    # Define the tensors required by spectral attack model
    p_means = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    p_covariances = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5), n, n))
    q_means = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    q_covariances = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5), n, n))
    num_phones_mask = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5)))


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


#pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
pkl = pickle.load(open(pkl_file, "rb"))

print("loaded pkl")

# get the phones
phones = get_phones()

# get the attack spectral vector
attack_model_path = "attack_model_seed1.pt"
attack_model = torch.load(attack_model_path)
attack_model.eval()
attack = attack_model.get_noise()
attack = attack.numpy()


# get the means and covariances split into p and q groups (for doing kl) with attack by frame in spectral space
p_means, p_covariances, q_means, q_covariances, mask = get_pdf_attack(pkl, phones, attack)

# get output labels
y = (pkl['score'])
y = np.array(y)

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

y_list = y.tolist()

mse = calculate_mse(y_pred_list, y_list)
pcc = calculate_pcc(y_pred, y)
less1 = calculate_less1(y_pred, y)
less05 = calculate_less05(y_pred, y)

print("mse: "+ str(mse)+"\n pcc: "+str(pcc)+"\n less than 1 away: "+ str(less1)+"\n less than 0.5 away: "+str(less05))
