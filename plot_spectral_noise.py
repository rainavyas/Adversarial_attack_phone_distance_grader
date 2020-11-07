import matplotlib.pyplot as plt
import pickle
from attack_models import Spectral_attack
import numpy as np
import torch
import torch_dct as dct


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

def spectral_convert(X, num_channels):
    X = torch.from_numpy(X).float()
    X_sq = X.squeeze()
    # Pad to spectral dimension
    padding = torch.zeros(num_channels - X_sq.size(0))
    padded_X = torch.cat((X_sq, padding))

    # Apply inverse dct
    log_spectral_X = dct.idct(padded_X)

    # Apply inverse log
    spectral_X = torch.exp(log_spectral_X)

    # Convert back to numpy
    spectral_X = spectral_X.detach().numpy()

    return spectral_X

def get_spectral_vects(obj, spk, phone, num_channels):
    spectral_vects = []

    for utt in range(len(obj['plp'][spk])):
        for w in range(len(obj['plp'][spk][utt])):
            for ph in range(len(obj['plp'][spk][utt][w])):
                # Check we are at the correct phone
                curr_phone = obj['phone'][spk][utt][w][ph]
                if curr_phone != phone:
                    continue
                for frame in range(len(obj['plp'][spk][utt][w][ph])):
                    X = np.array(obj['plp'][spk][utt][w][ph][frame])
                    X_spectral = spectral_convert(X, num_channels)
                    spectral_vects.append(X_spectral.tolist())

    return spectral_vects

pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
#pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
pkl = pickle.load(open(pkl_file, "rb"))


# get the attack spectral vector
attack_model_path = "attack_model_seed1.pt"
attack_model = torch.load(attack_model_path)
attack_model.eval()
attack = attack_model.get_noise()
attack = attack.detach().numpy()

# Choose speaker and phone to see channel energies for
spk = 723
phone = 35
num_channels = 24

phones = get_phones()
phone_letters = phones[phone]

spectral_vects = get_spectral_vects(pkl, spk, phone, num_channels)
spectral_vects = np.array(spectral_vects)

# Get the attacked spectral vectors too
attacked_spectral_vects = spectral_vects + attack

# Calculate the mean and standard deviation for each channel
channel_means = np.mean(spectral_vects, axis=0)
channel_stds = np.std(spectral_vects, axis=0)
attacked_channel_means = np.mean(attacked_spectral_vects, axis=0)

# Plot the bar graph
plt.style.use('ggplot')

channel_means = channel_means.tolist()
channel_stds = channel_stds.tolist()
attacked_channel_means = attacked_channel_means.tolist()

ind = np.arange(num_channels)
width = 0.45
x = [str(i) for i in range(num_channels)]
plt.bar(ind, channel_means, width, color='blue', yerr=channel_stds, label='Spectral Vectors Mean')
plt.bar(ind+width, attacked_channel_means, width, color='red', label='Attacked Spectral Vectors Mean')
plt.xticks(ind+width/2, x)
plt.xlabel("Spectrum Channel")
plt.ylabel("Spectral Energy")
plt.title("BLXXXgrd02 Data: Speaker = " + str(spk)+ ", Phone = "+str(phone) + " ('"+phone_letters+"')")
plt.legend(loc='best')
plt.ylim([0,1000])

# Save the figure as an image
plt.tight_layout()
plt.savefig('Spectral_Plot')
