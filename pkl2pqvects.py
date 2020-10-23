import numpy as np
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


def get_vects(obj, phones, max_num_mfccs_length):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector

    # Define the tensors required by spectral attack model
    p_vects = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5) , max_num_mfccs_length, n))
    q_vects = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5) , max_num_mfccs_length, n))
    num_phones_mask = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5)))
    p_mfcc_lengths = np.zeros((len(obj['plp']),  int((len(phones)-1)*(len(phones)-2)*0.5)))
    q_mfcc_lengths = np.zeros((len(obj['plp']),  int((len(phones)-1)*(len(phones)-2)*0.5)))


    for spk in range(len(obj['plp'])):
        print("On speaker " + str(spk) + " of " + str(len(obj['plp'])))
        Xs = np.zeros((len(phones) - 1, max_num_mfccs_length, n))
        N = np.zeros(len(phones) - 1)

        for utt in range(len(obj['plp'][spk])):
            for w in range(len(obj['plp'][spk][utt])):
                for ph in range(len(obj['plp'][spk][utt][w])):
                    # n.b. this is iterating through the phones that occur sequentially in a word
                    for frame in range(len(obj['plp'][spk][utt][w][ph])):
                        N[obj['phone'][spk][utt][w][ph]] += 1
                        X = np.array(obj['plp'][spk][utt][w][ph][frame])
                        curr_pos = N[obj['phone'][spk][utt][w][ph]] - 1
                        curr_pos = int(curr_pos.item())
                        Xs[obj['phone'][spk][utt][w][ph]][curr_pos] = X


        # Consturct every unique pairing of phones, related mfcc vectors
        k = 0
        for i in range(len(phones) - 1):
            for j in range(i + 1, len(phones) - 1):
                if N[i] == 0 or N[j] == 0:
                    # define Gaussian distribution that has 0 kl div
                    # later the mask will be used to make these features "-1"
                    num_phones_mask[spk][k] = 0
                    # Store length as 1 (in reality 0), to prevent division by 0 later on
                    p_mfcc_lengths[spk][k] = 1
                    q_mfcc_lengths[spk][k] = 1
                else:
                    num_phones_mask[spk][k] = 1
                    p_mfcc_lengths[spk][k] = N[i]
                    q_mfcc_lengths[spk][k] = N[j]

                p_vects[spk][k] = Xs[i].squeeze()
                q_vects[spk][k] = Xs[j].squeeze()

                k += 1

    return p_vects, q_vects, p_mfcc_lengths, q_mfcc_lengths, num_phones_mask


pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
#pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
pkl = pickle.load(open(pkl_file, "rb"))

print("loaded pkl")

# get the phones
phones = get_phones()

# get the mfcc vects as  [batch_size x 1128 x max_mfccs_per_phone x mfcc_dim] split into p and q groups (for doing kl)
max_num_mfccs_length = 4000
p_vects, q_vects, p_lengths, q_lengths, mask = get_vects(pkl, phones, max_num_mfccs_length)

# get output labels
y = (pkl['score'])
y = np.array(y)

# write to output file
output_file = 'BLXXXgrd02_pqvects.pkl'
#output_file = 'BLXXXeval3_pqvects.npz'
pkl_obj = [p_vects.tolist(), q_vects.tolist(),  p_lengths.tolist(), q_lengths.tolist(), mask.tolist(), y.tolist()]
pickle.dump(pkl_obj, open(output_file, "wb"))
