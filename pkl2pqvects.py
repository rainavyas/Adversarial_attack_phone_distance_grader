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

    lengths = []

    # Define the tensors required by spectral attack model
    p_vects = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    q_vects = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    num_phones_mask = np.zeros((len(obj['plp']), int((len(phones)-1)*(len(phones)-2)*0.5)))
    #mfcc_lenghts =


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
            lengths.append(N[ph])
            print(N[ph])
            if N[ph] !=0:
                SX[ph] /= N[ph]
                SX2[ph] /= N[ph]
            m2 = np.matmul(SX[ph], np.transpose(SX[ph]))
            Sig[ph] = SX2[ph] - m2
    print("largest", max(lengths))
        # k = 0
        # for i in range(len(phones) - 1):
        #     for j in range(i + 1, len(phones) - 1):
        #         if N[i] == 0 or N[j] == 0:
        #             num_phones_mask[spk][k] = 0
        #             # define Gaussian distribution that has 0 kl div
        #             # later the mask will be used to make these features "-1"
        #             p_covariances[spk][k] = np.eye(n)
        #             q_covariances[spk][k] = np.eye(n)
        #         else:
        #             num_phones_mask[spk][k] = 1
        #             p_covariances[spk][k] = Sig[i]
        #             q_covariances[spk][k] = Sig[j]
        #
        #         p_means[spk][k] = SX[i].squeeze()
        #         q_means[spk][k] = SX[j].squeeze()
        #
        #         k += 1

    return p_means, p_covariances, q_means, q_covariances, num_phones_mask


pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
#pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
pkl = pickle.load(open(pkl_file, "rb"))

print("loaded pkl")

# get the phones
phones = get_phones()

# get the mfcc vects as  [batch_size x 1128 x max_mfccs_per_phone x mfcc_dim] split into p and q groups (for doing kl)
max_num_mfccs_length = 100
p_vects, q_vects, mask, lengths = get_vects(pkl, phones, max_num_mfccs_length)

# get output labels
y = (pkl['score'])
y = np.array(y)

# # write to output file
# output_file = 'BLXXXgrd02_pqvects.npz'
# #output_file = 'BLXXXeval3_pqvects.npz'
# np.savez(output_file, p_vects, q_vects, mask, lengths, y)
