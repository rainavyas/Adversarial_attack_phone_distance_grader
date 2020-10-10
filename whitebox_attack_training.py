import torch
import numpy


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

    # Store all the means: num_spkrs x 48 x 13 x 1
    means = np.zeros((len(obj['plp']), len(phones)-1 , n, 1))

    # Store variances: num_spkrs x 48 x (13 x 13) -> ie 13x13 covariance matrix for each of the 48 phones for each speaker
    variances = np.zeros((len(obj['plp']), len(phones)-1, n, n))

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

        means[spk] = SX
        variances[spk] = Sig

    return means, variances



pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
pkl = pickle.load(open(pkl_file, "rb"))

# get the phones
phones = get_phones()

# get the means and covariances
means, covs = get_pdf(pkl, phones)
