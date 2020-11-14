import torch
import numpy
from pkl2pqvects import get_phones, get_vects

def get_pdf(obj, phones):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector
    num_spk = len(obj['plp'])
    num_spk = 2 # temp

    # Define the tensors required by spectral attack model
    p_means = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    p_covariances = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5), n, n))
    q_means = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5) , n))
    q_covariances = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5), n, n))
    num_phones_mask = np.zeros((num_spk, int((len(phones)-1)*(len(phones)-2)*0.5)))


    for spk in range(num_spk):
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

                p_means[spk][k] = SX[i].squeeze()
                q_means[spk][k] = SX[j].squeeze()

                k += 1

    return p_means, p_covariances, q_means, q_covariances, num_phones_mask


spk = 0
phone = 2


# Get all the p/q vects
pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
pkl = pickle.load(open(pkl_file, "rb"))
print("loaded pkl")
# get the phones
phones = get_phones()
max_len_frames = 4000
p_vects, q_vects, p_mask, q_mask, mask = get_vects(pkl, phones, max_len_frames)

# Apply torch operations
# Get p/q_lengths
p_lengths = torch.sum(p_frames_mask[:,:,:,0].squeeze(), dim=2).unsqueeze(dim=2).repeat(1,1,13)
q_lengths = torch.sum(q_frames_mask[:,:,:,0].squeeze(), dim=2).unsqueeze(dim=2).repeat(1,1,13)

# Compute means
p_means_tr = torch.sum(p_vects, dim=2)/p_lengths
q_means_tr = torch.sum(q_vects, dim=2)/q_lengths

p_means, p_covariances, q_means, q_covariances, num_phones_mask = get_pdf(pkl, phones)


print("Torch:")
print(p_means_tr[0][2])
print("Numpy:")
print(p_means)
