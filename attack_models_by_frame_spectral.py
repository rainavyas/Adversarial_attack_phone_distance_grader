import torch
import torch_dct as dct
from models import FCC

class Spectral_attack_by_frame(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path, init_root):

        super(Spectral_attack_by_frame, self).__init__()

        self.trained_model_path = trained_model_path

        self.noise_root = torch.nn.Parameter(init_root, requires_grad=True)
        # self.noise = torch.exp(self.noise_root)

        self.spectral_dim = spectral_dim
        self.mfcc_dim = mfcc_dim

    def get_pq_means_covs(self, p_vects, q_vects, p_frames_mask, q_frames_mask, num_phones_mask):

        # Get p/q_lengths
        p_lengths = torch.sum(p_frames_mask[:,:,:,0].squeeze(), dim=2).unsqueeze(dim=2).repeat(1,1,13)
        q_lengths = torch.sum(q_frames_mask[:,:,:,0].squeeze(), dim=2).unsqueeze(dim=2).repeat(1,1,13)

        # Compute means
        p_means = torch.sum(p_vects, dim=2)/p_lengths
        q_means = torch.sum(q_vects, dim=2)/q_lengths

        # Compute the p/q_covariances tensor
        p_vects_unsq = torch.unsqueeze(p_vects, dim=4)
        q_vects_unsq = torch.unsqueeze(q_vects, dim=4)

        p_vects_unsq_T = torch.transpose(p_vects_unsq, 3, 4)
        q_vects_unsq_T = torch.transpose(q_vects_unsq, 3, 4)

        p_means_squared = torch.squeeze(torch.sum(torch.matmul(p_vects_unsq, p_vects_unsq_T), dim=2)/p_lengths.unsqueeze(dim=3).repeat(1,1,1,13))
        q_means_squared = torch.squeeze(torch.sum(torch.matmul(q_vects_unsq, q_vects_unsq_T), dim=2)/q_lengths.unsqueeze(dim=3).repeat(1,1,1,13))

        p_means_unsq = torch.unsqueeze(p_means, dim=3)
        q_means_unsq = torch.unsqueeze(q_means, dim=3)

        p_means_unsq_T = torch.transpose(p_means_unsq, 2, 3)
        q_means_unsq_T = torch.transpose(q_means_unsq, 2, 3)

        p_m2 = torch.squeeze(torch.matmul(p_means_unsq, p_means_unsq_T))
        q_m2 = torch.squeeze(torch.matmul(q_means_unsq, q_means_unsq_T))

        p_covariances = p_means_squared - p_m2
        q_covariances = q_means_squared - q_m2

        # If no phone, make covariance matrix identity
        p_covariances_shifted = p_covariances - torch.eye(13)
        q_covariances_shifted = q_covariances - torch.eye(13)

        p_covariances_shifted_masked = p_covariances_shifted * num_phones_mask.unsqueeze(dim=2).repeat(1,1,13).unsqueeze(dim=3).repeat(1,1,1,13)
        q_covariances_shifted_masked = q_covariances_shifted * num_phones_mask.unsqueeze(dim=2).repeat(1,1,13).unsqueeze(dim=3).repeat(1,1,1,13)

        p_covs = p_covariances_shifted_masked + torch.eye(13)
        q_covs = q_covariances_shifted_masked + torch.eye(13)

        return p_means, p_covs, q_means, q_covs


    def forward(self, p_vects, q_vects, p_frames_mask, q_frames_mask, num_phones_mask):
        '''
        p/q_vects = [num_speakers X num_feats X max_num_mfcc_frames x mfcc_dim]
        p/q_lengths = [num_speakers X num_feats] -> stores the number of observed
                                                    frames associated
                                                    with the corresponding phone
        p/q_frames_mask = [num_speakers X num_feats X max_num_mfcc_frames]
                          -> The associated 0s and 1s mask of p/q_lengths
        num_phones_mask = [num_speakers X num_feats],
        with a 0 corresponding to position that should be -1 (no phones observed)
        and a 1 everywhere else.
        n.b. mfcc_dim = 13 usually (using c0 for energy instead of log-energy)
             num_feats = 46*47*0.5 = 1128 usually
             max_num_mfcc_frames = the maximum number of frames associated
             with a particular phone for any speaker -> often set to 4000
        '''
        # Apply the attack
        noise = torch.exp(self.noise_root)

        # Need to add spectral noise
        # Pad to spectral dimension
        padding = torch.zeros(p_vects.size(0), p_vects.size(1), p_vects.size(2), self.spectral_dim - self.mfcc_dim)
        padded_p_vects = torch.cat((p_vects, padding), 3)
        padded_q_vects = torch.cat((q_vects, padding), 3)

        # Apply inverse dct
        log_spectral_p = dct.idct(padded_p_vects)
        log_spectral_q = dct.idct(padded_q_vects)

        # Apply inverse log
        spectral_p = torch.exp(log_spectral_p)
        spectral_q = torch.exp(log_spectral_q)

        # Add the adversarial attack noise
        attacked_spectral_p = spectral_p + noise
        attacked_spectral_q = spectral_q + noise

        # Apply the log
        attacked_log_spectral_p = torch.log(attacked_spectral_p)
        attacked_log_spectral_q = torch.log(attacked_spectral_q)

        # Apply the dct
        attacked_padded_p = dct.dct(attacked_log_spectral_p)
        attacked_padded_q = dct.dct(attacked_log_spectral_q)

        # Truncate to mfcc dimension
        p_vects_attacked = torch.narrow(attacked_padded_p, 3, 0, self.mfcc_dim)
        q_vects_attacked = torch.narrow(attacked_padded_q, 3, 0, self.mfcc_dim)

        # Apply mask of zeros/ones, to ensure spectral noise only applied up to p/q lengths
        p_vects_masked = p_vects_attacked * p_frames_mask
        q_vects_masked = q_vects_attacked * q_frames_mask

        # Compute the p/q_means tensor and covariance tensor
        p_means, p_covariances, q_means, q_covariances = self.get_pq_means_covs(p_vects_masked, q_vects_masked, p_frames_mask, q_frames_mask, num_phones_mask)

        # add small noise to all covariance matrices to ensure they are non-singular
        p_covariances_noised = p_covariances + (1e-3*torch.eye(13))
        q_covariances_noised = q_covariances + (1e-3*torch.eye(13))

#        print(p_covariances_noised[0,3,:,:])
#        print(q_covariances_noised[1,4,:,:])

        # Pass through trained model
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        y = trained_model(p_means, p_covariances_noised, q_means, q_covariances_noised, num_phones_mask)

        return y



    def get_preds_no_noise(self, p_vects, q_vects, p_frames_mask, q_frames_mask, num_phones_mask):
        '''
        return the grade predictions with no adversarial attack
        '''
        # Compute the p/q_means tensor and covariance tensor
        p_means, p_covariances, q_means, q_covariances = self.get_pq_means_covs(p_vects, q_vects, p_frames_mask, q_frames_mask, num_phones_mask)
        # add small noise to all covariance matrices to ensure they are non-singular
        p_covariances_noised = p_covariances + (1e-3*torch.eye(13))
        q_covariances_noised = q_covariances + (1e-3*torch.eye(13))

        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        return trained_model(p_means, p_covariances_noised, q_means, q_covariances_noised, num_phones_mask)

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return torch.exp(self.noise_root)
