import torch
import torch_dct as dct
from models import FCC

class Spectral_attack(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path):

        super(Spectral_attack, self).__init__()
        self.trained_model = torch.load(trained_model_path)
        self.trained_model.eval()

        self.noise_root = torch.nn.Parameter(torch.randn(spectral_dim))
        self.noise = torch.exp(self.noise_root)

        self.spectral_dim = spectral_dim
        self.mfcc_dim = mfcc_dim


    def forward(self, p_means, p_covariances, q_means, q_covariances, num_phones_mask):
        '''
        p/q_means = [num_speakers X num_feats X mfcc_dim]
        p/q_covariances = [num_speakers X num_feats X mfcc_dim X mfcc_dim]
        num_phones_mask = [num_speakers X num_feats],
        with a 0 corresponding to positiion that should be -1 (no phones observed)
        and a 1 everywhere else.
        n.b. num_feats = 46*47*0.5 = 1128 usually, where 47 = num_phones
        '''

        # Need to add spectral noise
        # Pad to spectral dimension
        padding = torch.zeros(p_means.size(0), p_means.size(1), self.spectral_dim - self.mfcc_dim)
        padded_p_means = torch.cat((p_means, padding), 2)
        padded_q_means = torch.cat((q_means, padding), 2)

        # Apply inverse dct
        log_spectral_p = dct.idct(padded_p_means)
        log_spectral_q = dct.idct(padded_q_means)

        # Apply inverse log
        spectral_p = torch.exp(log_spectral_p)
        spectral_q = torch.exp(log_spectral_q)

        # Add the adversarial attack noise
        attacked_spectral_p = spectral_p + self.noise
        attacked_spectral_q = spectral_q + self.noise

        # Apply the log
        attacked_log_spectral_p = torch.log(attacked_spectral_p)
        attacked_log_spectral_q = torch.log(attacked_spectral_q)

        # Apply the dct
        attacked_padded_p = dct.dct(attacked_log_spectral_p)
        attacked_padded_q = dct.dct(attacked_log_spectral_q)

        # Truncate to mfcc dimension
        p_means_attacked = torch.narrow(2, 0, self.mfcc_dim)
        q_means_attacked = torch.narrow(2, 0, self.mfcc_dim)

        # Pass through trained model
        y = self.trained_model(p_means_attacked, p_covariances, q_means_attacked, q_covariances, num_phones_mask)

        return y

    def get_preds_no_noise(self, p_means, p_covariances, q_means, q_covariances, num_phones_mask):
        '''
        return the grade predictions with no adversarial attack
        '''
        return self.trained_model(p_means, p_covariances, q_means, q_covariances, num_phones_mask)

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return self.noise
