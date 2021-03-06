import torch
import torch_dct as dct
from models import FCC

class Spectral_attack_init(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path, init_root):

        super(Spectral_attack_init, self).__init__()

        self.trained_model_path = trained_model_path

        self.noise_root = torch.nn.Parameter(init_root, requires_grad=True)
        # self.noise = torch.exp(self.noise_root)

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
        noise = torch.exp(self.noise_root)

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
        attacked_spectral_p = spectral_p + noise
        attacked_spectral_q = spectral_q + noise

        # Apply the log
        attacked_log_spectral_p = torch.log(attacked_spectral_p)
        attacked_log_spectral_q = torch.log(attacked_spectral_q)

        # Apply the dct
        attacked_padded_p = dct.dct(attacked_log_spectral_p)
        attacked_padded_q = dct.dct(attacked_log_spectral_q)

        # Truncate to mfcc dimension
        p_means_attacked = torch.narrow(attacked_padded_p, 2, 0, self.mfcc_dim)
        q_means_attacked = torch.narrow(attacked_padded_q, 2, 0, self.mfcc_dim)

        # Pass through trained model
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        y = trained_model(p_means_attacked, p_covariances, q_means_attacked, q_covariances, num_phones_mask)

        return y

    def get_preds_no_noise(self, p_means, p_covariances, q_means, q_covariances, num_phones_mask):
        '''
        return the grade predictions with no adversarial attack
        '''
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        return trained_model(p_means, p_covariances, q_means, q_covariances, num_phones_mask)

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return torch.exp(self.noise_root)



class Spectral_attack_Taylor1_init(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path, init_root):

        super(Spectral_attack_Taylor1_init, self).__init__()

        self.trained_model_path = trained_model_path

        self.noise_root = torch.nn.Parameter(init_root, requires_grad=True)
        # self.noise = torch.exp(self.noise_root)

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
        noise = torch.exp(self.noise_root)

        # Need to add spectral noise with first order Taylor approximation
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

        # Hadamard division with the spectral noise
        attacked_spectral_p = noise/spectral_p
        attacked_spectral_q = noise/spectral_q

        # Apply the dct
        attacked_padded_p = dct.dct(attacked_spectral_p)
        attacked_padded_q = dct.dct(attacked_spectral_q)

        # Truncate to mfcc dimension
        p_means_attacked_second_term = torch.narrow(attacked_padded_p, 2, 0, self.mfcc_dim)
        q_means_attacked_second_term = torch.narrow(attacked_padded_q, 2, 0, self.mfcc_dim)

        # Combine Taylor expansion
        p_means_attacked = p_means + p_means_attacked_second_term
        q_means_attacked = q_means + q_means_attacked_second_term

        # Pass through trained model
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        y = trained_model(p_means_attacked, p_covariances, q_means_attacked, q_covariances, num_phones_mask)

        return y

    def get_preds_no_noise(self, p_means, p_covariances, q_means, q_covariances, num_phones_mask):
        '''
        return the grade predictions with no adversarial attack
        '''
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        return trained_model(p_means, p_covariances, q_means, q_covariances, num_phones_mask)

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return torch.exp(self.noise_root)
