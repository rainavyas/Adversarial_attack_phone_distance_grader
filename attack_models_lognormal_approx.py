import torch
import torch_dct as dct
from models import FCC

class Spectral_attack_lognormal(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path, init_root):

        super(Spectral_attack_lognormal, self).__init__()
        self.trained_model_path = trained_model_path
        self.noise_root = torch.nn.Parameter(init_root, requires_grad=True)
        self.spectral_dim = spectral_dim
        self.mfcc_dim = mfcc_dim

    def attack_mean(self, means, noise):
        # Need to add spectral noise
        # Pad to spectral dimension
        padding = torch.zeros(means.size(0), means.size(1), self.spectral_dim - self.mfcc_dim).to(self.device)
        padded_means = torch.cat((means, padding), 2)

        # Apply inverse dct
        log_spectral = dct.idct(padded_means)

        # Apply inverse log
        spectral = torch.exp(log_spectral)

        # Add the adversarial attack noise
        attacked_spectral = spectral + noise

        # Apply the log
        attacked_log_spectral = torch.log(attacked_spectral)

        # Apply the dct
        attacked_padded = dct.dct(attacked_log_spectral)

        # Truncate to mfcc dimension
        means_attacked = torch.narrow(attacked_padded, 2, 0, self.mfcc_dim)

        return means_attacked

    def attack_cov(self, means, covs, means_atck, noise):
        '''
        This update is derived from a log normal
        approximation of a shifted log normal distribution
        '''
        step1 = dct.idct(covs)
        step2 = torch.transpose(dct.idct(torch.transpose(step1, -1, -2)), -1, -2)
        step3 = torch.diagonal(step2, offset=0, dim1=-2, dim2=-1)
        step4 = dct.idct(means) + (step3*0.5)
        padding = torch.zeros(means.size(0), means.size(1), self.spectral_dim - self.mfcc_dim).to(self.device)
        padded_step4 = torch.cat((step4, padding), 2)
        step5 = torch.exp(padded_step4) + noise
        step6 = torch.log(step5)*2
        step6_trunc = torch.narrow(step6, 2, 0, self.mfcc_dim)
        step7 = step6_trunc - (2*dct.idct(means_atck))
        step8 = torch.diag_embed(step7)
        step9 = dct.dct(step8)
        step10 = torch.transpose(dct.dct(torch.transpose(step9, -1, -2)), -1, -2)

        # Make sure no negative diagonal values
        step11 = torch.diag_embed(torch.clamp(torch.diagonal(step10, offset=0, dim1=-2, dim2=-1), min=1.0))

        stepa = torch.diagonal(covs, offset=0, dim1=-2, dim2=-1)
        stepb = torch.diag_embed(stepa)

        #attacked_covs = covs - stepb + step11
        attacked_covs = step11 # Have to neglect off-diagonal terms to ensure covariance matrices are positive definite
        noised = attacked_covs + (1e-2*torch.eye(13).to(self.device))

        return noised


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

        # Attack the means
        p_means_attacked = self.attack_mean(p_means, noise)
        q_means_attacked = self.attack_mean(q_means, noise)

        # Attack the covariances
        p_covs_attacked = self.attack_cov(p_means, p_covariances, p_means_attacked, noise)
        q_covs_attacked = self.attack_cov(q_means, q_covariances, q_means_attacked, noise)

        # Pass through trained model
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        y = trained_model(p_means_attacked, p_covs_attacked, q_means_attacked, q_covs_attacked, num_phones_mask)


        return y.clamp(min=0.0, max=6.0)

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

    def to(self, device):
        super(Spectral_attack_lognormal, self).to(device)
        self.device = device
