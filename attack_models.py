import torch
from models import FCC


# class Spectral_attack(torch.nn.Module):
#     def __init__(self, spectral_dim, mfcc_dim, trained_model_path):
#
#         super(Spectral_attack, self).__init__()
#         self.trained_model = torch.load(trained_model_path)
#         self.trained_model.eval()
#
#         self.noise_root = torch.nn.Parameter(torch.randn(spectral_dim))
#         self.noise = torch.exp(self.noise_root)
#
#         self.spectral_dim = spectral_dim
#         self.mfcc_dim = mfcc_dim
#
#     def kl_div(self, mu1, mu2, sig1, sig2):
#         '''
#         return symmetric kl-div between two Gaussian pdfs
#         '''
#
#         s0 = sig1+1e-3 * torch.eye(self.mfcc_dim)
#         s1 = sig2+1e-3 * torch.eye(self.mfcc_dim)
#         s1m =
#
#     def get_features(self, means_by_phone, covs_by_phone):
#
#
#
#     def forward(self, means, covariances, num_phones_mask):
#         '''
#         means = [num_speakers * num_phones * mfcc_dim]
#         covariances = [num_speakers * num_phones * mfcc_dim * mfcc_dim]
#         num_phones_mask = [num_speakers * (mfcc_dim x mfcc_dim)],
#         with a 0 corresponding to positiion that should be -1 (no phones observed)
#         and a 1 everywhere else.
#         '''
#
#         p = torch.distributions.MultivariateNormal(means, covariances)
#
#
#
#
#         # Make all features that should be -1, -1 using mask
#         feats_shifted = feats + 1
#         feats_masked = feats_shifted * num_phones_mask
#         feats_correct = feats_masked - 1
#
#         # Pass through trained model
#         y = self.trained_model(feats_correct)
#
#         return y
#
#     def get_preds_no_noise(self, means, covariances):
#         '''
#         return the grade predictions with no adversarial attack
#         '''
#
#     def get_noise(self):
#         '''
#         return the spectral noise vector
#         '''
#         return self.noise



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
        n.b. num_feats = 46*47*0.5 = 1128 usually
        '''

        p = torch.distributions.MultivariateNormal(means, covariances)
        



        # Make all features that should be -1, -1 using mask
        feats_shifted = feats + 1
        feats_masked = feats_shifted * num_phones_mask
        feats_correct = feats_masked - 1

        # Pass through trained model
        y = self.trained_model(feats_correct)

        return y

    def get_preds_no_noise(self, means, covariances):
        '''
        return the grade predictions with no adversarial attack
        '''

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return self.noise
