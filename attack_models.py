import torch
from models import FCC


class Spectral_attack(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path):

        super(Spectral_attack, self).__init__()
        self.trained_model = torch.load(trained_model_path)
        self.trained_model.eval()

        self.noise_root = torch.nn.Parameter(torch.randn(spectral_dim))
        self.noise = torch.exp(self.noise_root)

    def forward(self, means, covariances):
        '''
        means = [num_speakers * num_phones * mfcc_vect_size]
        covariances = [num_speakers * num_phones * mfcc_vect_size * mfcc_vect_size]
        '''

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return self.noise
