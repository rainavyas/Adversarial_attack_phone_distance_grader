import torch
from models import FCC


class Spectral_attack(torch.nn.Module):
    def __init__(self, size, trained_model_path):

        super(Spectral_attack, self).__init__()
        self.trained_model = torch.load(trained_model_path)
        self.trained_model.eval()

        self.noise_root = torch.nn.Parameter(torch.randn(size))
        self.noise = torch.exp(self.noise_root)

    def forward(self, means, covariances):

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return self.noise
