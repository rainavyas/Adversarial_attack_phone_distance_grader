import torch
import torch_dct as dct
from models import FCC

class Spectral_attack_montecarlo(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path, init_root, sample_size=10):

        super(Spectral_attack_montecarlo, self).__init__()

        self.trained_model_path = trained_model_path

        self.noise_root = torch.nn.Parameter(init_root, requires_grad=True)
        # self.noise = torch.exp(self.noise_root)

        self.spectral_dim = spectral_dim
        self.mfcc_dim = mfcc_dim
        self.sample_size = sample_size

    def attack(self, samples, noise):
        '''
        Perform attack in the spectral space
        '''
        # Pad to spectral dimension
        padding = torch.zeros(samples.size(0), samples.size(1), samples.size(2), self.spectral_dim - self.mfcc_dim).to(self.device)
        padded_samples = torch.cat((samples, padding), 3)

        # Apply inverse dct
        log_spectral = dct.idct(padded_samples)

        # Apply inverse log
        spectral = torch.exp(log_spectral)

        # Add the adversarial attack noise
        attacked_spectral = spectral + noise

        # Apply the log
        attacked_log_spectral = torch.log(attacked_spectral)

        # Apply the dct
        attacked_padded = dct.dct(attacked_log_spectral)

        # Truncate to mfcc dimension
        samples_attacked = torch.narrow(attacked_padded, 3, 0, self.mfcc_dim)

        return samples_attacked

    def compute_mean_cov(self, vects):
        '''
        Return the mean and covariance matrices
        '''
        mean = torch.mean(vects, dim=-2)

        # Compute the covariance
        vects_unsq = torch.unsqueeze(vects, dim=-1)
        vects_unsq_T = torch.transpose(vects_unsq, -1, -2)
        vects_squared = torch.matmul(vects_unsq, vects_unsq_T)
        vects_squared_mean = torch.mean(vects_squared, dim=-3)

        means_unsq = torch.unsqueeze(mean, dim=-1)
        means_unsq_T = torch.transpose(means_unsq, -1, -2)
        means_squared = torch.matmul(means_unsq, means_unsq_T)

        cov = vects_squared_mean - means_squared

	# Make cov matrix diagonal to guarantee it is positive definite
        cov_diag = torch.diag_embed(torch.diagonal(cov, offset=0, dim1=-2, dim2=-1))

        #return mean, cov_diag
        return mean,  cov

    def forward(self, p_means, p_covariances, q_means, q_covariances, num_phones_mask):
        '''
        p/q_means = [num_speakers X num_feats X mfcc_dim]
        p/q_covariances = [num_speakers X num_feats X mfcc_dim X mfcc_dim]
        num_phones_mask = [num_speakers X num_feats],
        with a 0 corresponding to positiion that should be -1 (no phones observed)
        and a 1 everywhere else.
        n.b. num_feats = 46*47*0.5 = 1128 usually
        '''

        # Define Gaussian distribution
        p = torch.distributions.MultivariateNormal(p_means, p_covariances)
        q = torch.distributions.MultivariateNormal(q_means, q_covariances)

        # Sample from the distributions
        p_samples = p.sample((self.sample_size,)).to(self.device)
        q_samples = q.sample((self.sample_size,)).to(self.device)

        ps = torch.reshape(p_samples, (p_samples.size(1), p_samples.size(2), p_samples.size(0), p_samples.size(3)))
        qs = torch.reshape(q_samples, (q_samples.size(1), q_samples.size(2), q_samples.size(0), q_samples.size(3)))

        # Spectral attack the samples
        noise = torch.exp(self.noise_root)
        ps_attacked = self.attack(ps, noise)
        qs_attacked = self.attack(qs, noise)

        # Compute mean and covariance from samples
        p_means_attacked, p_covs_attacked = self.compute_mean_cov(ps_attacked)
        q_means_attacked, q_covs_attacked = self.compute_mean_cov(qs_attacked)

        # add small noise to all covariance matrices to ensure they are non-singular
        p_covariances_noised = p_covs_attacked + (1e-2*torch.eye(13).to(self.device))
        q_covariances_noised = q_covs_attacked + (1e-2*torch.eye(13).to(self.device))

        print("Before sampling")
        print(p_means[3,2,:])
        print("After Sampling")
        print(p_means_attacked[3,2,:])

        # Pass through trained model
        trained_model = torch.load(self.trained_model_path)
        trained_model.to(self.device)
        trained_model.eval()
        y = trained_model(p_means_attacked, p_covariances_noised, q_means_attacked, q_covariances_noised, num_phones_mask)
        #y = trained_model(p_means_attacked, p_covariances, q_means_attacked, q_covariances, num_phones_mask)

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
        super(Spectral_attack_montecarlo, self).to(device)
        self.device = device
