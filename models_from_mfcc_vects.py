import torch
import torch.nn.functional as F


class LPRON(torch.nn.Module):
    def __init__(self, num_features):

        super(LPRON, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.fc1 = torch.nn.Linear(num_features, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 1000)
        self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc5 = torch.nn.Linear(1000, 1000)
        self.fc6 = torch.nn.Linear(1000, 1000)
        self.fc7 = torch.nn.Linear(1000, 1)
        self.drop_layer = torch.nn.Dropout(p=0.5)

    def forward(self, p_vects, q_vects, p_lengths, q_lengths, num_phones_mask):
        '''
        p/q_vects = [num_speakers X num_feats X max_num_mfcc_frames x mfcc_dim]
        p/q_lengths = [num_speakers X num_feats] -> stores the number of observed
                                                    frames associated
                                                    with the corresponding phone
        num_phones_mask = [num_speakers X num_feats],
        with a 0 corresponding to position that should be -1 (no phones observed)
        and a 1 everywhere else.
        n.b. mfcc_dim = 13 usually (using c0 for energy instead of log-energy)
             num_feats = 46*47*0.5 = 1128 usually
             max_num_mfcc_frames = the maximum number of frames associated
             with a particular phone for any speaker -> often set to 4000
        '''

        # Compute the p/q_means tensor
        p_means = torch.sum(p_vects, dim=2)/p_lengths
        q_means = torch.sum(q_vects, dim=2)/q_lengths

        # Compute the p/q_covariances tensor
        p_vects_unsq = torch.unsqueeze(p_vects, dim=4)
        q_vects_unsq = torch.unsqueeze(q_vects, dim=4)

        p_vects_unsq_T = torch.transpose(p_vects_unsq, 3, 4)
        q_vects_unsq_T = torch.transpose(q_vects_unsq, 3, 4)

        p_means_squared = torch.squeeze(torch.sum(torch.matmul(p_vects_unsq_T, p_vects_unsq), dim=2)/p_lengths)
        q_means_squared = torch.squeeze(torch.sum(torch.matmul(q_vects_unsq_T, q_vects_unsq), dim=2)/q_lengths)

        p_means_unsq = torch.unsqueeze(p_means, dim=4)
        q_means_unsq = torch.unsqueeze(q_means, dim=4)

        p_means_unsq_T = torch.transpose(p_means_unsq, 2, 3)
        q_means_unsq_T = torch.transpose(q_means_unsq, 2, 3)

        p_m2 = torch.squeeze(torch.matmul(p_means_unsq_T, p_means_unsq))
        q_m2 = torch.squeeze(torch.matmul(q_means_unsq_T, q_means_unsq))

        p_covariances = p_means_squared - p_m2
        q_covariances = q_means_squared - q_m2

        # Add small noise to covariances to prevent non-singular in Cholesky in kl-div
        p_covariances = p_covariances + (1e-3*torch.eye(13))
        q_covariances = q_covariances + (1e-3*torch.eye(13))

        # from here it is the same as class FCC in models.py
        # compute symmetric kl-divergences between every phone distribution per speaker
        p = torch.distributions.MultivariateNormal(p_means, p_covariances)
        q = torch.distributions.MultivariateNormal(q_means, q_covariances)

        kl_loss = ((torch.distributions.kl_divergence(p, q) + torch.distributions.kl_divergence(q, p))*0.5)

        # log all the features
        # add small error to mak 0-kl distances not a NaN
        X = kl_loss + (1e-5)
        feats = torch.log(X)

        # Apply mask to get -1 features in correct place (i.e. where no phones observed)
        feats_shifted = feats + 1
        feats_masked = feats_shifted * num_phones_mask
        feats_correct = feats_masked - 1

        # pass through layers

        # Normalize each input vector
        X_norm = self.bn1(feats_correct)

        h1 = F.relu(self.fc1(X_norm))
        h2 = F.relu(self.fc2(self.drop_layer(h1)))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(self.drop_layer(h3)))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        y = self.fc7(h6)
        return y.squeeze()
