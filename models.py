import torch
import torch.nn.functional as F


class FCC(torch.nn.Module):
    def __init__(self, num_features):

        super(FCC, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.fc1 = torch.nn.Linear(num_features, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 1000)
        self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc5 = torch.nn.Linear(1000, 1000)
        self.fc6 = torch.nn.Linear(1000, 1000)
        self.fc7 = torch.nn.Linear(1000, 1)
        self.drop_layer = torch.nn.Dropout(p=0.5)

    def forward(self, X):

        # Check if input has nan
        nan_list = torch.isnan(torch.sum(X, 1)).tolist()
        if True in nan_list:
           print("some values from input are nan")

        # Normalize each input vector
        X_norm = self.bn1(X)

        h1 = F.relu(self.fc1(X_norm))
        h2 = F.relu(self.fc2(self.drop_layer(h1)))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(self.drop_layer(h3)))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        y = self.fc7(h6)
        return y.squeeze()

class FCC_shallow(torch.nn.Module):
    def __init__(self, num_features):

        super(FCC_shallow, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.fc1 = torch.nn.Linear(num_features, 1000)
        self.fc2 = torch.nn.Linear(1000, 30)
        self.fc3 = torch.nn.Linear(30, 30)
        self.fc4 = torch.nn.Linear(30, 1)
	    #self.drop_layer = torch.nn.Dropout(p=0.5)

    def forward(self, X):

        # Check if input has nan
        nan_list = torch.isnan(torch.sum(X, 1)).tolist()
        if True in nan_list:
           print("some values from input are nan")

        # Normalize each input vector
        X_norm = self.bn1(X)

        h1 = F.relu(self.fc1(X_norm))
        h2 = F.relu(self.fc2(h1))
        y = self.fc4(h2)
        return y.squeeze()
