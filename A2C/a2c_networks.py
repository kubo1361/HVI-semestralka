import torch.nn as nn
import torch.nn.functional as F
from RBF.Torch_RBF.torch_rbf import RBF, gaussian

class Network1(nn.Module):
    def __init__(self, state_dim, actions_count):
        super(Network1, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim

        # input
        self.fci = nn.Linear(self.state_dim, 512)
        # shared
        self.fcs1 = nn.Linear(512, 512)
        self.fcs2 = nn.Linear(512, 512)
        # actor and critic
        self.fca = nn.Linear(512, actions_count)
        self.fcc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fci(x))
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fcs2(x))

        outActor = self.fca(x)
        outCritic = self.fcc(x)

        return outActor, outCritic

class Network2(nn.Module):
    def __init__(self, state_dim, actions_count, basis_func=gaussian):
        super(Network2, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

         # input
        self.fci =RBF(self.state_dim, 512, self.basis_func)
        # shared
        self.fcs1 = nn.Linear(512, 512)
        self.fcs2 =RBF(512, 512, self.basis_func)
        # actor and critic
        self.fca = nn.Linear(512, actions_count)
        self.fcc = nn.Linear(512, 1)


    def forward(self, x):
        x =self.fci(x)
        x = self.fcs1(x)
        x = self.fcs2(x)

        outActor = self.fca(x)
        outCritic = self.fcc(x)

        return outActor, outCritic