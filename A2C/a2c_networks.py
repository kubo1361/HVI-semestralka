import torch.nn as nn
import torch.nn.functional as F
from RBF.Torch_RBF.torch_rbf import RBF, gaussian


class A2CDiscreteNetwork1(nn.Module):
    def __init__(self, state_dim, actions_count):
        super(A2CDiscreteNetwork1, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim

        self.shared = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(256, self.actions_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)


# TODO pouzi a2c continous siet ako benchmark, sprav pocet neuronov, spolocnych vrstiev, aktor-kritik vrstiev ako parameter


class A2CContinuousNetwork1(nn.Module):
    def __init__(self, state_dim, actions_count):
        super(A2CContinuousNetwork1, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim

        self.shared = nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
        )

        self.median = nn.Sequential(
            nn.Linear(512, self.actions_count),
            nn.Tanh(),
        )
        self.variance = nn.Sequential(
            nn.Linear(512, self.actions_count),
            nn.Softplus(),
        )

        self.critic = nn.Sequential(
            nn.Linear(512, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.median(shared_out), self.variance(shared_out), self.critic(shared_out)


class RBFnetwork1(nn.Module):
    def __init__(self, state_dim, actions_count, basis_func=gaussian):
        super(RBFnetwork1, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

        # input
        self.fci = RBF(self.state_dim, 512, self.basis_func)
        # shared
        self.fcs1 = nn.Linear(512, 512)
        self.fcs2 = RBF(512, 512, self.basis_func)
        # actor and critic
        self.fca = nn.Linear(512, actions_count)
        self.fcc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fci(x)
        x = self.fcs1(x)
        x = self.fcs2(x)

        outActor = self.fca(x)
        outCritic = self.fcc(x)

        return outActor, outCritic
