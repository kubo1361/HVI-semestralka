import torch.nn as nn
import torch.nn.functional as F

class Network1(nn.Module):
    def __init__(self, state_dim, actions_count):
        super(Network1, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim

        # input
        self.fci = nn.Linear(self.state_dim, 512)
        # shared
        self.fcs = nn.Linear(512, 512)
        # actor and critic
        self.fca = nn.Linear(512, actions_count)
        self.fcc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fci(x))
        x = F.relu(self.fcs(x))

        outActor = self.fca(x)
        outCritic = self.fcc(x)

        return outActor, outCritic
