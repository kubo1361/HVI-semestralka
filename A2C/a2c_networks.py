import torch.nn as nn
from RBF.Torch_RBF.torch_rbf import RBF, gaussian


class A2CDB(nn.Module):
    def __init__(self, state_dim, actions_count, n_neurons=256, shared_layers=3):
        super(A2CDB, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            nn.ReLU()
        ]
        
        for i in range(shared_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)

        self.actor = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class A2CCB(nn.Module):
    def __init__(self, state_dim, actions_count, n_neurons=256, shared_layers=3):
        super(A2CCB, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            nn.ReLU()
        ]
        
        for i in range(shared_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)

        self.median = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Tanh(),
        )
        self.variance = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Softplus(),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.median(shared_out), self.variance(shared_out), self.critic(shared_out)

class A2CDRBF1NA(nn.Module):
    def __init__(self, state_dim, actions_count,  n_neurons=256, shared_layers=3, basis_func=gaussian):
        super(A2CDRBF1NA, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            RBF(n_neurons, n_neurons, self.basis_func),
        ]
        
        for i in range(shared_layers - 2):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)

        self.actor = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class A2CDRBF1A(nn.Module):
    def __init__(self, state_dim, actions_count,  n_neurons=256, shared_layers=3, basis_func=gaussian):
        super(A2CDRBF1A, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            RBF(n_neurons, n_neurons, self.basis_func),
            nn.ReLU()
        ]
        
        for i in range(shared_layers - 2):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)
        self.actor = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class A2CCRBF1NA(nn.Module):
    def __init__(self, state_dim, actions_count,  n_neurons=256, shared_layers=3, basis_func=gaussian):
        super(A2CCRBF1NA, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            RBF(n_neurons, n_neurons, self.basis_func),
        ]
        
        for i in range(shared_layers - 2):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)
        
        self.median = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Tanh(),
        )
        self.variance = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Softplus(),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.median(shared_out), self.variance(shared_out), self.critic(shared_out)

class A2CCRBF1A(nn.Module):
    def __init__(self, state_dim, actions_count,  n_neurons=256, shared_layers=3, basis_func=gaussian):
        super(A2CCRBF1A, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            RBF(n_neurons, n_neurons, self.basis_func),
            nn.ReLU()
        ]
        
        for i in range(shared_layers - 2):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)

        self.median = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Tanh(),
        )
        self.variance = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Softplus(),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.median(shared_out), self.variance(shared_out), self.critic(shared_out)

# ----------------------------------------------------------------------------------------------

class A2CCRBF(nn.Module):
    def __init__(self, state_dim, actions_count,  n_neurons=256, shared_layers=2, basis_func=gaussian):
        super(A2CCRBF, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            RBF(n_neurons, n_neurons, self.basis_func),
        ]

        for i in range(shared_layers - 2):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())
        
        self.shared = nn.Sequential(*layers)
        
        self.median = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Tanh(),
        )
        self.variance = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
            nn.Softplus(),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.median(shared_out), self.variance(shared_out), self.critic(shared_out)

class A2CDRBF(nn.Module):
    def __init__(self, state_dim, actions_count,  n_neurons=256, shared_layers=2, basis_func=gaussian):
        super(A2CDRBF, self).__init__()
        self.actions_count = actions_count
        self.state_dim = state_dim
        self.basis_func = basis_func

        layers = [
            nn.Linear(self.state_dim, n_neurons),
            RBF(n_neurons, n_neurons, self.basis_func),
        ]

        for i in range(shared_layers - 2):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)

        self.actor = nn.Sequential(
            nn.Linear(n_neurons, self.actions_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_neurons, 1),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)