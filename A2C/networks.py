import torch.nn as nn
import torch.nn.functional as F

class networkV2(nn.Module):
    def __init__(self, actions_count):
        super(networkV2, self).__init__()
        self.actions_count = actions_count

        self.conv1s = nn.Conv2d(4, 32, 3, stride=2, padding=1)  # B, CH, H, W
        self.conv2s = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3s = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4s = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv5s = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv6s = nn.Conv2d(64, 32, 3, stride=1, padding=1)

        self.fca1 = nn.Linear(5 * 5 * 32, 512)
        self.fcc1 = nn.Linear(5 * 5 * 32, 512)

        self.fca2 = nn.Linear(512, actions_count)
        self.fcc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1s(x))
        x = F.relu(self.conv2s(x))
        x = F.relu(self.conv3s(x))
        x = F.relu(self.conv4s(x))
        x = F.relu(self.conv5s(x))
        x = F.relu(self.conv6s(x))

        x = x.flatten(start_dim=1)

        x_a = F.relu(self.fca1(x))
        x_c = F.relu(self.fcc1(x))

        outActor = self.fca2(x_a)
        outCritic = self.fcc2(x_c)

        return outActor, outCritic
