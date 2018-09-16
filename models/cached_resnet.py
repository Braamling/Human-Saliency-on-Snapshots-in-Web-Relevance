import torch.nn as nn

class ResNetCached(nn.Module):

    def __init__(self, expansion_size, output_size=1000):
        super(ResNetCached, self).__init__()
        self.fc1 = nn.Linear(512 * expansion_size, 512 * expansion_size)
        self.fc2 = nn.Linear(512 * expansion_size, 512 * expansion_size)
        self.fc3 = nn.Linear(512 * expansion_size, output_size)
        self.feature_size = output_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
