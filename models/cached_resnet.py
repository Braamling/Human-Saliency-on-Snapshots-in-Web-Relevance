import torch.nn as nn

class ResNetCached(nn.Module):

    def __init__(self, expansion_size, output_size=1000):
        super(ResNetCached, self).__init__()
        self.fc = nn.Linear(512 * expansion_size, output_size)
        self.feature_size = output_size

    def forward(self, x):
        x = self.fc(x)

        return x
