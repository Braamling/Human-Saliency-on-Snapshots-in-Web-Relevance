import torch
import torch.nn as nn

class LTR_features(nn.Module):
    def __init__(self, input_size, feature_size):
        super(LTR_features, self).__init__()
        self.feature_size = feature_size
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, feature_size),
        )

    def forward(self, x):
        features = self.model(x)
        return features

class ViP_features(nn.Module):
    def __init__(self, region_height):
        super(ViP_features, self).__init__()
        self.region_height = region_height
        self.local_perception_layer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        height = x.size()[2]
        print(x.size())
        splits = int(height/self.region_height)
        x = torch.split(x, splits, 2)
        print(x[0].size())

        apply_conv = lambda layer: self.local_perception_layer(layer)
        # TODO apply this to all in x.
        features = apply_conv(x)
        print(features.size())
        return features