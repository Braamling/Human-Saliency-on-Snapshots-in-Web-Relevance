import math

import torch.nn as nn

class TransformCache(nn.Module):

    def __init__(self, input_size=25088, hidden_layers='4096x4096', output_size=1000, init_weights=True):
        super(TransformCache, self).__init__()

        hidden_sizes = [input_size] + [int(x) for x in hidden_layers.split('x')] + [output_size]
        n_hidden_units = len(hidden_sizes)

        layers = []
        for i in range(n_hidden_units - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

            if i < n_hidden_units - 2:
                layers += [nn.ReLU(True), nn.Dropout()] # TODO should this dropout be configurable?

        self.classifier = nn.Sequential(*layers)

        self.feature_size = output_size
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
