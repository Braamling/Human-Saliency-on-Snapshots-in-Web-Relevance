import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

class LTR_score(nn.Module):
    def __init__(self, static_feature_size, feature_model=None):
        super(LTR_score, self).__init__()

        self.feature_model = feature_model
        if feature_model is None:
            x_in = feature_model.feature_size
        else:
            x_in = feature_model.feature_size + static_feature_size


        self.model = nn.Sequential(
            nn.Linear(x_in, 10),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(10, 1),
        )

    def forward(self, x, features):
        if self.feature_model is not None:
            x = self.feature_model(x)
            features = torch.cat((x, features), 1)
        output = self.model(features)

        return output

def pair_hinge_loss(positive, negative):
    # TODO add L2 regularization
    loss = torch.clamp(1.0 - positive + negative, 0.0)

    return loss.mean()

N = 10
N_features = 10
input_size = 100

n_static_features = torch.rand(N, N_features)
n_nn_input = torch.rand(N, input_size)
n_scores = torch.round(6 * torch.rand(N, 1) - 2)

p_static_features = torch.rand(N, N_features)
p_nn_input = torch.rand(N, input_size)
p_scores = n_scores + 1

feature_nn = LTR_features(input_size, 28)
pred_scorer = LTR_score(N_features, feature_nn)

use_gpu = torch.cuda.is_available()

if use_gpu:
    pred_scorer = pred_scorer.cuda()
    n_static_features, p_static_features = Variable(n_static_features.cuda()), Variable(p_static_features.cuda())
    n_nn_input, p_nn_input = Variable(n_nn_input.cuda()), Variable(p_nn_input.cuda())
    n_scores, p_scores = Variable(n_scores.cuda()), Variable(p_scores.cuda())

for i in range(1500):
    # Get positive and negative predictions
    positive = pred_scorer.forward(n_nn_input, n_static_features)
    negative = pred_scorer.forward(p_nn_input, p_static_features)

    # Prepare optimizer
    opt_parameters = pred_scorer.model.parameters()
    optimizer = optim.Adam(opt_parameters, lr=0.0001, weight_decay=1e-5)

    loss = pair_hinge_loss(positive, negative)
    # print('{} Loss: {}'.format(i, loss.data[0]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print(positive)
print(positive)
print(negative)
# print(feature_nn.forward(nn_input))

