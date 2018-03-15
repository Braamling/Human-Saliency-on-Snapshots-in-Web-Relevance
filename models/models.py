import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


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


"""
The LTR score model can hold a second feature network that can be fed
external features. The model is then trained end-to-end.
"""
class LTR_score(nn.Module):
    def __init__(self, static_feature_size, feature_model=None):
        super(LTR_score, self).__init__()
        self.feature_model = feature_model
        if feature_model is None:
            x_in = static_feature_size
        else:
            x_in = feature_model.feature_size + static_feature_size

        self.hidden = torch.nn.Linear(x_in, 10)   # hidden layer
        self.dropout = torch.nn.Dropout(.1)
        self.relu = torch.nn.ReLU()
        self.predict = torch.nn.Linear(10, 1) 

    def forward(self, image, static_features):
        if self.feature_model is not None:
            image = self.feature_model(image)
            if static_features.dim() == 1:
                static_features = static_features.unsqueeze(0)

            features = torch.cat((image, static_features), 1)
        else:
            features = static_features

        x = self.hidden(features)
        x = self.relu(x)
        if torch.nonzero(x).size(0) < 100:
            print("Warning, dead relu's", torch.nonzero(x).size(0))
            
        x = self.dropout(x)
        x = self.predict(x)  

        return x

class ViP_features(nn.Module):
    def __init__(self, region_height, feature_size, batch_size):
        super(ViP_features, self).__init__()
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.region_height = region_height
        self.local_perception_layer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.hidden_dim = 50 
        self.lstm = nn.LSTM(208, self.hidden_dim)
        self.reldecision = nn.Linear(self.hidden_dim, self.feature_size)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_gpu:
            return (Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def apply_lstm(self, x):
        for i, layer in enumerate(x):
            # Create correct dimensions
            if layer.dim() == 3:
                layer = layer.unsqueeze(0)

            if self.use_gpu:
                layer = Variable(layer.cuda())
            else:
                layer = Variable(layer)
            
            if i is 0:
                batch_size = layer.size()[0]
                hidden = self.init_hidden(batch_size)

            layer = self.local_perception_layer(layer)
            out, hidden = self.lstm(layer.view(1, -1, 208), hidden)

        return out.squeeze(0)

    def forward(self, x):
        height = x.size()[2]
        splits = int(height/self.region_height)
        x = torch.split(x, splits, 2)

        lstm_out = self.apply_lstm(x)
        tag_space = self.reldecision(lstm_out)
        return tag_space
