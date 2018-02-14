import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import argparse

from utils.saliencyLTRiterator import ClueWeb12Dataset

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
            x_in = static_feature_size
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

"""
This method prepares the dataloaders for training and returns a training/validation dataloader.
"""
def prepare_dataloaders(image_path, features_dir, batch_size):
    # Get the train/val datasets
    dataset = ClueWeb12Dataset(image_path, features_dir)
    
    # Prepare the loaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
    # Get the datasizes for logging purposes.
    dataset_sizes = len(dataset)

    return dataloader

def train_model(model, criterion, dataloader, use_gpu, optimizer, scheduler, num_epochs=25):

    model.train(True)  # Set model to training mode
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        
        if scheduler is not None:
            scheduler.step()
            

        running_loss = 0.0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and wrap them into varaibles
            if use_gpu:
                p_static_features = Variable(data[0][1].float().cuda())
                n_static_features = Variable(data[1][1].float().cuda())
            else:
                p_static_features = Variable(data[0][1].float())
                n_static_features = Variable(data[1][1].float())

            # Do the forward prop.
            positive = model.forward(None, p_static_features)
            negative = model.forward(None, n_static_features)
            
            # Compute the loss
            loss = criterion(positive, negative)

            running_loss += loss.data[0] * p_static_features.size(0)

            # print('{} Loss: {}'.format(i, loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss
        print('Loss: {}'.format(running_loss))

    return model

"""
Prepare the model with the correct weights and format the the configured use.
"""
def prepare_model(use_scheduler=True):
    use_gpu = torch.cuda.is_available()

    # feature_nn = LTR_features(input_size, 28)
    model = LTR_score(3)

    if use_gpu:
        model = model.cuda()

    opt_parameters = model.model.parameters()
    optimizer = optim.Adam(opt_parameters, lr=FLAGS.learning_rate, weight_decay=1e-5)

    if use_gpu:
        model = model.cuda()

    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, optimizer, scheduler, use_gpu


def train():
    dataloader = prepare_dataloaders(FLAGS.image_path, FLAGS.features_file, FLAGS.batch_size)
    model, optimizer, scheduler, use_gpu = prepare_model()
    train_model(model, pair_hinge_loss, dataloader, use_gpu, optimizer, scheduler, num_epochs=FLAGS.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--features_file', type=str, default='preprocessing/contextualFeaturesGenerator/storage/hdf5/context_features.hdf5',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--image_path', type=str, default='storage/salicon/images/',
                        help='The location of the salicon images for training.')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size used for training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs used to train.')
    parser.add_argument('--description', type=str, default='example_run',
                        help='The description of the run, for logging, output and weights naming.')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='The learning rate to use for the experiment')

    FLAGS, unparsed = parser.parse_known_args()

    train()

