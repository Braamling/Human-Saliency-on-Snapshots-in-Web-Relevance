import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import argparse
import torch.nn.functional as F
from models.models import LTR_features, ViP_features

from utils.saliencyLTRiterator import ClueWeb12Dataset
from utils.evaluate import Evaluate

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
        self.predict = torch.nn.Linear(10, 1) 

    def forward(self, image, static_features):
        if self.feature_model is not None:
            image = self.feature_model(image)
            features = torch.cat((image, static_features), 1)
        else:
            features = static_features
        # output = self.model(features)
        x = F.relu(self.hidden(features))
        x = self.predict(x)  

        return x

def pair_hinge_loss(positive, negative):
    # TODO add L2 regularization
    loss = torch.clamp(1.0 - positive + negative, 0.0)

    return loss.mean()

"""
This method prepares the dataloaders for training and returns a training/validation dataloader.
"""
def prepare_dataloaders():
    # Get the train/test datasets
    train_dataset = ClueWeb12Dataset(FLAGS.image_path, FLAGS.train_file)
    test_dataset = ClueWeb12Dataset(FLAGS.image_path, FLAGS.test_file)
    
    # Prepare the loaders
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                  shuffle=True, num_workers=4)
    # Initiate the Evaluation classes
    trainEval = Evaluate(FLAGS.train_file, train_dataset)
    testEval = Evaluate(FLAGS.test_file, test_dataset)

    return dataloader, trainEval, testEval

def train_model(model, criterion, dataloaders, use_gpu, optimizer, scheduler, num_epochs=25):
    dataloader, trainEval, testEval = dataloaders
    model.train(True)  # Set model to training mode
    # model.model.train(True)  # Set model to training mode
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        trainEval.eval(model)

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

            # image = Variable(data[0][0].float())
            # feature_nn = ViP_features(4)
            # feature_nn.forward(image)
            # break
            # print(p_static_features, data[0][2])
            # print(n_static_features, data[1][2])
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

    model = LTR_score(3)

    if use_gpu:
        model = model.cuda()

    opt_parameters = model.parameters()

    optimizer = optim.Adam(opt_parameters, lr=FLAGS.learning_rate, weight_decay=1e-5)
    # optimizer = optim.SGD(opt_parameters, lr=FLAGS.learning_rate, weight_decay=1e-5)

    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, optimizer, scheduler, use_gpu


def train():
    dataloaders = prepare_dataloaders()
    model, optimizer, scheduler, use_gpu = prepare_model()

    train_model(model, pair_hinge_loss, dataloaders, use_gpu, optimizer, scheduler, num_epochs=FLAGS.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str, default='preprocessing/contextualFeaturesGenerator/storage/hdf5/test_split_train.hdf5',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--test_file', type=str, default='preprocessing/contextualFeaturesGenerator/storage/hdf5/test_split_test.hdf5',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--image_path', type=str, default='storage/salicon/images/',
                        help='The location of the salicon images for training.')

    parser.add_argument('--batch_size', type=int, default=3,
                        help='The batch size used for training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs used to train.')
    parser.add_argument('--description', type=str, default='example_run',
                        help='The description of the run, for logging, output and weights naming.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate to use for the experiment')

    FLAGS, unparsed = parser.parse_known_args()

    train()

