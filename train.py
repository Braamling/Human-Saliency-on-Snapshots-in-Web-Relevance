import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import argparse
from models.models import LTR_features, LTR_score, ViP_features

from utils.saliencyLTRiterator import ClueWeb12Dataset
from utils.evaluate import Evaluate

import tensorboard_logger as tf_logger
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger("train")

def pair_hinge_loss(positive, negative):
    loss = torch.clamp(1.0 - positive + negative, 0.0)
    return loss.mean()

"""
This method prepares the dataloaders for training and returns a training/validation dataloader.
"""
def prepare_dataloaders():
    # Get the train/test datasets
    train_dataset = ClueWeb12Dataset(FLAGS.image_path, FLAGS.train_file, FLAGS.images)
    test_dataset = ClueWeb12Dataset(FLAGS.image_path, FLAGS.test_file, FLAGS.images)
    
    # Prepare the loaders
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                  shuffle=True, num_workers=4)
    # Initiate the Evaluation classes
    trainEval = Evaluate(FLAGS.train_file, train_dataset, FLAGS.images, "train")
    testEval = Evaluate(FLAGS.test_file, test_dataset, FLAGS.images, "test")

    logging.info("Training on: ")

    return dataloader, trainEval, testEval

def train_model(model, criterion, dataloaders, use_gpu, optimizer, scheduler, num_epochs=25):
    dataloader, trainEval, testEval = dataloaders


    tf_logger.configure(FLAGS.log_dir.format(FLAGS.description))
    

    # Set model to training mode
    # model.model.train(True)  # Set model to training mode
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train(False)  
        trainEval.eval(model, tf_logger, epoch)
        testEval.eval(model, tf_logger, epoch)
        model.train(True)  
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

            if not FLAGS.images:
                data[0][0] = data[1][0] = None

            positive = model.forward(data[0][0], p_static_features)
            negative = model.forward(data[1][0], n_static_features)

            # Compute the loss
            loss = criterion(positive, negative)

            running_loss += loss.data[0] * p_static_features.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # testEval.eval(model, tf_logger, epoch)

            # Print the loss

            # for param in model.parameters():
            #   print(param.data)
        tf_logger.log_value('train_loss', running_loss, epoch)
        print('Train_loss: {}'.format(running_loss))

    return model

"""
Prepare the model with the correct weights and format the the configured use.
"""
def prepare_model(use_scheduler=True):
    use_gpu = torch.cuda.is_available()

    if FLAGS.model is "ViP":
        model = LTR_score(FLAGS.content_feature_size, ViP_features(4, 10, FLAGS.batch_size))
    elif FLAGS.model is "features_only":
        model = LTR_score(FLAGS.content_feature_size)
    else:
        raise NotImplementedError("Model: {} is not implemented".format(FLAGS.model))

    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-5)

    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, optimizer, scheduler, use_gpu


def train():
    model, optimizer, scheduler, use_gpu = prepare_model()

    logger.info(model)

    dataloaders = prepare_dataloaders()

    train_model(model, pair_hinge_loss, dataloaders, use_gpu, optimizer, scheduler, num_epochs=FLAGS.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str, default='preprocessing/contextualFeaturesGenerator/storage/clueweb12_web_trec/Fold1/train.txt',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--test_file', type=str, default='preprocessing/contextualFeaturesGenerator/storage/clueweb12_web_trec/Fold1/test.txt',
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
    parser.add_argument('--content_feature_size', type=int, default=11,
                        help='The amount of context features')
    parser.add_argument('--model', type=str, default="features_only",
                        help='chose the model to train, (features_only, ViP)')
    parser.add_argument('--images', type=str, default="False",
                        help='set whether the images should be included for training.')
    parser.add_argument('--log_dir', type=str, default='storage/logs/{}',
                        help='The location to place the tensorboard logs.')

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.images = FLAGS.images is "True"

    logger.info(FLAGS)

    train()

