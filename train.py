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

import tensorboard_logger as tfl
import logging
import copy
import os

FORMAT = '%(name)s: [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger("train")

def pair_hinge_loss(positive, negative):
    loss = torch.clamp(1.0 - positive + negative, 0.0)
    return loss.mean()

"""
This method prepares the dataloaders for training and returns a training/validation dataloader.
"""
def prepare_dataloaders(train_file, test_file, vali_file):
    # Get the train/test datasets
    train_dataset = ClueWeb12Dataset(FLAGS.image_path, train_file, FLAGS.load_images,
                                     FLAGS.query_specific, FLAGS.only_with_image)
    test_dataset = ClueWeb12Dataset(FLAGS.image_path, test_file, FLAGS.load_images,
                                    FLAGS.query_specific, FLAGS.only_with_image)
    vali_dataset = ClueWeb12Dataset(FLAGS.image_path, vali_file, FLAGS.load_images,
                                    FLAGS.query_specific, FLAGS.only_with_image)

    # Prepare the loaders
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                  shuffle=True, num_workers=10)
    # Initiate the Evaluation classes
    trainEval = Evaluate(train_file, train_dataset, FLAGS.load_images, "train")
    testEval = Evaluate(test_file, test_dataset, FLAGS.load_images, "test")
    valiEval = Evaluate(test_file, test_dataset, FLAGS.load_images, "validation")

    return dataloader, trainEval, testEval, valiEval


"""
The fold iterator provides files to train, test and validate on for all folds and sessions.
To generator yields a test, train and validate file path that should be used for the current model.
"""
def fold_iterator():
    for fold in range(1, FLAGS.folds+1):
        for session in range(FLAGS.sessions_per_fold):
            fold_path = os.path.join(FLAGS.content_feature_dir, "Fold{}".format(fold))
            test = os.path.join(fold_path, "test.txt")
            train = os.path.join(fold_path, "train.txt")
            vali = os.path.join(fold_path, "vali.txt")
            yield test, train, vali



def train_model(model, criterion, dataloader, trainEval, testEval,
                use_gpu, optimizer, scheduler, description, num_epochs=25):
    
    tf_logger = tfl.Logger(FLAGS.log_dir.format(description))

    # Set model to training mode
    model.train(False)  
    best_model = copy.deepcopy(model)
    train_scores = trainEval.eval(model, tf_logger, 0)
    test_scores = testEval.eval(model, tf_logger, 0)
    best_test_score = test_scores

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

         
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

            if not FLAGS.load_images:
                data[0][0] = data[1][0] = None

            positive = model.forward(data[0][0], p_static_features)
            negative = model.forward(data[1][0], n_static_features)

            # Compute the loss
            loss = criterion(positive, negative)

            running_loss += loss.data[0] # * p_static_features.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.train(False) 
        train_scores = trainEval.eval(model, tf_logger, epoch)
        test_scores = testEval.eval(model, tf_logger, epoch)
        if best_test_score[FLAGS.optimize_on] < test_scores[FLAGS.optimize_on]:
            logger.debug("Improved the current best score.")
            best_test_score = test_scores
            best_model = copy.deepcopy(model)

        tf_logger.log_value('train_loss', running_loss, epoch)
        logger.info('Train_loss: {}'.format(running_loss))

    return best_test_score, model

"""
Prepare the model with the correct weights and format the the configured use.
"""
def prepare_model(use_scheduler=True):
    use_gpu = torch.cuda.is_available()

    if FLAGS.model == "ViP":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.dropout, FLAGS.hidden_size, ViP_features(16, 10, FLAGS.batch_size))
    elif FLAGS.model == "features_only":
        model = LTR_score(FLAGS.content_feature_size, FLAGS.dropout, FLAGS.hidden_size)
    else:
        raise NotImplementedError("Model: {} is not implemented".format(FLAGS.model))

    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-5)

    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, optimizer, scheduler, use_gpu


def train():
    test_scores = {}
    vali_scores = {}
    for i, (test, train, vali) in enumerate(fold_iterator(), 1):
        # Prepare all model components and initalize parameters.
        model, optimizer, scheduler, use_gpu = prepare_model()

        # Create a dataloader for training and three evaluation classes.
        dataloader, trainEval, testEval, valiEval = prepare_dataloaders(test, train, vali)

        if i == 1:
            logger.info(model)

        description = FLAGS.description + "_" + str(i)
        test_score, model = train_model(model, pair_hinge_loss, dataloader, trainEval, testEval,
                                        use_gpu, optimizer, scheduler, description, num_epochs=FLAGS.epochs)

        # Add and store the newly added scores.
        vali_score = valiEval.eval(model)
        test_scores = testEval.add_scores(test_scores, test_score)
        vali_scores = testEval.add_scores(vali_scores, vali_score)
        testEval.store_scores(FLAGS.optimized_scores_path + "_" + FLAGS.description, description, test_score)
        valiEval.store_scores(FLAGS.optimized_scores_path + "_" + FLAGS.description, description, vali_score)

    # Average the test and validation scores.
    test_scores = testEval.avg_scores(test_scores, i)
    vali_scores = valiEval.avg_scores(vali_scores, i)

    logger.info("Finished, printing best results now.")
    testEval.print_scores(test_scores)
    valiEval.print_scores(vali_scores)
    testEval.store_scores(FLAGS.optimized_scores_path + "_" + FLAGS.description, FLAGS.description + "_test_final", test_scores)
    valiEval.store_scores(FLAGS.optimized_scores_path + "_" + FLAGS.description, FLAGS.description + "_vali_final", vali_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_feature_dir', type=str, default='storage/clueweb12_web_trec/',
                        help='The location of all the folds with train, test and validation files.')
    parser.add_argument('--folds', type=int, default=5,
                        help='The amounts of folds to train on.')
    parser.add_argument('--sessions_per_fold', type=int, default=5,
                        help='The amount of training sessions to average per fold.')
    parser.add_argument('--image_path', type=str, default='storage/images/snapshots/',
                        help='The location of the salicon images for training.')

    parser.add_argument('--batch_size', type=int, default=3,
                        help='The batch size used for training.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='The amount of epochs used to train.')
    parser.add_argument('--description', type=str, default='example_run',
                        help='The description of the run, for logging, output and weights naming.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate to use for the experiment')
    parser.add_argument('--content_feature_size', type=int, default=11,
                        help='The amount of context features')
    parser.add_argument('--model', type=str, default="features_only",
                        help='chose the model to train, (features_only, ViP)')
    parser.add_argument('--load_images', type=str, default="True",
                        help='set whether the images should be loaded during training and evaluation.')
    parser.add_argument('--only_with_image', type=str, default="True",
                        help='set whether all documents without images should be excluded from the dataset')
    parser.add_argument('--query_specific', type=str, default="False",
                        help='set whether the images are query specific (ie. using query specific highlights)')
    parser.add_argument('--log_dir', type=str, default='storage/logs/{}',
                        help='The location to place the tensorboard logs.')
    parser.add_argument('--optimized_scores_path', type=str, default='storage/logs/optimized_scores',
                        help='The location to store the scores that were optimized.')
    parser.add_argument('--optimize_on', type=str, default='ndcg@5',
                        help='Give the measure to optimize the model on (ndcg@1, ndcg@5, ndcg@10, p@1, p@5, p@10, map).')

    parser.add_argument('--dropout', type=float, default=.1,
                        help='The dropout to use in the classification layer.')
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='The amount of hidden layers in the classification layer')

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.load_images = FLAGS.load_images == "True"
    FLAGS.only_with_image = FLAGS.only_with_image == "True"
    FLAGS.query_specific = FLAGS.query_specific == "True"

    logger.info(FLAGS)

    train()

