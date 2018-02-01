from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

from models.vgg16 import vgg16_bn, vgg16
from saliencyDataIterator import SaliencyDataset

plt.ion()   # interactive mode

import argparse
import tensorboard_logger as tfl

def save_image(data, name, grayscale=False):
    # image = data.data.cpu().numpy()[0]
    if grayscale:
        data = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = Image.fromarray(data)
    if not grayscale:
        im.mode = "RGB"
    im.save(name)


def train_model(model, criterion, dataloaders, use_gpu, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    tfl.configure(FLAGS.s1_log_dir)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients

                # forward
                outputs = model(inputs)

                # Compute and print loss
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                print('{} Loss: {:4f}'.format(phase, loss.data[0]))

            epoch_loss = running_loss / 1
            tfl.log_value('{}_loss'.format(phase), epoch_loss, epoch)

        outputs = model(inputs)
        save_image(inputs.data.cpu().numpy()[0][0], "input.png")
        save_image(outputs.data.cpu().numpy()[0], "output.png", True)
        save_image(labels.data.cpu().numpy()[0][0], "label.png", True)

        torch.save(model.state_dict(), FLAGS.s1_checkpoint)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), FLAGS.s1_weights_path)
    return model

def train():

    image_datasets = {x: SaliencyDataset(os.path.join(FLAGS.s1_image_path, x), 
                                     os.path.join(FLAGS.s1_heatmap_path, x)) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=FLAGS.s1_batch_size,
                                             shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()

    if FLAGS.s1_from_weights is not None:
        model_ft = vgg16(pretrained=False, state_dict=FLAGS.s1_from_weights)
    else:
        model_ft = vgg16(pretrained=True)

    for p in model_ft.classifier.parameters():
        p.features=False

    if use_gpu:
        model_ft = model_ft.cuda()

    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.classifier.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model_ft.classifier.parameters(), lr=0.0001, weight_decay=1e-5)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, nn.MSELoss(), dataloaders, use_gpu, optimizer, None,
                       num_epochs=FLAGS.s1_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Stage one arguments
    # TODO change train path back to training dir
    parser.add_argument('--s1_heatmap_path', type=str, default='storage/salicon/heatmaps/',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--s1_image_path', type=str, default='storage/salicon/images/',
                        help='The location of the salicon images for training.')

    parser.add_argument('--s1_weights_path', type=str, default='storage/weights/s1_weights.pth',
                        help='The location to store the stage one model weights.')
    parser.add_argument('--s1_checkpoint', type=str, default='storage/weights/s1_checkpoint.pth',
                        help='The location to store the stage one model intermediate checkpoint weights.')
    parser.add_argument('--s1_batch_size', type=int, default=32,
                        help='The batch size used for training.')
    parser.add_argument('--s1_epochs', type=int, default=10,
                        help='The amount of epochs used to train.')
    parser.add_argument('--s1_from_weights', type=str, default=None,
                        help='The model to start stage 1 from, if None it will start from scratch (or skip if only stage two is configured).')
    parser.add_argument('--s1_log_dir', type=str, default='logs/example_run',
                        help='The location to place the tensorboard logs.')

    parser.add_argument('--s2_train_heatmap_path', type=str, default='storage/FiWi/heatmaps/train/',
                        help='The location of the FiWi heatmaps data for training.')
    parser.add_argument('--s2_train_image_path', type=str, default='storage/FiWi/images/train/',
                        help='The location of the FiWi images for training.')

    parser.add_argument('--s2_val_heatmap_path', type=str, default='storage/FiWi/heatmaps/val/',
                        help='The location of the FiWi annotatation data for training.')
    parser.add_argument('--s2_val_image_path', type=str, default='storage/FiWi/images/val/',
                        help='The location of the FiWi images for training.')

    parser.add_argument('--s2_weights_path', type=str, default='storage/weights/s2_weights.h5',
                        help='The location to store the stage two model weights.')
    parser.add_argument('--s2_checkpoint', type=str, default='storage/weights/s2_weights_checkpoint.h5',
                        help='The location to store the stage two model intermediate checkpoint weights.')
    parser.add_argument('--s2_batch_size', type=int, default=29,
                        help='The batch size used for training.')
    parser.add_argument('--s2_epochs', type=int, default=10,
                        help='The amount of epochs used to train.')
    parser.add_argument('--s2_from_weights', type=str, default=None,
                        help='The model to start stage 2 from, if None it will start from scratch at stage one.')

    # Stage two arguments
    # TODO create stage two arguments.
    FLAGS, unparsed = parser.parse_known_args()

    train()
