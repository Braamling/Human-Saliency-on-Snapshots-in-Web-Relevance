from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Dense, Reshape
from keras.optimizers import SGD
from keras import backend as K 

from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np

"""
Load a stage one model from memory. Provide the pretrained weights location to initaiize a pretrained model
or leave it empty to recontruct the model from VGG16 trained using imagenet. 
"""
def VGG16_tranfer_stage_one(weights=None):
    if weights is None:
        # Load the VGG16 pretrained model on imagenet.
        base_model = VGG16(weights='imagenet')
    else:
        base_model = VGG16()

    # Get the output the fully connected layer #7. 
    fc7 = base_model.layers[-1].output

    # Remove final fully-connected layer of imagenet.
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []

    # Fix all non-fully connected during training.
    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # Add a dense and reshape layer
    fca = Dense(4096, activation='sigmoid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1))(fc7)
    saliency = Reshape((64, 64))(fca)

    # Rebuild the new model.
    model = Model(inputs=base_model.input, outputs=saliency)

    if weights is not None:
        model.load_weights(weights)

    # model.compile(optimizer=SGD(lr=1e-3, decay=5e-4, momentum=0.9), loss=euc_dist_keras)
    model.compile(optimizer='adam', loss='mse')

    return model

"""
Convert a pretrained stage one model to a trainable stage two model.
"""
def VGG16_stage_one_to_stage_two(model):
    # Load the VGG16 pretrained model on imagenet.

    # TODO check whether this is actually just the FCA layer of the saliency layer (if that is possible :P)
    # Fix all but the model fully connected during training.
    for layer in model.layers[:-1]:
        layer.trainable = False

    model.compile(optimizer=SGD(lr=1e-3, decay=5e-4, momentum=0.9), loss=euc_dist_keras)

    return model
    
def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

if __name__ == "__main__":
    print("This file is not executable")
    exit(-1)