from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Dense, Reshape
from keras import backend as K 

from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np

def VGG16_tranfer_stage_one():
    # Load the VGG16 pretrained model on imagenet.
    base_model = VGG16(weights='imagenet')

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
    fca = Dense(4096, activation='relu',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1))(fc7)
    saliency = Reshape((64, 64))(fca)

    # Rebuild the new model.
    model = Model(inputs=base_model.input, outputs=saliency)
    
    model.compile(optimizer='sgd', loss=euc_dist_keras)

    return model
    
def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

if __name__ == "__main__":
    model = VGG16_tranfer_stage_one()

    model.compile(optimizer='sgd', loss=euc_dist_keras)



    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)

    preds = model.predict(x)
    print(preds.shape)
