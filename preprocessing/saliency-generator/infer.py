from models.vgg16 import VGG16_tranfer_stage_one
# from saliencyDataIterator import SaliencyDataIterator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image


import argparse
np.set_printoptions(threshold=np.nan)

def infer():
    model = VGG16_tranfer_stage_one(FLAGS.weights_path)

    img = image.load_img(FLAGS.image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    
    print(features)

    features = features[0] / features[0].max()
    im = Image.fromarray(np.uint8(features*255))
    im.save(FLAGS.target_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='train_img.jpg',
                        help='The location to store the stage one model weights.')
    parser.add_argument('--target_path', type=str, default='test.png',
                        help='The location to store the stage one model weights.')
    parser.add_argument('--weights_path', type=str, default='storage/weights/s1_weights.h5',
                        help='The location to store the stage one model weights.')

    FLAGS, unparsed = parser.parse_known_args()
    
    infer()
