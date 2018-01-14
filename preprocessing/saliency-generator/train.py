from vgg16 import VGG16_tranfer_stage_one
from saliencyDataIterator import SaliencyDataIterator
from salicon.salicon import SALICON

import argparse

def train():
	model = VGG16_tranfer_stage_one()

	salicon=SALICON(FLAGS.salicon_annotation_path)

	iterator = SaliencyDataIterator(FLAGS.salicon_image_path, salicon)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--salicon_annotation_path', type=str, default='lib/salicon/annotations/fixations_val2014.json',
                        help='The location of the salicon annotatation data for training.')
    parser.add_argument('--salicon_image_path', type=str, default='lib/salicon/images/val2014/',
                        help='The location of the salicon images for trainig.')

    FLAGS, unparsed = parser.parse_known_args()
    
    train()
