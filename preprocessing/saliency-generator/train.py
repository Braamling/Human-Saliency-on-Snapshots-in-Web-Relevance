from models.vgg16 import VGG16_tranfer_stage_one, VGG16_stage_one_to_stage_two
from saliencyDataIterator import SaliencyDataIterator

import argparse

def train():
    model = VGG16_tranfer_stage_one()

    train_generator = SaliencyDataIterator(FLAGS.s1_train_image_path, FLAGS.s1_train_heatmap_path)
    val_generator = SaliencyDataIterator(FLAGS.s1_val_image_path, FLAGS.s1_val_heatmap_path)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // FLAGS.s1_batch_size,
                        epochs=FLAGS.s1_epochs,
                        validation_data=val_generator,
                        validation_steps=val_generator.samples // FLAGS.s1_batch_size)

    # Save the intermediate weights of stage one and convert the model to stage two.
    model.save_weights(FLAGS.s1_weights_path)
    model = VGG16_stage_one_to_stage_two(model)

    # train_salicon = SALICON(FLAGS.s1_train_heatmap_path)
    train_generator = SaliencyDataIterator(FLAGS.s2_train_image_path, FLAGS.s2_train_heatmap_path)
    val_generator = SaliencyDataIterator(FLAGS.s2_val_image_path, FLAGS.s2_val_heatmap_path)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // FLAGS.s2_batch_size,
                        epochs=FLAGS.s2_epochs,
                        validation_data=val_generator,
                        validation_steps=val_generator.samples // FLAGS.s2_batch_size)

    # Save the final weights
    model.save_weights(FLAGS.s2_weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Stage one arguments
    # TODO change train path back to training dir
    parser.add_argument('--s1_train_heatmap_path', type=str, default='storage/salicon/heatmaps/train/',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--s1_train_image_path', type=str, default='storage/salicon/images/train/',
                        help='The location of the salicon images for training.')

    parser.add_argument('--s1_val_heatmap_path', type=str, default='storage/salicon/heatmaps/val/',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--s1_val_image_path', type=str, default='storage/salicon/images/val/',
                        help='The location of the salicon images for training.')

    parser.add_argument('--s1_weights_path', type=str, default='storage/weights/s1_weights.h5',
                        help='The location to store the stage one model weights.')
    parser.add_argument('--s1_batch_size', type=int, default=32,
                        help='The batch size used for training.')
    parser.add_argument('--s1_epochs', type=int, default=10,
                        help='The amount of epochs used to train.')

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
    parser.add_argument('--s2_batch_size', type=int, default=32,
                        help='The batch size used for training.')
    parser.add_argument('--s2_epochs', type=int, default=10,
                        help='The amount of epochs used to train.')

    # Stage two arguments
    # TODO create stage two arguments.
    FLAGS, unparsed = parser.parse_known_args()
    
    train()
