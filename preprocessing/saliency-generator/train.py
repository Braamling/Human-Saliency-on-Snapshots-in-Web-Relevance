from models.vgg16 import VGG16_tranfer_stage_one, VGG16_stage_one_to_stage_two
from saliencyDataIterator import SaliencyDataIterator
from keras.callbacks import ModelCheckpoint

import argparse

def train():
    # If weights are specific, load from weights, otherwise load from imagenet.
    if FLAGS.s1_from_weights is not None: 
        model = VGG16_tranfer_stage_one(FLAGS.s1_from_weights)
    else:
        model = VGG16_tranfer_stage_one()


    # If s2 weights are set, load these weight directly.
    if FLAGS.s2_from_weights is not None: 
        model = VGG16_tranfer_stage_one(FLAGS.s2_from_weights)
        model = VGG16_stage_one_to_stage_two(model)
    else:
        train_generator = SaliencyDataIterator(FLAGS.s1_train_image_path, FLAGS.s1_train_heatmap_path)
        val_generator = SaliencyDataIterator(FLAGS.s1_val_image_path, FLAGS.s1_val_heatmap_path)

        checkpointer = ModelCheckpoint(filepath=FLAGS.s1_checkpoint, verbose=1, save_best_only=True)
        model.fit_generator(train_generator,
                            steps_per_epoch=train_generator.samples // FLAGS.s1_batch_size,
                            epochs=FLAGS.s1_epochs,
                            validation_data=val_generator,
                            validation_steps=val_generator.samples // FLAGS.s1_batch_size, 
                            callbacks=[checkpointer])

        # Save the intermediate weights of stage one and convert the model to stage two.
        model.save_weights(FLAGS.s1_weights_path)

        model = VGG16_stage_one_to_stage_two(model)

    # train_salicon = SALICON(FLAGS.s1_train_heatmap_path)
    train_generator = SaliencyDataIterator(FLAGS.s2_train_image_path, FLAGS.s2_train_heatmap_path)
    val_generator = SaliencyDataIterator(FLAGS.s2_val_image_path, FLAGS.s2_val_heatmap_path)


    checkpointer = ModelCheckpoint(filepath=FLAGS.s2_checkpoint, verbose=1, save_best_only=True)
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // FLAGS.s2_batch_size,
                        epochs=FLAGS.s2_epochs,
                        validation_data=val_generator,
                        validation_steps=train_generator.samples // FLAGS.s2_batch_size,
                        callbacks=[checkpointer])

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
    parser.add_argument('--s1_checkpoint', type=str, default='storage/weights/s1_weights_checkpoint.h5',
                        help='The location to store the stage one model intermediate checkpoint weights.')
    parser.add_argument('--s1_batch_size', type=int, default=32,
                        help='The batch size used for training.')
    parser.add_argument('--s1_epochs', type=int, default=10,
                        help='The amount of epochs used to train.')
    parser.add_argument('--s1_from_weights', type=str, default=None,
                        help='The model to start stage 1 from, if None it will start from scratch (or skip if only stage two is configured).')

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
