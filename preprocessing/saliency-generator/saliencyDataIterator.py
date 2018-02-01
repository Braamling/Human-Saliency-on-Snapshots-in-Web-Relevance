from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from keras.preprocessing.image import Iterator, load_img, img_to_array, _count_valid_files_in_directory

from keras import backend as K

from PIL import Image as pil_image

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

def _reshape_image(img, target_size, interpolation='nearest'):
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
        if interpolation not in _PIL_INTERPOLATION_METHODS:
            raise ValueError(
                'Invalid interpolation method {} specified. Supported '
                'methods are {}'.format(
                    interpolation,
                    ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
        resample = _PIL_INTERPOLATION_METHODS[interpolation]
        img = img.resize(width_height_tuple, resample)
    return img

class SaliencyDataIterator(Iterator):
    """Iterator capable of reading images from a directory on disk and adding 
    the correct saliency map.
    # Arguments
        directory: Path to the directory to read images from.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, img_directory, saliency_directory=None,
                 target_size=(224, 224), 
                 saliency_size=(64, 64), batch_size=32, 
                 shuffle=True, interpolation='nearest', 
                 seed=None, data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 color_mode='rgb'):
        if data_format is None:
            data_format = K.image_data_format()
        self.img_directory = img_directory
        self.saliency_directory = saliency_directory        
        self.target_size = tuple(target_size)
        self.saliency_size = tuple(saliency_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.batch_size = batch_size

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = os.listdir(self.img_directory)
        self.samples = len(self.filenames)

        print('Found %d samples.' % (self.samples))

        super(SaliencyDataIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def filename_to_id(self, filename):
        # Remove file extension, everything before number and convert to int to remove leading 0's.
        return int(filename.split('.')[0].split("_")[-1])

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array),) + self.saliency_size, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            # Load the input image
            img = load_img(os.path.join(self.img_directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            batch_x[i] = x

            # Load the saliency image
            img = load_img(os.path.join(self.saliency_directory, fname),
                           grayscale=True,
                           target_size=self.saliency_size,
                           interpolation=self.interpolation)

            img_array = np.asarray(img, dtype=K.floatx())
            batch_y[i] = img_array

        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
