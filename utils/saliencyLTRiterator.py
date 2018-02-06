from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader 
from torchvision import transforms

import os
import numpy as np


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# TODO duplicate from saliencyLTRiterator, seperate
def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class ClueWeb12Dataset(Dataset):
    def __init__(self, image_dir, features_file, scores_file):
        """
        Args:
            img_dir (string): directory containing all images for the ClueWeb12 webpages
            features_file (string): a file containing the features scores for each query document pair.
            scores_file (string): a file containing the scores for each query document pair.
        """
        self.make_dataset(image_dir, features_file, scores_file)
        self.img_transform = transforms.Compose([transforms.Resize((224,224), interpolation=2), 
                                                 transforms.ToTensor()])

    def make_dataset(self, image_dir, features_file, scores_file):
        dataset = []

        # TODO get a positive and negative sample together with the image.
        for fname in os.listdir(image_dir):
            image = os.path.join(image_dir, fname)
            heatmap = os.path.join(heatmap_dir, fname)
            if is_image_file(image) and is_image_file(heatmap):
                item = (image, heatmap)
                dataset.append(item)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.img_transform(default_loader(self.dataset[idx][0]))

        negative_sample = (n_image, n_features, n_score)
        positive_sample = (p_image, p_features, p_score)

        return negative_sample, positive_sample