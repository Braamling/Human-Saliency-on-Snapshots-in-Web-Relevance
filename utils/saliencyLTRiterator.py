from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader 
from torchvision import transforms

from preprocessing.contextualFeaturesGenerator.utils.featureStorage import FeatureStorage

import os
import numpy as np
import random


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
    def __init__(self, image_dir, features_file):
        """
        Args:
            img_dir (string): directory containing all images for the ClueWeb12 webpages
            features_file (string): a file containing the features scores for each query document pair.
            scores_file (string): a file containing the scores for each query document pair.
        """
        self.make_dataset(image_dir, features_file)
        self.img_transform = transforms.Compose([transforms.Resize((224,224), interpolation=2), 
                                                 transforms.ToTensor()])

    def make_dataset(self, image_dir, features_file):
        featureStorage = FeatureStorage(features_file)
        dataset = []

        # External doc id to internal doc id
        self.ext2int = {}
        self.idx2posneg = {}

        # Create a dataset with all query-document pairs
        for i, (q_id, score, d_id, vec) in enumerate(featureStorage.get_pairs()):
            # Calculate the query-score index.
            qs_idx = "{}:{}".format(q_id, score)

            if qs_idx not in self.idx2posneg:
                posnegs = self._get_alt_scores_docs(featureStorage, q_id, score)

                self.idx2posneg[qs_idx] = posnegs

            # Only add query-document pairs with available negative or positive samples.
            if len(self.idx2posneg[qs_idx]) != 0:
                self.ext2int[d_id] = i

                image = os.path.join(image_dir, d_id)
                item = (image, q_id, score, d_id, vec)
                dataset.append(item)

        # Convert all external ids in idx2posneg to internal ids.
        for qs_idx in self.idx2posneg.keys():
            self.idx2posneg[qs_idx] = [self.ext2int[i] for i in self.idx2posneg[qs_idx]]

        self.dataset = dataset

    """
    Get all documents within the same query with a different score.

    featureStorage: FeatureStorage, a loaded feature storage to retrieve query-docs pairs from
    q_id: int, the query id to search documents in
    score: int, the score of the current document, only documents with other scores are added. 
    """
    def _get_alt_scores_docs(self, featureStorage, q_id, doc_score):
        ids = []
        for score, doc_id in featureStorage.get_documents_in_query(q_id):
            if doc_score != score:
                ids.append(doc_id)

        return ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, q_id, score, _, _ = self.dataset[idx]

        # Get the id for the query-score pair
        qs_idx = "{}:{}".format(q_id, score)

        # Sample a second document with a different score for the same query.
        posneg_idx = random.randint(0, len(self.idx2posneg[qs_idx]))

        if self.dataset[idx][2] > self.dataset[posneg_idx][2]:
            p_image, _, p_score, _, p_vec = self.dataset[idx]
            n_image, _, n_score, _, n_vec = self.dataset[posneg_idx]
        else:
            p_image, _, p_score, _, p_vec = self.dataset[posneg_idx]
            n_image, _, n_score, _, n_vec = self.dataset[idx]

        # Load the postive and negative input image
        # p_image = self.img_transform(default_loader(p_image))
        # n_image = self.img_transform(default_loader(n_image))

        positive_sample = (p_image, p_vec, p_score)
        negative_sample = (n_image, n_vec, n_score)
        return positive_sample, negative_sample