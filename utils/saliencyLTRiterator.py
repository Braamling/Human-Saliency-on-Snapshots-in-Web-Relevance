from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader 
from torchvision import transforms
from .customExceptions import NoImageAvailableException, NoRelDocumentsException

from preprocessing.contextualFeaturesGenerator.utils.featureStorage import FeatureStorage

import os
import numpy as np
import random
import logging 

logger = logging.getLogger('Dataset')

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
    def __init__(self, image_dir, features_file, get_images=False, query_specific=False, only_with_image=False, size=(64,64), grayscale=True):
        """
        Args:
            img_dir (string): directory containing all images for the ClueWeb12 webpages
            features_file (string): a file containing the features scores for each query document pair.
            scores_file (string): a file containing the scores for each query document pair.
        """
        self.get_images = get_images
        self.query_specific = query_specific
        self.only_with_image = only_with_image
        self.image_dir = image_dir

        self.make_dataset(image_dir, features_file)
        if grayscale:
            self.img_transform = transforms.Compose([transforms.Resize(size, interpolation=2), 
                                                     transforms.Grayscale(),
                                                     transforms.ToTensor()])
        else:
            self.img_transform = transforms.Compose([transforms.Resize(size, interpolation=2), 
                                                     transforms.ToTensor()])

    def make_dataset(self, image_dir, features_file):
        featureStorage = FeatureStorage(features_file, image_dir, self.query_specific, self.only_with_image)
        dataset = []
        train_dataset = []

        # External doc id to internal doc id
        train_ext2int = {}
        self.ext2int = {}
        self.idx2posneg = {}
        i = 0
        # Create a dataset with all query-document pairs
        for q_id, score, d_id, vec, image in featureStorage.get_all_entries():
            # Make the query-score index.
            qs_idx = "{}:{}".format(q_id, score)

            # Create an entry for a query-document apir
            query_doc_idx = "{}:{}".format(q_id, d_id)

            # Add an entry with all documents that have a different score
            if qs_idx not in self.idx2posneg:
                posnegs = self._get_alt_scores_docs(featureStorage, q_id, score)
                self.idx2posneg[qs_idx] = posnegs

            # Check whether any documents were found with a different relevant score 
            # and if we filter all documents without an images whether the images exists.
            if len(self.idx2posneg[qs_idx]) > 0:
                self.ext2int[query_doc_idx] = i
                i += 1
                # Create the dataset entry
                item = (image, q_id, score, d_id, vec)
                dataset.append(item)

        logger.info("Added a dataset with {} queries with {} documents".format(len(featureStorage.get_queries()), i))
        # Convert all external ids in idx2posneg to internal ids.
        for qs_idx in self.idx2posneg.keys():
            self.idx2posneg[qs_idx] = [self.ext2int[i] for i in self.idx2posneg[qs_idx] if len(self.idx2posneg[qs_idx]) > 0]

        self.dataset = dataset


    """
    Get all documents within the same query with a different score.

    featureStorage: FeatureStorage, a loaded feature storage to retrieve query-docs pairs from
    q_id: int, the query id to search documents in
    score: int, the score of the current document, only documents with other scores are added. 
    """
    def _get_alt_scores_docs(self, featureStorage, query_id, doc_score):
        ids = []
        for score, doc_id in featureStorage.get_documents_in_query(query_id):
            if doc_score != score:
                ids.append("{}:{}".format(query_id, doc_id))

        return ids


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, q_id, score, _, _ = self.dataset[idx]

        # Get the id for the query-score pair
        qs_idx = "{}:{}".format(q_id, score)

        posneg_idx = random.randint(1, len(self.idx2posneg[qs_idx]))
        posneg_idx = self.idx2posneg[qs_idx][posneg_idx-1]

        # print(self.dataset[idx][2], self.dataset[posneg_idx][2])
        if self.dataset[idx][2] > self.dataset[posneg_idx][2]:
            p_image, _, p_score, _, p_vec = self.dataset[idx]
            n_image, _, n_score, _, n_vec = self.dataset[posneg_idx]
        else:
            p_image, _, p_score, _, p_vec = self.dataset[posneg_idx]
            n_image, _, n_score, _, n_vec = self.dataset[idx]

        # Load the postive and negative input image
        # p_vecimage = self.img_transform(default_loader(p_image))
        # n_image = self.img_transform(default_loader(n_image))
        if self.get_images:
            p_image = self.img_transform(default_loader(p_image))
            n_image = self.img_transform(default_loader(n_image))
        

        # The model will filter out the vector, but an empty vector is not supported by pytorch.
        if len(n_vec) is 0:
            p_vec = -1
            n_vec = -1
        
        positive_sample = (p_image, p_vec, p_score)
        negative_sample = (n_image, n_vec, n_score)

        return positive_sample, negative_sample

    """
    Get a specific Clueweb document
    """
    def get_document(self, doc_id, query_id):
        query_doc_idx = "{}:{}".format(query_id, doc_id)

        if query_doc_idx not in self.ext2int:
            print(query_doc_idx)
            raise NoRelDocumentsException("document not in index, probably no relevant documents were found")

        image, _, score, _, vec = self.dataset[self.ext2int[query_doc_idx]]

        # The model will filter out the vector, but an empty vector is not supported by pytorch.
        if len(vec) is 0:
            vec = -1

        if self.get_images:
            image = self.img_transform(default_loader(image))
        return (image, vec, score)
