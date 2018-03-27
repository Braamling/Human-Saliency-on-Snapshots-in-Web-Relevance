import torch
import numpy as np
import time
from preprocessing.contextualFeaturesGenerator.utils.featureStorage import FeatureStorage
from torch.autograd import Variable
from .customExceptions import NoImageAvailableException, NoRelDocumentsException


import logging 
import time

logger = logging.getLogger('Evaluate')

"""
This class can be used to easily evaluate a LTR model in various 
stages of the training process.

Evaluation measures are taken from https://gist.github.com/bwhite/3726239
"""
class Evaluate():
    def __init__(self, path, dataset, load_images, prefix):
        self.dataset = dataset
        self.storage = FeatureStorage(path, dataset.image_dir, dataset.query_specific, dataset.only_with_image)
        self.prepare_eval_data()
        self.use_gpu = torch.cuda.is_available()
        self.load_images = load_images
        self.prefix = prefix

    """
    Get all the ranked queries and their scores. 
    """
    def prepare_eval_data(self):
        self.query_ids = self.storage.get_queries()
        self.queries = {}
        for q_id in self.query_ids:
            self.queries[q_id] = self.storage.get_scores(q_id)

    def _eval_query(self, query_id, model):
        try:
            predictions = self._get_scores(query_id, model)
    
            scores = {}
            scores["ndcg@1"] = self.ndcg_at_k(predictions, 1)
            scores["ndcg@5"] = self.ndcg_at_k(predictions, 5)
            scores["ndcg@10"] = self.ndcg_at_k(predictions, 10)
            scores["p@1"] = self.precision_at_k(predictions, 1)
            scores["p@5"] = self.precision_at_k(predictions, 5)
            scores["p@10"] = self.precision_at_k(predictions, 10)
            scores["map"] = self.average_precision(predictions)

        except NoRelDocumentsException as e:
            logger.warning("query {} gave an exception: {}".format(query_id, e))
            self.failed += 1
            return {}
        except Exception as e:
            logger.error("Throwing the error from loading a documet in query {}...".format(query_id))
            raise e


        return scores

    def _get_scores(self, query_id, model):
        logger.debug('Starting to prepare {} batch to evaluate query {}'.format(self.prefix, query_id))
        start = time.time()

        predictions = []
        batch_vec = []
        batch_score = []
        images = []
        for doc, score in self.queries[query_id]:
            try:
                image, vec, rel_score = self.dataset.get_document(doc, query_id)

                batch_vec.append(vec)
                batch_score.append(score)
                images.append(image)

                if score is not rel_score:
                    logger.error(query_id, doc, score, rel_score, vec)
                    raise Exception("Somehow the relevance score in the dataset and feature storage are different.")
            except NoImageAvailableException as e:
                logger.debug("Document {} in query {} does not have an image and is excluded from evaluation. ".format(doc, query_id))
            except IOError as e:
                logger.error("Document {} in query {} gave an exception while loading, file is probabily corrupt. ".format(doc, query_id))
                raise e


        batch_vec = np.vstack( batch_vec )


        logger.debug('Batch ready, {} seconds since start'.format(time.time() - start))
        if self.use_gpu:
            batch_vec = Variable(torch.from_numpy(batch_vec).float().cuda())
        else:
            batch_vec = Variable(torch.from_numpy(batch_vec).float())

        # TODO check whether this can be done in batches
        if self.load_images:
            images = torch.stack(images)

        batch_pred = model.forward(images, batch_vec).data.cpu().numpy()
        batch_pred_2 = model.forward(images, batch_vec).data.cpu().numpy()

        logger.debug('Made predictions, {} seconds since start'.format(time.time() - start))

        predictions = [(pred[0], score) for pred, score in zip(batch_pred, batch_score)]

        # Sort predictions and replace with relevance scores.
        logger.debug('test log')

        # Shuffle the prediction before sorting to make sure equal predictions are 
        # in random order.
        np.random.shuffle(predictions)
        predictions = sorted(predictions, key=lambda x: -x[0])
        _, predictions = zip(*predictions)

        logger.debug('Sorted predictions, {} seconds since start'.format(time.time() - start))
        return predictions

    def add_scores(self, scores, to_add_scores):
        for key in to_add_scores.keys():
            if key not in scores:
                scores[key] = 0
            scores[key] += to_add_scores[key]

        return scores

    def avg_scores(self, scores, n):
        for key in scores.keys():
            scores[key] = scores[key] / n

        return scores

    def print_scores(self, scores):
        for key in sorted(list(scores.keys())):
            logger.info("{}_{} {}".format(self.prefix, key, scores[key]))

    """
    Append a dict of scores to file. 

    Path: Full path to the file to be stored
    Description: Prefix for run identification to the scores that will be appended 
    Scores: Dict with score name as key and score value as value.
    """
    def store_scores(self, path, description, scores):
        with open(path, "a") as f:
            scores = " ".join(["{0}:{1:.4f}".format(k, scores[k]) for k in sorted(list(scores.keys()))])
            f.write("{} {}\n".format(description, scores))

    def _log_scores(self, scores, tf_logger, epoch):
        for key in scores.keys():
            tf_logger.log_value('{}_{}'.format(self.prefix, key), scores[key], epoch)
        
    def eval(self, model, tf_logger=None, epoch=None):
        self.failed = 0 
        scores = {}
        for q_id in self.queries.keys():
            self.add_scores(scores, self._eval_query(q_id, model))

        n = len(self.queries.keys()) - self.failed
        scores = self.avg_scores(scores, n)

        self.print_scores(scores)
        if tf_logger is not None:
            self._log_scores(scores, tf_logger, epoch)

        return scores



    # TODO REWRITE YOURSELF
    def dcg_at_k(self, r, k, method=0):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    # TODO REWRITE YOURSELF
    def ndcg_at_k(self, r, k, method=0):
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k, method) / dcg_max

    def precision_at_k(self, r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        >>> r = [0, 0, 1]
        >>> precision_at_k(r, 1)
        0.0
        >>> precision_at_k(r, 2)
        0.0
        >>> precision_at_k(r, 3)
        0.33333333333333331
        >>> precision_at_k(r, 4)
        Traceback (most recent call last):
            File "<stdin>", line 1, in ?
        ValueError: Relevance score length < k
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(self, r):
        """Score is average precision (area under PR curve)
        Relevance is binary (nonzero is relevant).
        >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> delta_r = 1. / sum(r)
        >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
        0.7833333333333333
        >>> average_precision(r)
        0.78333333333333333
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Average precision
        """
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)
