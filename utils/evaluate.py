import torch
import numpy as np
import time
from preprocessing.contextualFeaturesGenerator.utils.featureStorage import FeatureStorage
from torch.autograd import Variable

import logging 
import time

logger = logging.getLogger('Evaluate')
"""
This class can be used to easily evaluate a LTR model in various 
stages of the training process.

Evaluation measures are taken from https://gist.github.com/bwhite/3726239
"""
class Evaluate():
    def __init__(self, hdf5_path, dataset, images, prefix):
        self.storage = FeatureStorage(hdf5_path)
        self.dataset = dataset
        self.prepare_eval_data()
        self.use_gpu = torch.cuda.is_available()
        self.images = images
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
        except Exception as e:
            # logger.warning("query {} gave an exception".format(query_id))
            self.failed += 1
            return {}


        return scores

    def _get_scores(self, query_id, model):
        logger.debug('Starting to prepare {} batch to evaluate query {}'.format(self.prefix, query_id))
        start = time.time()

        predictions = []
        batch_vec = []
        batch_score = []
        for doc, score in self.queries[query_id]:
            image, vec, rel_score = self.dataset.get_document(doc, query_id)

            batch_vec.append(vec)
            batch_score.append(score)

            if score is not rel_score:
                print(query_id, doc, score, rel_score, vec)

        batch_vec = np.vstack( batch_vec )
        # print(batch_vec)
        logger.debug('Batch ready, {} seconds since start'.format(time.time() - start))
        if self.use_gpu:
            batch_vec = Variable(torch.from_numpy(batch_vec).float().cuda())
        else:
            batch_vec = Variable(torch.from_numpy(batch_vec).float())

        # TODO check whether this can be done in batches
        if not self.images:
            image = None

        # batch_pred = model.forward(batch_vec).data.numpy()
        batch_pred = model.forward(image, batch_vec).data.numpy()
        # batch_pred = tmp

        logger.debug('Made predictions, {} seconds since start'.format(time.time() - start))

        predictions = [(pred[0], score) for pred, score in zip(batch_pred, batch_score)]
        # print(predictions)
        # Sort predictions and replace with relevance scores.
        logger.debug('test log')

        # Shuffle the prediction before sorting to make sure equal predictions are 
        # in random order.
        np.random.shuffle(predictions)
        predictions = sorted(predictions, key=lambda x: -x[0])
        _, predictions = zip(*predictions)

        logger.debug('Sorted predictions, {} seconds since start'.format(time.time() - start))
        return predictions

    def _add_scores(self, scores, to_add_scores):
        for key in to_add_scores.keys():
            if key not in scores:
                scores[key] = 0
            scores[key] += to_add_scores[key]

        return scores

    def _print_scores(self, scores):
        n = len(self.queries.keys()) - self.failed
        for key in scores.keys():
            print("{}_{} {}".format(self.prefix, key, scores[key]/n))

    def _log_scores(self, scores, tf_logger, epoch):
        n = len(self.queries.keys())
        for key in scores.keys():
            tf_logger.log_value('{}_{}'.format(self.prefix, key), scores[key]/n, epoch)
        
    def eval(self, model, tf_logger=None, epoch=None):
        self.failed = 0 
        scores = {}
        for q_id in self.queries.keys():
            self._add_scores(scores, self._eval_query(q_id, model))

        self._print_scores(scores)
        if tf_logger is not None:
            self._log_scores(scores, tf_logger, epoch)



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
