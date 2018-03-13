import torch
import numpy as np

from preprocessing.contextualFeaturesGenerator.utils.featureStorage import FeatureStorage
from torch.autograd import Variable

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
        predictions = self._get_scores(query_id, model)

        scores = {}
        scores["ndcg@1"] = self.ndcg_at_k(predictions, 1)
        scores["ndcg@5"] = self.ndcg_at_k(predictions, 5)
        scores["ndcg@10"] = self.ndcg_at_k(predictions, 10)
        scores["p@1"] = self.precision_at_k(predictions, 1)
        scores["p@5"] = self.precision_at_k(predictions, 5)
        scores["p@10"] = self.precision_at_k(predictions, 10)
        scores["map"] = self.average_precision(predictions)

        return scores

    def _get_scores(self, query_id, model):
        predictions = []
        for doc, score in self.queries[query_id]:
            image, vec, rel_score = self.dataset.get_document(doc)
            if score is not rel_score:
                print(doc, score, rel_score, vec)
            # assert score is rel_score, "document {} has different rel scores".format(doc)

            if self.use_gpu:
                vec = Variable(torch.from_numpy(vec).float().cuda())
            else:
                vec = Variable(torch.from_numpy(vec).float())

            # TODO check whether this can be done in batches
            if not self.images:
                image = None

            pred = model.forward(image, vec).data[0]
            if type(pred) is not float:
                pred = pred[0]
            predictions.append((pred, score))

        # Sort predictions and replace with relevance scores.
        predictions = sorted(predictions, key=lambda x: -x[0])
        _, predictions = zip(*predictions)

        return predictions

    def _add_scores(self, scores, score):
        for key in scores.keys():
            if key not in scores:
                scores[key] = 0
            scores[key] += score[key]

        return scores

    def _print_scores(self, scores):
        print("scores: ")
        n = len(self.queries)
        for key in scores.keys():
            print("{}_{} {}".format(self.prefix, key, scores[key]/n))

    def _log_scores(self, scores, tf_logger, epoch):
        n = len(self.queries)
        for key in scores.keys():
            tf_logger.log_value('{}_{}'.format(prefix, key), scores[key]/n, epoch)
        
    def eval(self, model, tf_logger=None, epoch=None):
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
