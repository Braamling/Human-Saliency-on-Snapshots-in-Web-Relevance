import torch
import numpy as np

from preprocessing.contextualFeaturesGenerator.utils.featureStorage import FeatureStorage
from torch.autograd import Variable

"""
This class can be used to easily evaluate a LTR model in various 
stages of the training process.
"""
class Evaluate():
    def __init__(self, hdf5_path, dataset, images):
        self.storage = FeatureStorage(hdf5_path)
        self.dataset = dataset
        self.prepare_eval_data()
        self.use_gpu = torch.cuda.is_available()
        self.images = images

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
        scores["ndcg_1"] = self.ndcg_at_k(predictions, 1)
        scores["ndcg_2"] = self.ndcg_at_k(predictions, 2)
        scores["ndcg_5"] = self.ndcg_at_k(predictions, 5)

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

            predictions.append((model.forward(image, vec).data[0][0], score))

        # Sort predictions and replace with relevance scores.
        predictions = sorted(predictions, key=lambda x: -x[0])
        _, predictions = zip(*predictions)

        return predictions

    def _add_scores(self, scores, score):
        for key in scores.keys():
            scores[key] += score[key]

        return scores

    def _print_scores(self, scores):
        print("scores: ")
        n = len(self.queries)
        for key in scores.keys():
            print(key, scores[key]/n)
        
    def eval(self, model):
        scores = {}
        scores["ndcg_1"] = 0
        scores["ndcg_2"] = 0
        scores["ndcg_5"] = 0
        for q_id in self.queries.keys():
            self._add_scores(scores, self._eval_query(q_id, model))
        self._print_scores(scores)



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
