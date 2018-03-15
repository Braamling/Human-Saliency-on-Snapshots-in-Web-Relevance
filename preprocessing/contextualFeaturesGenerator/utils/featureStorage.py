import numpy as np
import os.path

from .LETORIterator import LETORIterator

"""
This class acts as an interface for the hdf5 storage containing all 
query-document pairs with their contextual features and judgment scores.

The interface can both be used to add new queries, documents and features, 
but also to iterate over the content of a hdf5 file.
"""
class FeatureStorage():
    def __init__(self, path):
        # self.f = h5py.File(hdf5_path, "a")
        self.letorIterator = LETORIterator(path)
        self.pairs = []
        self.scores = {}
        self.queries = {}
        self.q_docs = {}

        self.parse()


    def parse(self):
        for query_id, doc_id, rel_score, features in self.letorIterator.feature_iterator():
            query_id, rel_score = int(query_id), int(rel_score)
            if query_id not in self.queries:
                self.queries[query_id] = {}

            if rel_score not in self.queries[query_id]:
                self.queries[query_id][rel_score] = {}

            features = [float(f) for f in features]
            self.queries[query_id][rel_score][doc_id] = np.asarray(features)


    """
    Retrieve all features for a document and query pair. Both query and non
    query specific features are retrieved.
    
    query_id: int, id of the query
    document_id: str, id of the document
    score: int, judgement score for the document query pair
    """
    def get_query_document_features(self, query_id, doc_id, rel_score):
        return self.queries[query_id][rel_score][doc_id]

    """
    Get all documents in a query in an array with typle values of (score, document_id)
    """
    def get_documents_in_query(self, query_id):
        documents = []
        for rel_score in self.queries[query_id]:
            for doc_id in self.queries[query_id][rel_score]:
                documents.append((rel_score, doc_id))
                # documents = np.concatenate((documents, list(score.keys())))
             
        return documents


    """
    Get all available query-document pairs in an array of with tuple values of
    (query_id, score, document_id, feature_vectore)
    """
    def get_all_entries(self):
        for query_id in self.queries:
            for rel_score in self.queries[query_id]:
                for doc_id in self.queries[query_id][rel_score]:
                    vec = self.queries[query_id][rel_score][doc_id]
                    yield (query_id, rel_score, doc_id, vec)


    """
    Get all available query-document pairs in an array of with tuple values of
    (query_id, score, document_id, feature_vectore)
    """
    def get_queries(self):
        return list(self.queries.keys())


    """
    Get a sorted list of tuples (doc_id, score) for a specific query
    """
    def get_scores(self, query_id):
        if query_id not in self.scores:
            self.scores[query_id] = []

            for scores in self.queries[query_id].keys():
                for doc_id in self.queries[query_id][scores].keys():
                    self.scores[query_id].append((doc_id, int(scores)))

            self.scores[query_id] = sorted(self.scores[query_id], key=lambda x: -x[1])

        return self.scores[query_id]

    """
    Add query to query list
    """
    # def _add_score(self, route):
        # route = route.split( _id, int(score)))

    def print_index(self):
        self.f.visit(self.printname)

    def printname(self, name):
        print(name)