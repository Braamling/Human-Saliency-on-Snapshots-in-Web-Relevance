import h5py
import numpy as np
import os.path

"""
This class acts as an interface for the hdf5 storage containing all 
query-document pairs with their contextual features and judgment scores.

The interface can both be used to add new queries, documents and features, 
but also to iterate over the content of a hdf5 file.
"""
class FeatureStorage():
    def __init__(self, hdf5_path):
        self.f = h5py.File(hdf5_path, "a")

    """
    Add a feature to a query document pair. 

    query_id: int, id of the query
    document_id: str, id of the document
    score: int, judgement score for the document query pair
    feature_name: str, name to provide to the feature
    value: float, value to add to the feature
    """
    def add_query_document_feature(self, query_id, document_id, score, feature_name, value):
        location = self.format_location(query_id, document_id, score)
        if location not in self.f:
            group = self.f.create_group(location)
        else:
            group = self.f[location]

        group.attrs[feature_name] = value

    """
    Add a feature to a document. 

    document_id: str, id of the document
    feature_name: str, name to provide to the feature
    value: float, value to add to the feature
    """
    def add_document_feature(self, document_id, feature_name, value):
        if document_id not in self.f:
            group = self.f.create_group(document_id)
        else:
            group = self.f[document_id]

        group.attrs[feature_name] = value

    """
    Retrieve all document features that are non query specific

    document_id: str, document id to retrieve.
    """
    def get_all_document_features(self, document_id):
        if document_id in self.f:
            group = self.f[document_id]
            return {x: group.attrs[x] for x in group.attrs}

        return {}

    """
    Retrieve all features for a document and query pair. Both query and non
    query specific features are retrieved.
    
    query_id: int, id of the query
    document_id: str, id of the document
    score: int, judgement score for the document query pair
    """
    def get_all_query_document_features(self, query_id, document_id, score):
        location = self.format_location(query_id, document_id, score)

        group = self.f[location]

        features = self.get_all_document_features(document_id)
        for x in group.attrs:
            features[x] = group.attrs[x]

        return features

    """
    Convert a dict with features into a numpy vector with the correct ordering.

    features: dict, feature names as index with floats as feature values.
    """
    def make_feature_vector(self, features):
        # Create a list of sorted feature names
        feature_names = list(features.keys())
        feature_names.sort()

        vec = np.asarray([features[name] for name in feature_names])

        return vec

    """
    Format a query_id, document_id and score to a valid (sub)group location.

    query_id: int, id of the query
    document_id: str, id of the document
    score: int, judgement score for the document query pair
    """ 
    def format_location(self, query_id, document_id=None, score=None):
        if document_id is None:
            return "{}".format(query_id)
            
        return "{}/{}/{}".format(query_id, score, document_id)

    """
    Get all documents in a query in an array with typle values of (score, document_id)
    """
    def get_documents_in_query(self, query_id):
        self.pairs = []
        # self.f[query_id]
        self.f[str(query_id)].visit(self._add_document)
        return self.pairs

    """
    Add all query-document pairs with their score to the class pairs variable.

    route: str, the route of the (sub)group that is being visited.
    """
    def _add_document(self, route):
        route = route.split("/")
        if len(route) == 2:
            score, d_id = route
            self.pairs.append((int(score), d_id))

    """
    Get all available query-document pairs in an array of with tuple values of
    (query_id, score, document_id, feature_vectore)
    """
    def get_pairs(self):
        self.pairs = []
        self.f.visit(self._add_pair)
        return self.pairs

    """
    Add all query-document pairs with their score to the class pairs variable.

    route: str, the route of the (sub)group that is being visited.
    """
    def _add_pair(self, route):
        route = route.split("/")
        if len(route) == 3:
            q_id, score, d_id = route
            q_id, score = int(q_id), int(score)
            features = self.get_all_query_document_features(q_id, d_id, score)

            vec = self.make_feature_vector(features)

            self.pairs.append((q_id, score, d_id, vec))

    def print_index(self):
        self.f.visit(self.printname)

    def printname(self, name):
        print(name)