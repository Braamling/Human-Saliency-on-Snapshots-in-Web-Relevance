from utils.featureStorage import FeatureStorage
from utils.customFeatureIterator import CustomFeatureIterator
import argparse

# def parse_pyndri(feature_storage, trec_iterator):
#     scorer = PyndriScorer(FLAGS.context_path)

#     for query_id, query, documents in trec_iterator.query_document_iterator():
#         doc_ids = list(documents.keys())
#         for doc_id, score in scorer.bm_scores(doc_ids, query):
#             feature_storage.add_query_document_feature(query_id, doc_id, documents[doc_id],
#                                                       'bm25', score)

#     for query_id, query, documents in trec_iterator.query_document_iterator():
#         doc_ids = list(documents.keys())
#         for doc_id, score in scorer.tfidf_scores(doc_ids, query):
#             feature_storage.add_query_document_feature(query_id, doc_id, documents[doc_id],
#                                                       'tfidf', score)

#     for query_id, query, documents in trec_iterator.query_document_iterator():
#         doc_ids = list(documents.keys())
#         for doc_id, score in scorer.lm_scores(doc_ids, query):
#             feature_storage.add_query_document_feature(query_id, doc_id, documents[doc_id],
#                                                       'lm', score)

def add_custom_features(feature_storage, custom_feature_iterator):
    for query_id, doc_id, rel_score, features in custom_feature_iterator.feature_iterator():
        for i, feature_score in enumerate(features):
            feature_storage.add_query_document_feature(query_id, doc_id, rel_score,
                                                       str(i), float(feature_score))
        print(query_id, doc_id)


def main():
    feature_storage = FeatureStorage(FLAGS.hdf5_path)
    custom_feature_iterator = CustomFeatureIterator(FLAGS.feature_path)

    add_custom_features(feature_storage, custom_feature_iterator)

    print(feature_storage.get_pairs())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_path', type=str, default='storage/custom_features',
                        help='The location of the content that should be added.')
    parser.add_argument('--hdf5_path', type=str, default='storage/hdf5/context_features.hdf5',
                        help='The location where to store and load the context_features.')

    FLAGS, unparsed = parser.parse_known_args()

    main()