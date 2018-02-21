from utils.featureStorage import FeatureStorage
from utils.trecIterator import TrecIterator
from utils.pyndriScorer import PyndriScorer
import argparse

def parse_pyndri(feature_storage, trec_iterator):
    scorer = PyndriScorer(FLAGS.context_path)

    for query_id, query, documents in trec_iterator.query_document_iterator():
        doc_ids = list(documents.keys())
        for doc_id, score in scorer.bm_scores(doc_ids, query):
            feature_storage.add_query_document_feature(query_id, doc_id, documents[doc_id],
                                                      'bm25', score)

    for query_id, query, documents in trec_iterator.query_document_iterator():
        doc_ids = list(documents.keys())
        for doc_id, score in scorer.tfidf_scores(doc_ids, query):
            feature_storage.add_query_document_feature(query_id, doc_id, documents[doc_id],
                                                      'tfidf', score)

    for query_id, query, documents in trec_iterator.query_document_iterator():
        doc_ids = list(documents.keys())
        for doc_id, score in scorer.lm_scores(doc_ids, query):
            feature_storage.add_query_document_feature(query_id, doc_id, documents[doc_id],
                                                      'lm', score)

def main():
    feature_storage = FeatureStorage(FLAGS.hdf5_path)
    trec_iterator = TrecIterator(FLAGS.trec_path)

    if FLAGS.context_type == 'pyndri':
        parse_pyndri(feature_storage, trec_iterator)

    print(feature_storage.get_pairs())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--trec_path', type=str, default='storage/TREC/web-2013',
                        help='The folder location of all TREC files that should be added.')
    parser.add_argument('--context_type', type=str, default=None,
                        help='The type of context that should be added.')
    parser.add_argument('--context_path', type=str, default='storage/pyndri/clueweb12b',
                        help='The location of the content that should be added.')
    parser.add_argument('--hdf5_path', type=str, default='storage/hdf5/context_features.hdf5',
                        help='The location where to store and load the context_features.')

    FLAGS, unparsed = parser.parse_known_args()

    main()