from utils.featureStorage import FeatureStorage
import argparse
import numpy
import math

def main():
    feature_storage = FeatureStorage(FLAGS.hdf5_path)
    train_storage = FeatureStorage(FLAGS.target.format("train"))
    test_storage = FeatureStorage(FLAGS.target.format("test"))

    # Make the split
    queries = feature_storage.get_queries()
    train_len = int(len(queries) * FLAGS.train_split)
    numpy.random.shuffle(queries)
    training, test = queries[:train_len], queries[train_len:]

    for q_id in training:
        feature_storage.f.copy(str(q_id), train_storage.f)

    for q_id in test:
        feature_storage.f.copy(str(q_id), test_storage.f)
    # print(train_storage.print_index())
    # # feature_storage.f.copy("201", train_storage.f)
    print(train_storage.print_index())
    print()
    print(test_storage.print_index())
    # print(feature_storage.get_queries())
    # print(feature_storage.get_pairs())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hdf5_path', type=str, default='storage/hdf5/context_features.hdf5',
                        help='The location where to store and load the context_features.')
    parser.add_argument('--target', type=str, default='storage/hdf5/test_split_{}.hdf5',
                        help='The location where all splits should be stored.')
    parser.add_argument('--train_split', type=float, default=.8,
                        help='The part of queries to use for training.')


    FLAGS, unparsed = parser.parse_known_args()

    main()