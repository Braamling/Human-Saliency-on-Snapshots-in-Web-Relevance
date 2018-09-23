import argparse
import itertools
import subprocess


def tune():
    commands = FLAGS.run_cmd.split()
    if FLAGS.input_type in ('snapshots', 'saliency'):
        commands += ['--query_specific', 'False']
    else:
        commands += ['--query_specific', 'True']

    if FLAGS.input_type in ('snapshots', 'saliency', 'highlights'):
        commands += ['--load_images', 'True']
    else:
        commands += ['--load_images', 'True']

    if FLAGS.input_type in ('saliency'):
        commands += FLAGS.saliency_path.split()

    commands += FLAGS.image_path.format(FLAGS.input_type).split()
    commands += FLAGS.cache_path.format(FLAGS.model, FLAGS.input_type).split()
    commands += FLAGS.infrastructure.format(FLAGS.infrastructure_type).split()
    commands += FLAGS.cache_vector.format(FLAGS.cache_vector_size).split()

    parameter_keys = sorted(FLAGS.parameters.keys())

    iter = list(
        itertools.product(*[FLAGS.parameters[x] for x in parameter_keys]))

    for i, parameter_values in enumerate(iter):
        parameters = []
        for x, y in zip(parameter_keys, parameter_values):
            parameters += [x, y]

        run = commands + parameters

        subprocess.call(run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='resnet152',
                        help='Name of the model to train.')
    parser.add_argument('--infrastructure_type', type=str, default='transform_cache',
                        help='Name of the model to train.')
    parser.add_argument('--input_type', type=str, default='snapshots',
                        help='Name of the input to train on.')
    parser.add_argument('--cache_vector_size', type=str, default='2048',
                        help='Name of the input to train on.')

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.image_path = '--image_path storage/images_224x224/{}/'
    FLAGS.saliency_path = '--saliency_path storage/images_224x224/saliency/'
    FLAGS.cache_vector = '--cache_vector_size {}'
    FLAGS.description = '--description {}_{}_lr_{}_cd_{}_vd_{}_vl_{}_m_{}_b_{}'
    FLAGS.cache_path = '--cache_path storage/model_cache/{}-{}-cache'
    FLAGS.infrastructure = '--model {}'
    FLAGS.run_cmd = 'python3 train.py --content_feature_size 11 --content_feature_dir storage/clueweb12_web_trec  --epochs 20'

    FLAGS.parameters = {
        '--learning_rate':          ['0.0001', '0.00005'],
        '--classification_dropout': ['0.10', '0.20'],
        '--visual_dropout':         ['0.50', '0.10'],
        '--visual_layers':          ['4096x4096x4096x4096', '4096x4096x4096', '2048x2048x2048x2048',
                                     '1024x1024x1024x1024x1024x1024',
                                     '1024x1024x1024x1024', '2048x4096x2048'],
        '--optimize_on':            ['ndcg@1', 'ndcg@10', 'p@1', 'p@10', 'map'],
        '--batch_size':             ['100']
    }

    tune()
