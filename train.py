import argparse
import os
import sys
import tensorflow as tf

from reader import create_inputs
from model import SegModel


BATCH_SIZE = 1
NUM_STEPS = 4000
LEARNING_RATE = 0.02
LOGDIR_ROOT = './logdir'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SEG_PARAMS = './wavenet_params.json'
L2_REGULARIZATION_STRENGTH = 0

def get_arguments():
    parser = argparse.ArgumentParser(description='segmentation network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many image files to process at once.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--seg_params', type=str, default=SEG_PARAMS,
                        help='JSON file with the network parameters.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Disabled by default')
    parser.add_argument('--logdir_root', type=str, default=LOGDIR_ROOT,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    return parser.parse_args()

def get_default_logdir(logdir_root):
    print(logdir_root)
    print(STARTED_DATESTRING)
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir

def main():
	args = get_arguments()

    with open(args.seg_params, 'r') as f:
        seg_params = json.load(f)

    logdir_root = args.logdir_root
    logdir = get_default_logdir(logdir_root)

    image, labels = create_inputs()

    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=1)

    queue = tf.RandomShuffleQueue(
        256,
        ['uint8', 'uint8'],
        shapes=[(params['lr_size'], params['lr_size'], 3),
                (params['lr_size'] - params['edge'], params['lr_size'] - params['edge'], 3 * params['ratio']**2)])
    enqueue = queue.enqueue([lr_image, hr_data])
    batch_input = queue.dequeue_many(args.batch_size)

    # Create coordinator.
    coord = tf.train.Coordinator()

    net = SegModel(
    	batch_size=args.batch_size,
    	dilations=seg_params['dilations'])

    loss = net.loss(input_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)
