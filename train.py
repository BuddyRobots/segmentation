import argparse
import os
import sys
import tensorflow as tf


BATCH_SIZE = 1
NUM_STEPS = 4000
LEARNING_RATE = 0.02
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
    return parser.parse_args()


def main():
	args = get_arguments()

    with open(args.seg_params, 'r') as f:
        seg_params = json.load(f)

    # Create coordinator.
    coord = tf.train.Coordinator()

    net = SegModel(
    	batch_size=args.batch_size,
    	dilations=seg_params['dilations'])

    loss = net.loss(input_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)
