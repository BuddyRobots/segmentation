import argparse
import os
import sys
import json
from scipy import misc
import tensorflow as tf

from model import SegModel

BATCH_SIZE = 1
KLASS = 2
INPUT_CHANNEL = 1
SEG_PARAMS = './seg_params.json'

def get_arguments():
	parser = argparse.ArgumentParser(description='Generation script')
	parser.add_argument('checkpoint', type=str,
						help='Which model checkpoint to generate from')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='How many image files to process at once.')
	parser.add_argument('--klass', type=str, default=KLASS,
						help='Number of segmentation classes.')
	parser.add_argument('--input_channel', type=str, default=INPUT_CHANNEL,
						help='Number of input channel.')
	parser.add_argument('--seg_params', type=str, default=SEG_PARAMS,
						help='JSON file with the network parameters.')
	parser.add_argument('--image', type=str,
						help='The limage waiting for processed.')
	parser.add_argument('--out_path', type=str,
						help='The output path for the segmentation result image')
	return parser.parse_args()

def check_params(seg_params):
	if seg_params['network_type'] == "atrous":
		if len(seg_params['dilations']) - len(seg_params['channels']) != 1:
			print("For atrous net, the length of 'dilations' must be greater then the length of 'channels' by 1.")
			return False
		if len(seg_params['kernel_size']) != len(seg_params['dilations']):
			print("For atrous net, the length of 'dilations' must be equal to the length of 'kernel_size'.")
			return False
	return True

def main():
	args = get_arguments()

	with open("./seg_params.json", 'r') as f:
		seg_params = json.load(f)

	if check_params(seg_params) == False:
		return

	net = SegModel(
		input_channel=args.input_channel,
		klass=args.klass,
		batch_size=args.batch_size,
		network_type=seg_params['network_type'],
		kernel_size=seg_params['kernel_size'],
		dilations=seg_params['dilations'],
		strides=seg_params['strides'],
		channels=seg_params['channels'])

	input_image = tf.placeholder(tf.uint8)
	output_image = net.generate(input_image)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, args.checkpoint)

	input_image_data = misc.imread(args.image, mode='L')

	output_image_data = sess.run(output_image, feed_dict={input_image: input_image_data})
	width, height = input_image_data.shape
	for x in range(width):
		for y in range(height):
			if input_image_data[x][y] == 255:
				if output_image_data[0][x][y] == 0:
					input_image_data[x][y] = 125
	misc.imsave(args.out_path, input_image_data)

if __name__ == '__main__':
	main()