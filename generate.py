import argparse
import os
import sys
import tensorflow as tf

from model import SegModel


def get_arguments():
	parser = argparse.ArgumentParser(description='EspcnNet generation script')
	parser.add_argument('checkpoint', type=str,
						help='Which model checkpoint to generate from')
	parser.add_argument('--image', type=str,
						help='The low-resolution image waiting for processed.')
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

	with open("./params.json", 'r') as f:
		params = json.load(f)

	if check_params(seg_params) == False:
		return

	sess = tf.Session()

if __name__ == '__main__':
	main()