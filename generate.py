import argparse
import os
import sys
import json
from scipy import misc
import numpy as np
import tensorflow as tf

from model import SegModel

BATCH_SIZE = 1
KLASS = 6
INPUT_CHANNEL = 3
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
	parser.add_argument('--batch_generate', type=bool, default=False,
						help='Whether to generate for multiple images.')
	parser.add_argument('--input_path', type=str,
						help='The name of the folder in which the images and labels are (when batch_generate is True).')
	parser.add_argument('--image', type=str,
						help='The image waiting for processed.')
	parser.add_argument('--label', type=str,
						help='The label data, used to determine which pixels are unnecessary to calculate.')
	parser.add_argument('--out_path', type=str,
						help='The output path for the segmentation result image (if batch_generate is True, this is a dir, else is a file.)')
	return parser.parse_args()

def check_params(seg_params):
	if seg_params['network_type'] != 'atrous' and seg_params['network_type'] != 'deconv':
		print("Network type can only be atrous or deconv.")
		return False
	if seg_params['network_type'] == "atrous":
		if len(seg_params['dilations']) - len(seg_params['channels']) != 1:
			print("For atrous net, the length of 'dilations' must be greater then the length of 'channels' by 1.")
			return False
		if len(seg_params['kernel_size']) != len(seg_params['dilations']):
			print("For atrous net, the length of 'dilations' must be equal to the length of 'kernel_size'.")
			return False
	if seg_params['network_type'] == 'deconv':
		if len(seg_params['strides']) - len(seg_params['channels']) != 1:
			print("For deconv net, the length of 'strides' must be greater then the length of 'channels' by 1.")
			return False
		if len(seg_params['kernel_size']) != len(seg_params['strides']):
			print("For deconv net, the length of 'strides' must be equal to the length of 'kernel_size'.")
			return False
	return True

def generate_one(sess, net, image, label, out_path, input_channel, klass, input_image, output_image):
	if input_channel == 1:
		input_image_data = misc.imread(image, mode='L')
		input_image_data = np.expand_dims(input_image_data, axis=2)
	else:
		input_image_data = misc.imread(image, mode='RGB')

	output_image_data = sess.run(output_image, feed_dict={input_image: input_image_data})
	height, width, _ = input_image_data.shape

	output_label = np.zeros([height, width], dtype='int32')

	# label_data = np.fromfile(label, dtype='byte')
	# label_data = label_data.reshape([height, width])
	for x in range(height):
		for y in range(width):
			# if label_data[x][y] > 0:
			output_label[x][y] = 255 * (output_image_data[0][x][y] + 1) / klass
			# if input_image_data[x][y][0] == 0:
			# 	output_label[x][y] = 0
			# else:
			# 	output_label[x][y] = 255 * (output_image_data[0][x][y] + 1) / klass
			# output_label[x][y] = 255 * (output_image_data[0][x][y] + 1) / klass
			# if output_image_data[0][x][y] == 0:
			# 	input_image_data[x][y][0] = input_image_data[x][y][0] / 2 + 255 / 2
			# 	input_image_data[x][y][1] = input_image_data[x][y][1] / 2
			# 	input_image_data[x][y][2] = input_image_data[x][y][2] / 2
	misc.imsave(out_path, output_label)
	# misc.imsave(out_path, input_image_data)

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
		channels=seg_params['channels'],
		with_bn=seg_params['with_bn'],
		phase_train=False)

	input_image = tf.placeholder(tf.uint8)
	output_image = net.generate(input_image)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, args.checkpoint)

	# with sess.as_default():
	# 	# for v in tf.trainable_variables():
	# 	# 	vc = tf.constant(v.eval())
	# 	# 	tf.assign(v, vc)
	# 	tf.train.write_graph(sess.graph_def, "./", 'remy_model_2.pb', as_text=False)
	# return

	if args.batch_generate:
		for (dirpath, dirnames, filenames) in os.walk(args.input_path + '/images'):
			for filename in filenames:
				image = args.input_path + '/images/' + filename
				# label = args.input_path + '/labels/' + filename.replace('.png', '.dat')
				out_path = args.out_path + '/' + filename
				generate_one(sess=sess,
							 net=net,
							 image=image,
							 # label=label,
							 label="",
							 out_path=out_path,
							 input_channel=args.input_channel,
							 klass=args.klass,
							 input_image=input_image,
							 output_image=output_image)
	else:
		generate_one(sess=sess,
					 net=net,
					 image=args.image,
					 label=args.label,
					 out_path=args.out_path,
					 input_channel=args.input_channel,
					 klass=args.klass,
					 input_image=input_image,
					 output_image=output_image)

if __name__ == '__main__':
	main()
