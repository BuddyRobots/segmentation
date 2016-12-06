import tensorflow as tf
from tensorflow.contrib import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def image_label_list(training_set_dir):
	image_name_list = [ ]
	label_name_list = [ ]
	for (dirpath, dirnames, filenames) in os.walk(training_set_dir + '/labels'):
		print "Training set has " + str(len(filenames)) + " files."
	for (dirpath, dirnames, filenames) in os.walk(training_set_dir + '/labels'):
		label_name_list.extend(map(lambda x: training_set_dir + '/labels/' + x, filenames))
		image_name_list.extend(map(lambda x: training_set_dir + '/images/' + x.replace('.dat', '.png'), filenames))
		break
	return image_name_list, label_name_list

def create_inputs(input_channel, training_set_dir, batch_size):

	image_name_list, label_name_list = image_label_list(training_set_dir)

	seed = np.random.randint(1000)

	image_name_queue = tf.train.string_input_producer(image_name_list, seed=seed)
	label_name_queue = tf.train.string_input_producer(label_name_list, seed=seed)

	image_reader = tf.WholeFileReader()
	_, image_content = image_reader.read(image_name_queue)
	image_tensor = tf.image.decode_png(image_content, channels=input_channel)

	label_reader = tf.WholeFileReader()
	_, label_content = label_reader.read(label_name_queue)
	label_tensor = tf.decode_raw(label_content, tf.uint8)

	if batch_size > 1:
		image_batch, label_batch = tf.train.shuffle_batch(tensors=[image_tensor, label_tensor],
														  batch_size=batch_size,
														  shapes=[[224, 224, 3], [224 * 224]],
														  capacity=50000,
														  min_after_dequeue=10000)

		return image_batch, label_batch
	else:
		return image_tensor, label_tensor
