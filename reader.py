import tensorflow as tf
from tensorflow.contrib import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from prepare_data import my_shuffle


def image_label_list():
    image_name_list = [ ]
    label_name_list - [ ]
    for (dirpath, dirnames, filenames) in os.walk('training_set/labels'):
        label_name_list.extend(map(lambda x: 'training_set/labels/' + x, filenames))
        image_name_list.extend(map(lambda x: 'training_set/images/' + x + '.png', filenames))
        break
    return image_name_list, label_name_list

def read_image_and_label(input_queue):
    image_content = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_content, channels=1)

    label_content = tf.WholeFileReader()
    _, label_value = reader.read(files)
    label = tf.decode_raw(label_value, tf.uint8)
    label = tf.reshape(label, tf.get_shape())
    return image, label

def create_inputs():

    image_name_list, label_name_list = image_label_list()

    image_names = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    label_names = ops.convert_to_tensor(label_list, dtype=dtypes.string)

    input_queue = tf.train.slice_input_producer([images, labels])

    image, label = read_image_and_label(input_queue)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=1)
    return image_batch, label_batch

create_inputs()
