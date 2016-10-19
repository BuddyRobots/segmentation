import tensorflow as tf
from tensorflow.contrib import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def image_label_list():
    image_name_list = [ ]
    label_name_list = [ ]
    for (dirpath, dirnames, filenames) in os.walk('training_set/labels'):
        label_name_list.extend(map(lambda x: 'training_set/labels/' + x, filenames))
        image_name_list.extend(map(lambda x: 'training_set/images/' + x.replace('.dat', '.png'), filenames))
        break
    return image_name_list, label_name_list

def create_inputs():

    image_name_list, label_name_list = image_label_list()

    # image_names = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    # label_names = ops.convert_to_tensor(label_list, dtype=dtypes.string)

    seed = np.random.randint(1000)

    image_name_queue = tf.train.string_input_producer(image_name_list, seed=seed)
    label_name_queue = tf.train.string_input_producer(label_name_list, seed=seed)

    image_reader = tf.WholeFileReader()
    _, image_content = image_reader.read(image_name_queue)
    image_tensor = tf.image.decode_png(image_content, channels=3)

    label_reader = tf.WholeFileReader()
    _, label_content = label_reader.read(label_name_queue)
    label_tensor = tf.decode_raw(label_content, tf.uint8)
    # tf.reshape(label_tensor, image_tensor.get_shape())

    return image_tensor, label_tensor

create_inputs()
