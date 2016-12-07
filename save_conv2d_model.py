import json
import sys
import os
import tensorflow as tf
import numpy as np

from reader import create_inputs
from model import SegModel
from Conv2DModel import Conv2DModel
from train import get_arguments
from train import check_params
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def main():
    load_model_dir = "/Users/ZhangPengfei/Work/git/segmentation/logdir/owl.ckpt"
    save_model_dir = "/Users/ZhangPengfei/Work/git/segmentation/logdir/conv2d/model"

    args = get_arguments()

    with open(args.seg_params, 'r') as f:
        seg_params = json.load(f)

    if not check_params(seg_params):
        return

    # ----------------------------------------------
    # ---------- original atrous network -----------
    # ----------------------------------------------
    original_graph = tf.Graph()

    with original_graph.as_default():
        original_net = SegModel(
            input_channel=args.input_channel,
            klass=args.klass,
            batch_size=args.batch_size,
            network_type=seg_params['network_type'],
            kernel_size=seg_params['kernel_size'],
            dilations=seg_params['dilations'],
            strides=seg_params['strides'],
            channels=seg_params['channels']
        )

        sess = tf.Session(graph=original_graph)
        saver = tf.train.Saver()
        saver.restore(sess, load_model_dir)

        # print([n.name for n in original_graph.as_graph_def().node])

    filter_data_list = get_atrous_filter_narrays(original_net, sess, dilations=seg_params['dilations'])
    bias_data_list = get_atrous_bias_narrays(original_net, sess)

    # print(bias_data_list)

    original_graph.finalize()
    sess.close()

    # ----------------------------------------------
    # ----------- new conv2d network ---------------
    # ----------------------------------------------

    new_graph = tf.Graph()

    with new_graph.as_default():
        image_batch, label_batch = create_inputs(input_channel=args.input_channel,
                                                 training_set_dir=args.training_set_dir,
                                                 batch_size=args.batch_size)
        queue = tf.FIFOQueue(256, ['uint8', 'uint8'])
        enqueue = queue.enqueue([image_batch, label_batch])
        input_data = queue.dequeue()

        new_net = Conv2DModel(
            input_channel=args.input_channel,
            klass=args.klass,
            batch_size=args.batch_size,
            network_type="conv2d",
            kernel_size=seg_params['conv2d_kernel_size'],
            dilations=seg_params['conv2d_dilations'],
            strides=seg_params['strides'],
            channels=seg_params['channels'],
            filter_data_list=filter_data_list,
            bias_data_list=bias_data_list
        )

        loss = new_net.loss(input_data)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)

        sess = tf.Session(graph=new_graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        # print([n.name for n in new_graph.as_graph_def().node])
        # print(new_graph.get_tensor_by_name("conv2d_filter_1:0").eval(session=sess))

        saver = tf.train.Saver()
        save(saver, sess, save_model_dir)

        with sess.as_default():
            minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["NETWORK_OUTPUT"])
            tf.train.write_graph(minimal_graph, '.', 'model.pb', as_text=False)


def get_atrous_filter_narrays(net, sess, dilations):
    filters = list()
    if net.network_type == "atrous":
        for layer_idx, dilation in enumerate(net.dilations):
            flt = sess.run(net.variables['atrous']['filters'][layer_idx])
            flt = fill_zeros(flt, dilation)
            filters.append(flt)
    return filters


def get_atrous_bias_narrays(net, sess):
    biases = list()
    if net.network_type == "atrous":
        for i, channel in enumerate(net.channels):
            if i == 0:
                continue
            bias = sess.run(net.variables['atrous']['biases'][i - 1])
            biases.append(bias)
    return biases


def fill_zeros(data, dilation):
    shape = [2*dilation + 1, 2*dilation + 1, data.shape[2], data.shape[3]]
    new_data = np.zeros(shape=shape, dtype=np.float32)

    for i in xrange(0, 3):
        for j in xrange(0, 3):
            new_data[i*dilation, j*dilation, :, :] = data[i, j, :, :]
    return new_data


def save(saver, sess, save_model_dir):
    model_name = 'model'
    checkpoint_path = os.path.join(save_model_dir, model_name)
    print('Storing checkpoint to {} ...'.format(save_model_dir))
    sys.stdout.flush()

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    saver.save(sess, checkpoint_path)
    print(' Done.')


if __name__ == '__main__':
    main()
