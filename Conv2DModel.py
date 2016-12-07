import tensorflow as tf


class Conv2DModel(object):
    def __init__(self,
                 input_channel,
                 klass,
                 batch_size,
                 network_type,
                 kernel_size,
                 dilations,
                 strides,
                 channels,
                 filter_data_list,
                 bias_data_list):
        self.input_channel = input_channel
        self.klass = klass
        self.batch_size = batch_size
        self.network_type = network_type
        if self.network_type == "conv2d":
            self.dilations = dilations
        if self.network_type == "deconv":
            self.strides = strides
        self.kernel_size = kernel_size
        # channels.insert(0, self.input_channel)
        # channels.append(self.klass)
        self.channels = channels
        self.filter_data_list = filter_data_list
        self.bias_data_list = bias_data_list

        self.variables = self._create_variables()

    def _create_variables(self):
        var = dict()
        if self.network_type == "conv2d":
            var['conv2d'] = dict()
            var['conv2d']['filters'] = list()

            print(self.filter_data_list[1].shape)


            for i, dilation in enumerate(self.dilations):
                var['conv2d']['filters'].append(create_variable('conv2d_filter_' + str(i),
                                                                self.filter_data_list[i]))
            var['conv2d']['biases'] = list()
            for i, channel in enumerate(self.channels):
                if i == 0:
                    continue
                var['conv2d']['biases'].append(create_bias_variable('conv2d_bias_' + str(i),
                                                                    self.bias_data_list[i - 1]))
        return var

    def loss(self, input_data):
        image, label = self._preprocess(input_data)

        image = tf.identity(image, name="NETWORK_INPUT")

        output = self._create_network(image)

        # value for the elements in label after preprocess:
        # -1: not care
        # i (i >= 0): the i-th class (0-based)
        # this is to match that the labels parameter for
        # tf.nn.sparse_softmax_cross_entropy_with_logits ranges from [0, num_classes]
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#sparse_softmax_cross_entropy_with_logits
        label_indicator = tf.greater(label, -1)
        effective_label = tf.boolean_mask(tensor=label,
                                          mask=label_indicator)

        output = tf.reshape(output, [-1, self.klass])
        effective_output = tf.boolean_mask(tensor=output,
                                           mask=label_indicator)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=effective_output,
            labels=effective_label)
        reduced_loss = tf.reduce_mean(loss)
        tf.scalar_summary('loss', reduced_loss)
        return reduced_loss

    def _preprocess(self, input_data, generate=False):
        if generate is True:
            image = input_data
            image = tf.cast(tf.expand_dims(image, 0), tf.float32)
            label = None
        else:
            image = input_data[0]
            label = input_data[1]
            if self.batch_size > 1:
                # if shuffle batch is used in reader.py, there is no need to expand dimensions
                image = tf.cast(image, tf.float32)
                label = tf.reshape(label, [self.batch_size, -1])
            else:
                image = tf.cast(tf.expand_dims(image, 0), tf.float32)
                label = tf.reshape(label, [-1])
            # value for the elements in label before preprocess can be:
            # 0: not care
            # i (i > 0): the i-th class (1-based)
            # this is because each element is saved as one byte (unsigned 8-bit int) in the label file
            # and its range is from 0 to 255
            label = tf.cast(label, tf.int32)
            label -= 1
        # tf.nn.conv2d(padding='SAME') always pads 0 to the input tensor,
        # thus make the value of the white pixels in the image 0
        image = 1.0 - image / 255.0
        return image, label

    def _create_network(self, input_data):
        if self.network_type == 'conv2d':
            current_layer = input_data
            for layer_idx, dilation in enumerate(self.dilations):
                if dilation == 1:
                    conv = tf.nn.conv2d(input=current_layer,
                                        filter=self.variables['conv2d']['filters'][layer_idx],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
                else:
                    print("Conv2DModel.py _create_network : FATAL have dilation != 1 !!")
                with_bias = tf.nn.bias_add(conv, self.variables['conv2d']['biases'][layer_idx])
                if layer_idx == len(self.dilations) - 1:
                    current_layer = with_bias
                else:
                    current_layer = tf.nn.relu(with_bias)
            retval = tf.identity(current_layer, name="NETWORK_OUTPUT")
            return current_layer


def create_variable(name, data):
    variable = tf.get_variable(name, initializer=tf.constant(data))
    return variable


def create_bias_variable(name, data):
    variable = tf.get_variable(name, initializer=tf.constant(data))
    return variable
