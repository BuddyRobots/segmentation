import tensorflow as tf

def create_variable(name, shape):
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	variable = tf.Variable(initializer(shape=shape), name=name)
	return variable

def create_bias_variable(name, shape):
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	return tf.Variable(initializer(shape=shape), name)


class SegModel(object):
	def __init__(self,
				 input_channel,
				 klass,
				 batch_size,
				 network_type,
				 kernel_size,
				 dilations,
				 strides,
				 channels):
		self.input_channel = input_channel
		self.klass = klass
		self.batch_size = batch_size
		self.network_type = network_type
		if self.network_type == "atrous":
			self.dilations = dilations
		if self.network == "deconv":
			self.strides = strides
		self.kernel_size = kernel_size
		channels.insert(0, self.input_channel)
		channels.append(self.klass)
		self.channels = channels
		self.variables = self._create_variables()

	def _create_variables(self):
		var = dict()

		if network_type == "atrous":
			var['atrous'] = dict()
			var['atrous']['filters'] = list()
			for i, dilation in enumerate(self.dilations):
				var['atrous']['filters'].append(create_variable('filter',
													  [self.kernel_size[i],
													   self.kernel_size[i],
													   self.channels[i],
													   self.channels[i+1]]))
			var['atrous']['biases'] = list()
			for i, channel in enumerate(self.channels):
				if i == 0:
					continue
				var['atrous']['biases'].append(create_bias_variable('bias', [channel]))
		return var

	def _preprocess(self, input_data):
		label = input_data[1]
		label = tf.reshape(label, [-1])
		# value for the elements in label before preprocess can be:
		#	0: not care
		#	i (i > 0): the i-th class (1-based)
		# this is because eac element is saved as one byte (unsigned 8-bit int) in the label file, and its range is from 0 to 255
		label = tf.cast(label, tf.int32)
		label = label - 1
		image = input_data[0]
		image = tf.cast(tf.expand_dims(image, 0), tf.float32)
		image = image / 255.0

	def _create_network(input_data):
		if self.network_type == 'atrous':
			current_layer = input_batch
			for layer_idx, dilation in enumerate(self.dilations):
				conv = tf.nn.atrous_conv2d(value=current_layer,
										   filters=self.variables['atrous']['filters'][layer_idx],
										   rate=self.dilations[layer_idx],
										   padding='SAME')
				with_bias = tf.nn.bias_add(conv, self.variables['atrous']['biases'][layer_idx])
				if idx == len(self.dilations) - 1:
					current_layer = with_bias
				else:
					current_layer = tf.nn.relu(with_bias)
			return current_layer

	def loss(self,
			 input_data):
		image, label = self._preprocess(input_data)

		output = self._create_network(image)

		# value for the elements in label after preprocess can be:
		#	-1: not care
		#	i (i >= 0): the i-th class (0-based)
		# this is because that the labels parameter for sparse_softmax_cross_entropy_with_logits ranges from [0, num_classes]
		# https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#sparse_softmax_cross_entropy_with_logits
		label_indicator = tf.greater(label, -1)
		effective_label = tf.boolean_mask(tensor=label,
										  mask=label_indicator)
		output = tf.reshape(output, [-1, klass])
		effective_output = tf.boolean_mask(tensor=output,
										   mask=label_indicator)

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=effective_output,
			labels=effective_label)
		reduced_loss = tf.reduce_mean(loss)
