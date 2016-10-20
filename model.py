import tensorflow as tf

class SegModel(object):
	def __init__(self,
				 input_channel,
				 klass,
				 batch_size,
				 dilations):
		self.input_channel = input_channel
		self.klass = klass
		self.batch_size = batch_size
		sefl.dilations = dilations
		self.variables = self._create_variables()

	def _create_variables(self):
		var = dict()
		for i, dilation in enumerate(self.dilations):
			current_filter = _create_variables

	def _create_dilation_layer(self, input_batch, layer_index, dilation):
		pass

	def _create_network(input_data):
		f = tf.Variable(initializer(shape=[3, 3, input_channel, klass]),
									name=name)
		current_layer = tf.nn.conv2d(input=input_data,
									 filter=f,
									 strides=[1, 1, 1, 1],
									 padding='SAME')
		return current_layer
		# current_layer = input_batch
		# for layer_index, dilation in enumerate(self.dilations):
		# 	output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation)

	def loss(self,
			 input_data):
		image = input_data[0]
		label = input_data[1]
		image = tf.cast(tf.expand_dims(image, 0), tf.float32)

		output = self._create_network(image)

		label = tf.reshape(label, [-1])
		# the definition of elements in label:
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

		effective_label = tf.cast(effective_label, tf.int32)
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=effective_output,
			labels=effective_label)
		reduced_loss = tf.reduce_mean(loss)
