import tensorflow as tf

def create_variable(name, shape):
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	# initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	variable = tf.Variable(initializer(shape=shape), name=name)
	return variable

def create_bias_variable(name, shape):
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	return tf.Variable(initializer(shape=shape), name)

def batch_norm(x, n_out, phase_train):
	"""
	Batch normalization on convolutional maps.
	Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
	Args:
		x:           Tensor, 4D BHWD input maps
		n_out:       integer, depth of input maps
		phase_train: boolean tf.Varialbe, true indicates training phase
		scope:       string, variable scope
	Return:
		normed:      batch-normalized maps
	"""
	with tf.variable_scope('bn'):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.9)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train,
							mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed

class SegModel(object):
	def __init__(self,
				 input_channel,
				 klass,
				 batch_size,
				 kernel_size,
				 dilations,
				 channels,
				 with_bn,
				 phase_train):
		self.input_channel = input_channel
		self.klass = klass
		self.batch_size = batch_size
		self.dilations = dilations
		self.kernel_size = kernel_size
		channels.insert(0, self.input_channel)
		channels.append(self.klass)
		self.channels = channels
		self.with_bn = with_bn
		if self.with_bn == True:
			self.phase_train = tf.Variable(phase_train)
		else:
			self.phase_train = phase_train
		self.variables = self._create_variables()

	def _create_variables(self):
		var = dict()

		var['filters'] = list()
		for i, dilation in enumerate(self.dilations):
			var['filters'].append(create_variable('filter',
												  [self.kernel_size[i],
												  self.kernel_size[i],
												  self.channels[i],
												  self.channels[i + 1]]))
		var['biases'] = list()
		for i, channel in enumerate(self.channels):
			if i == 0:
				continue;
			var['biases'].append(create_bias_variable('bias', [channel]))
		return var

	def _preprocess(self, input_data, generate=False):
		if generate == True:
			image = input_data
			image = tf.cast(tf.expand_dims(image, 0), tf.float32)
			label = None
		else:
			image = input_data[0]
			label = input_data[1]
			label = tf.reshape(label, [-1])
			# if shuffle batch is used in reader.py, there is no need to expand dimensions
			if self.batch_size == 1:
				image = tf.expand_dims(image, 0)
			image = tf.cast(image, tf.float32)
			# value for the elements in label before preprocess can be:
			#	0: not care
			#	i (i > 0): the i-th class (1-based)
			# this is because each element is saved as one byte (unsigned 8-bit int) in the label file, and its range is from 0 to 255
			label = tf.cast(label, tf.int32)
			label = label - 1
		# tf.nn.conv2d(padding='SAME') always pads 0 to the input tensor,
		# thus make the value of the white pixels in the image 0
		image = 1.0 - image / 255.0
		return image, label

	def _create_network(self, input_data):
		current_layer = input_data
		for layer_idx, dilation in enumerate(self.dilations):
			if dilation > 1:
				current_shape = tf.shape(current_layer)
				current_layer = tf.transpose(current_layer, perm=[1, 2, 3, 0])
				current_layer = tf.reshape(tensor=current_layer,
										   shape=[224 / dilation,
								  				  224 / dilation,
								  				  current_shape[3],
								  				  self.batch_size * dilation * dilation])
				current_layer = tf.transpose(current_layer, perm=[3, 0, 1, 2])
			conv = tf.nn.conv2d(input=current_layer,
								filter=self.variables['filters'][layer_idx],
								strides=[1, 1, 1, 1],
								padding='SAME')
			if dilation > 1:
				current_shape = tf.shape(conv)
				conv = tf.transpose(conv, perm=[1, 2, 3, 0])
				conv = tf.reshape(tensor=conv,
								  shape=[224,
								  		 224,
								  		 current_shape[3],
								  		 self.batch_size])
				conv = tf.transpose(conv, perm=[3, 0, 1, 2])

			# the bias is unnecessary when batch normalization is adopted
			if self.with_bn:
				# the first element in the channels array is the input channel size,
				conv = batch_norm(conv, self.channels[layer_idx + 1], self.phase_train)
			else:
				conv = tf.nn.bias_add(conv, self.variables['biases'][layer_idx])
			# the output layer has no nonlinearity
			if layer_idx == len(self.dilations) - 1:
				current_layer = conv
			else:
				current_layer = tf.nn.relu(conv)
		self.probe = self.variables['biases'][9]
		current_layer = tf.identity(current_layer, name="NETWORK_OUTPUT")
		return current_layer

	def loss(self, input_data):
		image, label = self._preprocess(input_data)

		image = tf.identity(image, name="NETWORK_INPUT")

		output = self._create_network(image)

		# value for the elements in label after preprocess:
		#	-1: not care
		#	i (i >= 0): the i-th class (0-based)
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

	def generate(self, image):
		image, _ = self._preprocess(input_data=image,
									generate=True)
		output = self._create_network(image)
		output_image = tf.argmax(input=output,
								 dimension=3)
		return output_image
