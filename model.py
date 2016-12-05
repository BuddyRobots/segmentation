import tensorflow as tf

def create_variable(name, shape):
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	# initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
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
		if self.network_type == "deconv":
			self.strides = strides
		self.kernel_size = kernel_size
		channels.insert(0, self.input_channel)
		channels.append(self.klass)
		self.channels = channels
		self.variables = self._create_variables()

	def _create_variables(self):
		var = dict()

		if self.network_type == "atrous":
			var['atrous'] = dict()
			var['atrous']['filters'] = list()
			for i, dilation in enumerate(self.dilations):
				var['atrous']['filters'].append(create_variable('atrous_filter',
																[self.kernel_size[i],
																 self.kernel_size[i],
																 self.channels[i],
																 self.channels[i + 1]]))
			var['atrous']['biases'] = list()
			for i, channel in enumerate(self.channels):
				if i == 0:
					continue
				var['atrous']['biases'].append(create_bias_variable('atrous_bias', [channel]))
		if self.network_type == "deconv":
			var['deconv'] = dict()
			var['deconv']['filters'] = list()
			for i, stride in enumerate(self.strides):
				if stride > 0:
					var['deconv']['filters'].append(create_variable('deconv_filter',
																	[self.kernel_size[i],
																	 self.kernel_size[i],
																	 self.channels[i],
																	 self.channels[i + 1]]))
				else:
					var['deconv']['filters'].append(create_variable('deconv_filter',
																	[self.kernel_size[i],
																	 self.kernel_size[i],
																	 self.channels[i + 1],
																	 self.channels[i]]))
			var['deconv']['biases'] = list()
			for i, channel in enumerate(self.channels):
				if i == 0:
					continue;
				var['deconv']['biases'].append(create_bias_variable('deconv_bias', [channel]))
		return var

	def _preprocess(self, input_data, generate=False):
		if generate == True:
			image = input_data
			image = tf.cast(tf.expand_dims(image, 0), tf.float32)
			label = None
		else:
			image = input_data[0]
			image = tf.cast(tf.expand_dims(image, 0), tf.float32)
			label = input_data[1]
			label = tf.reshape(label, [-1])
			# value for the elements in label before preprocess can be:
			#	0: not care
			#	i (i > 0): the i-th class (1-based)
			# this is because each element is saved as one byte (unsigned 8-bit int) in the label file, and its range is from 0 to 255
			label = tf.cast(label, tf.int32)
			label = label - 1
		# tf.nn.conv2d(padding='SAME') always pads 0 to the input tensor,
		# thus make the value of the white pixels in the image 0
		image = 1.0 - image / 255.0
		if self.network_type == 'deconv':
			# for deconv networks, the width and the height of the input tensor must be divisable by the downsampling scale
			self.scale = 1
			for stride in self.strides:
				if stride > 1:
					self.scale = self.scale * stride
			self.h = tf.shape(image)[1]
			self.w = tf.shape(image)[2]
			self.h_pad = (self.scale - tf.shape(image)[1] % self.scale) % self.scale
			self.w_pad = (self.scale - tf.shape(image)[2] % self.scale) % self.scale
			image = tf.pad(image, [[0, 0], [self.h_pad, 0], [self.w_pad, 0], [0, 0]])
		return image, label

	def _create_network(self, input_data):
		if self.network_type == 'atrous':
			current_layer = input_data
			for layer_idx, dilation in enumerate(self.dilations):
				if dilation == 1:
					conv = tf.nn.conv2d(input=current_layer,
										filter=self.variables['atrous']['filters'][layer_idx],
										strides=[1,1,1,1],
										padding='SAME')
				else:
					conv = tf.nn.atrous_conv2d(value=current_layer,
											   filters=self.variables['atrous']['filters'][layer_idx],
											   rate=dilation,
											   padding='SAME')
				with_bias = tf.nn.bias_add(conv, self.variables['atrous']['biases'][layer_idx])
				if layer_idx == len(self.dilations) - 1:
					current_layer = with_bias
				else:
					current_layer = tf.nn.relu(with_bias)
			retval = tf.identity(current_layer, name="NETWORK_OUTPUT")
			return current_layer

		if self.network_type == 'deconv':
			current_layer = input_data
			for layer_idx, stride in enumerate(self.strides):
				if stride > 0:
					conv = tf.nn.conv2d(input=current_layer,
										filter=self.variables['deconv']['filters'][layer_idx],
										strides=[1, stride, stride, 1],
										padding='SAME')
				else:
					current_shape = tf.shape(current_layer)
					output_shape = tf.concat(0, [tf.mul(tf.slice(current_shape, [0], [3]), tf.constant([1, -stride, -stride], tf.int32)),
												 tf.constant([self.channels[layer_idx + 1]], tf.int32)])
					strides = tf.constant([1, -stride, -stride, 1], tf.int32)
					conv = tf.nn.conv2d_transpose(value=current_layer,
												  filter=self.variables['deconv']['filters'][layer_idx],
												  output_shape=output_shape,
												  strides=[1, -stride, -stride, 1],
												  padding='SAME')
				with_bias = tf.nn.bias_add(conv, self.variables['deconv']['biases'][layer_idx])
				if layer_idx == len(self.strides) - 1:
					current_layer = with_bias
				else:
					current_layer = tf.nn.relu(with_bias)
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
		if self.network_type == 'deconv':
			# For deconv network, in the preprocess procedure, the image
			# is padded with zero to make the image size divisable by the downsampling
			# scale. The padded zero should be sliced before calculating loss
			# with labels
			output = tf.slice(output, [0, self.h_pad, self.w_pad, 0], [1, self.h, self.w, self.klass])
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
		print("AAAAAAAAAAAAAAAAA")
		print(image.name)
		image, _ = self._preprocess(input_data=image,
									generate=True)
		output = self._create_network(image)
		output_image = tf.argmax(input=output,
								 dimension=3)
		print("AAAAAAAAAAAAAAAAA")
		print(output_image.name)
		return output_image
