import tensorflow as tf

class SegModel(object):
	def __init__(self,
				 batch_size,
				 dilations):
		self.batch_size = batch_size
		sefl.dilations = _create_dilation_layers

		self.variables = self._create_variables()

	def _create_variables(self):
		var = dict()
		for i, dilation in enumerate(self.dilations):
			current_filter = _create_variables


	def _create_dilation_layer(self, input_batch, layer_index, dilation):
		pass

	def _create_network(input_batch):
		current_layer = input_batch
		for layer_index, dilation in enumerate(self.dilations):
			output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation)

	def loss(self,
			 input_batch):
		pass
