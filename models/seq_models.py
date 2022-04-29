import torch.nn as nn
import math
from collections import OrderedDict
'''THIS SCRIPT IS NOT USED FOR GENERATING MAIN MODELS BUT FOR A CLEANER PRETRAINED MODEL TO LIFT THE '''
#Seqential model 
class sequential_model(nn.Module):
	def __init__(self, 
				input_size = int,         #Length of input sequence
				output_channels = int,  #Number of cell types to predict
				conv_sizes = list,      #Sizes of the convolutional layers
				conv_widths = list,
				max_pool_widths = list, #Width of the max pooling of above layers
				fc_sizes = list,        #Sizes of the fully connected layers
				 dropout=list):         #Dropout of the above layers

		super(sequential_model, self).__init__()

		self.input_size		= input_size
		self.conv_widths	= conv_widths
		self.max_pool_widths= max_pool_widths
		self.conv_sizes		= conv_sizes
		self.in_channels	= [4, *conv_sizes] #Input channels = 4<-one hot encoded
		model_list = []

		#The convolutional layers
		for layer in range(1,len(self.in_channels)):
			model_list.append(('conv_{}'.format(layer),
								conv_block(in_channels 		= self.in_channels[layer-1],
											out_channels 	= self.in_channels[layer],
											conv_width 		= conv_widths[layer-1],
											max_pool_width 	= max_pool_widths[layer-1])))

		#The Flatten layer
		#Reshape conv output to fit fc_input. 600 input -> 200x10 output -> 2000 flattened
		model_list.append(('flatten', LambdaLayer(lambda x: x.view(x.size(0), -1))))

		#The fully connected layers
		fc_inputs = [self.flatten_size(), *fc_sizes]
		for layer_idx in range(1,len(fc_inputs)):
			model_list.append(('fc_{}'.format(layer_idx),
								fc_block(in_size = fc_inputs[layer_idx-1],
								out_size = fc_inputs[layer_idx],
								dropout = dropout[layer_idx-1])))

		#Final layer, to be removed after loaded weights
		model_list.append(('cell_specific_prediction', 
							nn.Sequential(nn.Linear(fc_inputs[-1], output_channels),
											nn.Sigmoid())))

		#Convert model list to model
		self.model = nn.Sequential(OrderedDict(model_list))

	#Forward pass
	def forward(self, x):
		return self.model(x)

	#Function needed for getting input size to fc_layers 
	def flatten_size(self):
		size = self.input_size
		for idx in range(len(self.conv_widths)):
			size = math.floor((size-self.conv_widths[idx])/1)+1
			size = math.floor((size-self.max_pool_widths[idx])/self.max_pool_widths[idx])+1
		return size*self.conv_sizes[-1]
	
#Produces a convolution block
def conv_block(in_channels, out_channels, conv_width, max_pool_width):
	conv_list = [nn.Conv2d(in_channels, out_channels, (conv_width,1)),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(), 
				nn.MaxPool2d((max_pool_width,1),(max_pool_width,1))] #Max pool kernel, max pool stride
	return nn.Sequential(*conv_list)

#Produces a fully connected block
def fc_block(in_size, out_size, dropout):
	fc_list = [nn.Linear(in_size, out_size),
				nn.BatchNorm1d(out_size, 1e-05, 0.1, True),
				nn.ReLU(),
				nn.Dropout(dropout)]
	return nn.Sequential(*fc_list)

#Lambda layer function defined, to include flatten in sequential
class LambdaLayer(nn.Module):
	def __init__(self, lambda_fn):
		super(LambdaLayer, self).__init__()
		self.lambda_fn = lambda_fn 
	def forward(self, x):
		return self.lambda_fn(x)
