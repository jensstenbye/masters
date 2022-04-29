# def autoencoder
import torch.nn as nn
import torch
import numpy as np
from . import distributions as distr

#Commit log
#Added mean extract leayer to DCA
#Added some test space to experiment with z*w(1000) -> linear(1000,1000) -> log_mean(1000)

#Produces a convolution block
def get_device():
	
	if torch.cuda.is_available():  
		dev = "cuda:0" 
	else:  
		dev = "cpu"  
	return torch.device(dev) 

class ConvBlock(nn.Module):
	def __init__(self,
				in_channels,
				out_channels,
				conv_height,
				conv_width):
		super(ConvBlock, self).__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, (conv_height, conv_width)),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		)

	def forward(self, x):
		return self.conv_block(x)

	def load_basset_weights(self, basset_layer, freeze):
		#Function for transferring a convolutional layer from basset to our sequential model
		#Load Basset pretrained state dict
		'''Lift directly from kipoi instead of this messed intermediate??'''
		basset_pretrained_dict = torch.load('/faststorage/project/scRNA-seq/scRNA_redone/models/Sequential_pretrained.pth')
		basset_mapping={'1':'model.conv_1.0','2':'model.conv_2.0','3':'model.conv_3.0'}

		#Pull out blocks state dict and update the weights and biases
		'''Lift batchnorm??'''
		model_dict = self.conv_block.state_dict()
		model_dict['0.weight'] = basset_pretrained_dict[f'{basset_mapping[str(basset_layer)]}.weight']
		model_dict['0.bias'] = basset_pretrained_dict[f'{basset_mapping[str(basset_layer)]}.bias']
		self.conv_block.load_state_dict(model_dict)

		#Freeze convolutional layers, keeping batchnorm running
		'''SPØRGSMÅL, FREEZE BATCHNORM'''
		if freeze:
			for name, param in self.conv_block.named_parameters():
					if name[0]=='0':
						param.requires_grad = False 

		return


class MaxPoolBlock(nn.Module):
	def __init__(self,
				maxp_height,
				maxp_width):
		super(MaxPoolBlock, self).__init__()
		self.maxp_block = nn.Sequential(
			nn.MaxPool2d((maxp_height,maxp_width),(maxp_height,maxp_width))
		)

	def forward(self, x):
		return self.maxp_block(x)

class AvgPoolBlock(nn.Module):
	def __init__(self,
				avgp_height,
				avgp_width):
		super(AvgPoolBlock, self).__init__()
		self.avgp_block = nn.Sequential(
			nn.AvgPool2d((maxp_height,maxp_width),(maxp_height,maxp_width))
		)

	def forward(self, x):
		return self.avgp_block(x)

#Produces a fully connected block
class LinearBlock(nn.Module):
	def __init__(self,
				input_len,
				output_len,
				dropout_rate):
		super(LinearBlock, self).__init__()
		block = [nn.Linear(input_len, output_len),
				nn.BatchNorm1d(output_len),
				nn.ReLU()]
		if dropout_rate:
			block.append(nn.Dropout(dropout_rate))
		
		self.linear_block = nn.Sequential(*block)

	def forward(self, x):
		return self.linear_block(x)

class DropoutBlock(nn.Module):
	def __init__(self,
				dropout_rate):
		super(DropoutBlock, self).__init__()
		self.block = nn.Sequential(
				nn.Dropout(dropout_rate)
		)

	def forward(self, x):
		return self.block(x)






######################DCA type autoencoder#########################333
#Autoencoder which fits negative binomial distributions to each gene with a mean from the autoencoder
#And dispersion modeled as a free variable.
class DeepCountAutoencoder(nn.Module):
	def __init__(self, 
				input_size,
				encoder_sizes=[64,32],
				decoder_sizes=[64],
				dropout_rates=0.0,
				**kwargs):
		super(DeepCountAutoencoder, self).__init__()
		'''Not like DCA where it is decoded from latent, do we stand on this?'''
		self.name='DeepCountAutoencoder'
		self.log_dispersion     = torch.nn.Parameter(torch.randn(input_size,    #Define a dispersion parameter to fit
												requires_grad=True, device=get_device()))
		
		#The encoder
		layers=[LinearBlock(input_size,encoder_sizes[0], dropout_rates)]
		for layer in range(1,len(encoder_sizes)):
			layers += [LinearBlock(encoder_sizes[layer-1],encoder_sizes[layer], dropout_rates)]
		#The Decoder
		for layer in range(len(decoder_sizes)):
			layers += [LinearBlock(encoder_sizes[-1],decoder_sizes[layer], dropout_rates)]
		#The mean extractor (log(mean))/output layer
		layers += [nn.Linear(decoder_sizes[-1], input_size)]

		self.net = nn.Sequential(*layers)

	def forward(self, x):
		#Autoencoder outputs predicted log_mean of NB for gene in cell
		log_mean = self.net(x)
		
		dispersion = torch.exp(self.log_dispersion)
		mean = torch.exp(log_mean)

		return dispersion, mean


#SequenceCountEncoderLinearDecoder generates NB means by matrix product of latent variables of each cell, using an encoder
#on count matrix, with the latent state of each gene using a convolutional network
class SequenceCountEncoderLinearDecoder(nn.Module):
	def __init__(self, 
				input_size,
				seq_len=600,
				encoder_sizes=[64, 32],
				dropout_rates=0.0,
				batchnorm=True,
				lambd=0.0, 
				basset_layers=1,
				basset_learn=[],
				final_conv_size=4,
				**kwargs):

		super(SequenceCountEncoderLinearDecoder, self).__init__()   
		self.name='SequenceCountEncoderLinearDecoder'
		#Dictionary with values for the 3 first convolutional layers of basset  
		self.basset_arch={'in_channels':[4,300,200],'out_channels':[300,200,200],
						'conv_height':[19,11,7], 'conv_width':[1,1,1], 
						'maxp_width':[1,1,1],'maxp_height':[3,4,4]}   
		self.basset_layers = basset_layers  
		#We model dispersion as an independent variable   
		self.seq_len = seq_len                           
		self.log_dispersion     = torch.nn.Parameter(torch.clamp(torch.randn(input_size,    #Define a dispersion parameter to fit
												requires_grad=True, device=get_device()), min=1e-4, max=1e4))


		###The count based encoder###
		layers=[LinearBlock(input_size,encoder_sizes[0], dropout_rates)]
		for layer in range(1,len(encoder_sizes)):
			layers += [LinearBlock(encoder_sizes[layer-1],encoder_sizes[layer], dropout_rates)]
		self.count_net = nn.Sequential(*layers)

		###The Sequence based model###
		###First load relevant basset layers###
		layers=[]
		for layer in range(0, basset_layers):
			freeze=False
			if (layer+1) not in basset_learn:
				freeze = True

			layers += [ConvBlock(self.basset_arch['in_channels'][layer], self.basset_arch['out_channels'][layer],
								self.basset_arch['conv_height'][layer], self.basset_arch['conv_width'][layer])]
			layers[-1].load_basset_weights(layer+1, freeze)

			layers += [MaxPoolBlock(self.basset_arch['maxp_height'][layer], self.basset_arch['maxp_width'][layer])]

		#Create final seq-conv-layer
		'''Size determination of this, output of basset 194,46,10 depending on layer'''
		layers += [ConvBlock(self.basset_arch['in_channels'][basset_layers-1], encoder_sizes[-1],
								final_conv_size, 1)]
		#Run a final maxpool across the whole output to reduce each filter to a single value
		self.basset_out_len = self._calc_basset_out()
		seq_out_len = int(self._calc_conv_out(self.basset_out_len, final_conv_size,1))
		layers += [MaxPoolBlock(seq_out_len, 1)]

		self.seq_net = nn.Sequential(*layers)
		

	def forward(self, x, tss):
		#Latent gene weights from sequential module
		#Latent cell weights from count module
		w = torch.squeeze(self.seq_net(tss))
		z = self.count_net(x)
		#Transpose so w fits with z=(cells,k), w=(k, genes) -> z*w=(cells, genes)
		w = torch.transpose(w, 0, 1)   
		
		#Mean predicted directly log(mean) goes out of bounds when exponentiated after matrix product
		mean = torch.mm(z,w)
		dispersion = torch.exp(self.log_dispersion)

		return dispersion, mean
	
	def _calc_basset_out(self):
		#Gives len of output after basset runthrough
		seq_len = self.seq_len
		conv_height = self.basset_arch['conv_height'][0:(self.basset_layers)]
		maxp_height = self.basset_arch['maxp_height'][0:(self.basset_layers)]
		for conv, maxp in zip(conv_height, maxp_height):
			seq_len = self._calc_conv_out(seq_len, conv, 1)
			seq_len = self._calc_conv_out(seq_len, maxp, maxp)
		return seq_len
	
	#Function for calculating outputsize of convolutional layers
	def _calc_conv_out(self, input_width, conv_height, stride):
		return np.floor((input_width-conv_height)/stride)+1

class SequenceCountEncoderLinearDecoderFC(nn.Module):
	def __init__(self, 
				input_size,
				seq_len=600,
				encoder_sizes=[64, 32],
				dropout_rates=0.0,
				batchnorm=True,
				lambd=0.0, 
				basset_layers=1,
				basset_learn=[],
				final_conv_size=4,
				**kwargs):
		super(SequenceCountEncoderLinearDecoderFC, self).__init__()   
		self.name='SequenceCountEncoderLinearDecoderFC'
		#Dictionary with values for the 3 first convolutional layers of basset  
		self.basset_arch={'in_channels':[4,300,200],'out_channels':[300,200,200],
						'conv_height':[19,11,7], 'conv_width':[1,1,1], 
						'maxp_width':[1,1,1],'maxp_height':[3,4,4]}   
		self.basset_layers = basset_layers  
		#We model dispersion as an independent variable   
		self.seq_len = seq_len                           
		self.log_dispersion     = torch.nn.Parameter(torch.clamp(torch.randn(input_size,    #Define a dispersion parameter to fit
												requires_grad=True, device=get_device()), min=1e-4, max=1e4))


		###The count based encoder###
		layers=[LinearBlock(input_size,encoder_sizes[0], dropout_rates)]
		for layer in range(1,len(encoder_sizes)):
			layers += [LinearBlock(encoder_sizes[layer-1],encoder_sizes[layer], dropout_rates)]
		self.count_net = nn.Sequential(*layers)

		###The Sequence based model###
		###First load relevant basset layers###
		layers=[]
		for layer in range(0, basset_layers):
			freeze=False
			if (layer+1) not in basset_learn:
				freeze = True

			layers += [ConvBlock(self.basset_arch['in_channels'][layer], self.basset_arch['out_channels'][layer],
								self.basset_arch['conv_height'][layer], self.basset_arch['conv_width'][layer])]
			layers[-1].load_basset_weights(layer+1, freeze)

			layers += [MaxPoolBlock(self.basset_arch['maxp_height'][layer], self.basset_arch['maxp_width'][layer])]

		#Create final seq-conv-layer
		'''Size determination of this, output of basset 194,46,10 depending on layer'''
		layers += [ConvBlock(self.basset_arch['in_channels'][basset_layers-1], 50,
								final_conv_size, 1)]
		#Insert a convolutional layer to reduce input to FC layer
		self.basset_out_len = self._calc_basset_out()
		seq_out_len = int(self._calc_conv_out(self.basset_out_len, final_conv_size,1)) * 50
		layers += [nn.Flatten()]
		layers += [LinearBlock(seq_out_len,encoder_sizes[-1], dropout_rates)]

		self.seq_net = nn.Sequential(*layers)

		###TEST SPACE###
		#Apply linear layer to each 
		#self.log_mean = nn.Linear(980, 980)

	def forward(self, x, tss):
		#Latent gene weights from sequential module
		#Latent cell weights from count module
		w = torch.squeeze(self.seq_net(tss))


		z = self.count_net(x)
		#Transpose so w fits with z=(cells,k), w=(k, genes) -> z*w=(cells, genes)
		w = torch.transpose(w, 0, 1)  

		##TEST SPACE
		#latent_full = torch.mm(z,w)
		#log_mean = self.log_mean(latent_full)
		#mean = torch.exp(log_mean)

		
		
		#Mean predicted directly log(mean) goes out of bounds when exponentiated after matrix product
		mean = torch.mm(z,w)
		dispersion = torch.exp(self.log_dispersion)

		return dispersion, mean

	def _calc_basset_out(self):
		#Gives len of output after basset runthrough
		seq_len = self.seq_len
		conv_height = self.basset_arch['conv_height'][0:(self.basset_layers)]
		maxp_height = self.basset_arch['maxp_height'][0:(self.basset_layers)]
		for conv, maxp in zip(conv_height, maxp_height):
			seq_len = self._calc_conv_out(seq_len, conv, 1)
			seq_len = self._calc_conv_out(seq_len, maxp, maxp)
		return seq_len
	def _calc_conv_out(self, input_width, conv_height, stride):
		return np.floor((input_width-conv_height)/stride)+1

#############################Test space#############################################
class SequenceCountEncoderLinearDecoderAVG(nn.Module):
	def __init__(self, 
				input_size,
				seq_len=600,
				encoder_sizes=[64, 32],
				dropout_rates=0.0,
				batchnorm=True,
				lambd=0.0, 
				basset_layers=1,
				basset_learn=[],
				final_conv_size=4,
				**kwargs):

		super(SequenceCountEncoderLinearDecoderAVG, self).__init__()   
		self.name='SequenceCountEncoderLinearDecoder'
		#Dictionary with values for the 3 first convolutional layers of basset  
		self.basset_arch={'in_channels':[4,300,200],'out_channels':[300,200,200],
						'conv_height':[19,11,7], 'conv_width':[1,1,1], 
						'maxp_width':[1,1,1],'maxp_height':[3,4,4]}   
		self.basset_layers = basset_layers  
		#We model dispersion as an independent variable   
		self.seq_len = seq_len                           
		self.log_dispersion     = torch.nn.Parameter(torch.clamp(torch.randn(input_size,    #Define a dispersion parameter to fit
												requires_grad=True, device=get_device()), min=1e-4, max=1e4))


		###The count based encoder###
		layers=[LinearBlock(input_size,encoder_sizes[0], dropout_rates)]
		for layer in range(1,len(encoder_sizes)):
			layers += [LinearBlock(encoder_sizes[layer-1],encoder_sizes[layer], dropout_rates)]
		self.count_net = nn.Sequential(*layers)

		###The Sequence based model###
		###First load relevant basset layers###
		layers=[]
		for layer in range(0, basset_layers):
			freeze=False
			if (layer+1) not in basset_learn:
				freeze = True

			layers += [ConvBlock(self.basset_arch['in_channels'][layer], self.basset_arch['out_channels'][layer],
								self.basset_arch['conv_height'][layer], self.basset_arch['conv_width'][layer])]
			layers[-1].load_basset_weights(layer+1, freeze)

			layers += [MaxPoolBlock(self.basset_arch['maxp_height'][layer], self.basset_arch['maxp_width'][layer])]

		#Create final seq-conv-layer
		'''Size determination of this, output of basset 194,46,10 depending on layer'''
		layers += [ConvBlock(self.basset_arch['in_channels'][basset_layers-1], encoder_sizes[-1],
								final_conv_size, 1)]
		#Run a final maxpool across the whole output to reduce each filter to a single value
		self.basset_out_len = self._calc_basset_out()
		seq_out_len = int(self._calc_conv_out(self.basset_out_len, final_conv_size,1))
		layers += [AvgPoolBlock(seq_out_len, 1)]

		self.seq_net = nn.Sequential(*layers)
		

	def forward(self, x, tss):
		#Latent gene weights from sequential module
		#Latent cell weights from count module
		w = torch.squeeze(self.seq_net(tss))
		z = self.count_net(x)
		#Transpose so w fits with z=(cells,k), w=(k, genes) -> z*w=(cells, genes)
		w = torch.transpose(w, 0, 1)   
		
		#Mean predicted directly log(mean) goes out of bounds when exponentiated after matrix product
		mean = torch.exp(torch.mm(z,w))
		dispersion = torch.exp(self.log_dispersion)

		return dispersion, mean
	
	def _calc_basset_out(self):
		#Gives len of output after basset runthrough
		seq_len = self.seq_len
		conv_height = self.basset_arch['conv_height'][0:(self.basset_layers)]
		maxp_height = self.basset_arch['maxp_height'][0:(self.basset_layers)]
		for conv, maxp in zip(conv_height, maxp_height):
			seq_len = self._calc_conv_out(seq_len, conv, 1)
			seq_len = self._calc_conv_out(seq_len, maxp, maxp)
		return seq_len
	
	#Function for calculating outputsize of convolutional layers
	def _calc_conv_out(self, input_width, conv_height, stride):
		return np.floor((input_width-conv_height)/stride)+1


if __name__ == '__main__':
	#criterion = NegativeBinomialLoss
	DCA_net	  = DeepCountAutoencoder(input_size=500)
	SCELD_net = SequenceCountEncoderLinearDecoder(input_size=500, basset_layers=3, basset_learn=[1])

