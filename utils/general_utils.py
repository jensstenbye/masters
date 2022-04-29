import torch
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def get_device():
	
	if torch.cuda.is_available():  
		dev = "cuda:0" 
	else:  
		dev = "cpu"  
	return torch.device(dev)  

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gpu_mem_sum(device):
	if device.type == 'cuda':
		print(torch.cuda.get_device_name(0))
		print('Memory Usage:')
		print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,2), 'GB')
		print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,2), 'GB')
	return

def create_dir_from_path(path):
	clean_path = os.path.dirname(path) #Remove any trailing string
	if not os.path.exists(clean_path):
		os.makedirs(clean_path, exist_ok=True)
	return

def create_loss_graph(loss_df, output_path_prefix):
	train_loss = loss_df['Train_loss']
	test_loss = loss_df['Test_loss']
	epochs = loss_df['Epoch']

	plt.plot(epochs, train_loss, 'b', label='Training loss')
	plt.plot(epochs, test_loss, 'r', label='Test loss')
	plt.ylim(ymin=0)
	plt.title('Training and Test loss')
	plt.legend()
	plt.savefig(f'{output_path_prefix}_graph.png',bbox_inches='tight')
	plt.close()

	return