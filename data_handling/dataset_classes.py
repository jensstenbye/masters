from torch.utils.data import Dataset, DataLoader, random_split
import torch
import scanpy as sc
import sys
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from Bio.SeqIO.FastaIO import SimpleFastaParser

#Commit log
#Added write_adata to countsDataset

#This dataset loads scRNA counts from a h5 file and adds utility for:
# 1. Fetching observations with associated size factors and raw counts
# 2. Normalizing, scaling and log1p counts to optimize training
# 3. Splitting the data into a fraction for training and a fraction for testing
class CountsDataset(Dataset):
	def __init__(self, counts_path):
		self.adata = sc.read(counts_path,
							cache=True)

	def __len__(self):
		#Return length of data
		return self.adata.n_obs
	def __getitem__(self, idx):
	 	#Return processed-counts, size-factor and raw-counts
		return self.adata.X[idx], self.adata.obs['size_factors'][idx], self.adata.layers['counts'][idx] 
	
	def test_train_subset(self, test_fraction, val_fraction=0, seed=0):
		#Returns 2 or 3 pytorch subsets of size of given fractions
		if not (val_fraction):
			test_n, train_n = int(np.floor(test_fraction*self.__len__())), int(np.ceil((1-test_fraction)*self.__len__()))
			if seed:
				return random_split(self, [test_n,train_n], generator=torch.Generator().manual_seed(seed))
			else:
				return random_split(self, [test_n,train_n])
		else:
			test_n, train_n = int(np.floor(test_fraction*self.__len__())), int(np.ceil((1-test_fraction-val_fraction)*self.__len__()))
			val_n = self.__len__() - test_n - train_n
			if seed:
				return random_split(self, [test_n, train_n, val_n], generator=torch.Generator().manual_seed(seed))
			else:
				return random_split(self, [test_n, train_n, val_n])
	
	def write_adata(self, path):
		self.adata.write(path,as_dense='X')

	

#This dataset loads tss sequences from a fastq file and add utility for
# 1: One-hot encoding of the sequence
'''Sort encoder and CNN genes in same order?'''
class SequenceDataset(Dataset):
	def __init__(self, fasta_path):
		#Read data
		self.fasta = self.__fetch_fasta_to_df(fasta_path)
		self.mapping = dict((base,int_map) for int_map, base in enumerate(['A','C','G','T']))

	def __len__(self):
		#Returns number of genes/sequences in dataset
		return len(self.fasta)

	def __getitem__(self, idx):
		#Returns onehot representation of sequence and name
		if isinstance(idx, slice):
			#If slice initialize tensor of known length to fill
			seq_len=len(self.fasta.iloc[0]['Sequence'])
			slice_len = idx.stop-idx.start
			onehot_slice=torch.empty(size=(slice_len, 4, seq_len))
			name_slice=[]

			for seq_idx in range(0, slice_len):
				sequence = self.fasta.iloc[seq_idx+idx.start]['Sequence']

				#print('sequence', sequence[0:10], seq_idx, self.fasta.iloc[seq_idx+idx.start].name)
				name_slice.append(self.fasta.iloc[seq_idx+idx.start].name)
				onehot_slice[seq_idx,:,:] = self.__seq_to_onehot_tensor(sequence)
			return onehot_slice, name_slice

		else:
			sequence = self.fasta.iloc[idx]['Sequence']
			name = self.fasta.iloc[idx].name
			one_hot = self.__seq_to_onehot_tensor(sequence)
			return one_hot, name

	def __seq_to_onehot_tensor(self, sequence):
		#Returns onehot encoded representation of sequence dim(4xSeq)
		test = torch.tensor([self.mapping[base] for base in sequence])
		one_hot = torch.nn.functional.one_hot(
			test,
			num_classes=len(self.mapping),
		)
		return torch.transpose(one_hot, 0,1)

	def __fetch_fasta_to_df(self, fasta_path):
		with open(fasta_path, 'r') as fasta_file:
			identifiers = []
			sequences = []
			for title, sequence in SimpleFastaParser(fasta_file):					
				identifiers.append(title.split(None, 1)[0])  # First word is ID
				sequences.append(sequence)
		return pd.DataFrame(sequences, index = identifiers, columns=['Sequence'])	

	def test_train_subset(self, test_fraction):
		#Returns 2 pytorch subsets of size test_pct*len and (1-test_pct)*len
		test_n, train_n = int(np.floor(test_fraction*self.__len__())), int(np.ceil((1-test_fraction)*self.__len__()))
		return random_split(self, [test_n,train_n])
	
	def reindex_by_list(self, gene_id_list):
		self.fasta = self.fasta.reindex(gene_id_list)
		


if __name__=='__main__':
	count_data=CountsDataset('/home/jbs/scRNA-seq/steps/preprocessed/redone/PBMC_win600.h5')
	full_dl = DataLoader(count_data, len(count_data), shuffle=False)
	count_data.write_adata('/home/jbs/scRNA-seq/steps/test_write_adata.h5')
	count_data=CountsDataset('/home/jbs/scRNA-seq/steps/test_write_adata.h5')
	#test_subset, train_subset = count_data.test_train_subset(0.25)
	#tss = SequenceDataset('/home/jbs/scRNA-seq/steps/preprocessed/redone/seq_win600.fa')




