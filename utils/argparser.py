import configargparse
import os

#Commit log
#Added write output option

parser=configargparse.ArgParser(default_config_files=['/home/jbs/scRNA-seq/scRNA_redone/config.ini'])
parser.add('-c','--my-config', is_config_file=True, help='config file path')
parser.add('-m','--model', required=True, choices=['DCA','SCELD'], help='Model choice: DCA or SCELD')
parser.add('--output_path_prefix', required=True, type=str,help='Location of outputfiles with prefix i.e. "./outputdir/outprefix"')
parser.add('--encoder_sizes',type=int, nargs='+', help="Size of encoder layres, note last encoder layer also determines output channels of sequential layer, default=[64, 32]")
parser.add('--decoder_sizes',type=int, nargs='+', help="Size of decoder layers")
parser.add('--dropout_rates',type=float, help='Dropout rates between fully connceted layers')
parser.add('--basset_layers',type=int, help='Number of basset layers to include values between 1-3')
parser.add('--basset_learn',type=int,nargs='+',help='Which basset layers should have learning enables')
parser.add('--final_conv_size',type=int,help='Size of conv layer applied to basset output[10, 46, 194]')
parser.add('--tss_fasta-path', type=str,help='Location of fasta file with tss sequences (of 600bp')
parser.add('--count_adata_path', type=str, help='Location of preprocessed counts in the anndata format')
parser.add('--batch_size',type=int,help='Size of training batches')
parser.add('--epochs',type=int, help='Number of epochs to run')
parser.add('--early_stop',type=int, help='Number of epochs with no improvement before stopping')
parser.add('--learning_rate',type=float, help='Learning rate')
parser.add('--weight_decay',type=float, help='Weight decay')
parser.add('--seed', type=int, default=0, help='Seed for data split for reproducible results, 0 for random')
parser.add('--write_output', type=bool, default=False, help='If model should predict on full data with best model and write output')
args = parser.parse_args()
args = vars(args)

def write_config_from_args(args):
	config_path = f'{args["output_path_prefix"]}_config.ini'
	with open(config_path, 'w') as config_file:
		for key in args:
			print(f'{key}={args[key]}', file=config_file)
	return


if __name__ == '__main__':
	args = argparser.args
