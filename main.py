
from .data_handling import preprocess as pp
from .data_handling.dataset_classes import CountsDataset, SequenceDataset
from .utils import general_utils, argparser
from .models.sc_models import SequenceCountEncoderLinearDecoder, DeepCountAutoencoder
from .training_tools.training import train

#Commit log
#Added write_output

def train_model_run(args_dict):
	device = general_utils.get_device()
	count_data = CountsDataset(args['count_adata_path'])

	if args['model']=='SCELD':
		sequence_data = SequenceDataset(args['tss_fasta_path'])
		sequence_data.reindex_by_list(count_data.adata.var_names.to_list())
	elif args['model']=='DCA':
		sequence_data = False

	##############################Load model#######################
	model_dict = {
    'SCELD': SequenceCountEncoderLinearDecoder,
    'DCA':   DeepCountAutoencoder }
	nn_model = model_dict[args['model']](input_size=count_data.adata.n_vars, seq_len=600, **args)
	nn_model.to(device)
	# ############################Train model ############################
	loss_df = train(model = nn_model, 
		count_data=count_data,
		seq_data=sequence_data,
		output_path_prefix=args['output_path_prefix'],
		batch_size=args['batch_size'],
		epochs=args['epochs'],
		early_stop=args['early_stop'],
		learning_rate=args['learning_rate'],
		weight_decay=args['weight_decay'],
		seed=args['seed'],
		encoder_sizes=args['encoder_sizes'],
		write_output=args['write_output'])

	general_utils.create_loss_graph(loss_df, args['output_path_prefix'])

	return
	

#If running this script as the main script, get the argparse directly
if __name__ == '__main__':
	args = argparser.args
	general_utils.create_dir_from_path(args['output_path_prefix'])
	argparser.write_config_from_args(args)
	train_model_run(args)
	


