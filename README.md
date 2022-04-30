## scRNA denoising code

This repository contains the code used to create and train the scRNA denoising models described in my masters thesis, and a notebook for preprocessing data used in the project. The conda environment used for this project can be reconstructed from 'masters_env.txt' with the correct package versions.
The code is for documentation purposes only and is non-functional as is. A short description of the files is given below.

main.py     -> Main script, constructs and trains specified model<br>
config.ini  -> Config file read by main.py, containing default parameters for model

-training tools/<br>
--training.py -> Script training the model  

-notebooks/<br>
--scRNA_preprocess.ipynb -> Notebook for preprocessing counts, TSS and corrupting/downsampling counts<br>
--notebook_utils.py      -> Script containing misc functions used in notebook

-models/<br>
--sc_models.py      -> Script containing the DCA and SCELD models<br>
--distributions.py  -> Script containing containing the negative binomial distributions its NLL<br>
--Sequential_pretrained.pth -> Pretrained basset model, used to lift weights to SCELD

-data_handling/<br>
--dataset_classes.py -> Script containing dataclasses used to hold count and TSS data when training

-utils<br>
--argparser.py      -> Script containing all commands used to run the main.py script<br>
--general_utils.py  -> Script with misc functions
