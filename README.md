## scRNA denoising code

This repository contains the code used to create and train the scRNA denoising models described in my masters thesis, and a notebook for preprocessing data used in the project.
The code is for documentation and is non-functional as is. A short description of the files is given below

main.py     -> Main script, constructs and trains specified model
config.ini  -> Config file read by main.py, containing default parameters for model

-training tools/
--training.py -> Script training the model

-Notebooks/
--scRNA_preprocess.ipynb -> Notebook for preprocessing counts, TSS and corrupting/downsampling counts
--notebook_utils.py      -> Script containing misc functions used in notebook

-models/
--sc_models.py      -> Script containing the DCA and SCELD models
--distributions.py  -> Script containing containing the negative binomial distributions and negative log likelihood function of this
--Sequential_pretrained.pth -> Pretrained basset model, used to lift weights to SCELD

-data_handling/
--dataset_classes.py -> Script containing dataclasses used to hold count and TSS data when training

-utils
--argparser.py      -> Script containing all commands used to run the main.py script
--general_utils.py  -> Script with misc functions
