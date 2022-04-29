import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader
import gc #maybe implement garbage collection
from ..utils import general_utils as utils
from ..models.distributions import NegativeBinomialLoss
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg')
 

def train(model, 
          count_data,
          output_path_prefix,
          encoder_sizes,
          seq_data=False,
          batch_size=128,
          epochs=600,
          early_stop=25,
          learning_rate=0.001,
          weight_decay=5e-4,
          seed=None,
          write_output=False,
          **kwargs):

    z_size=encoder_sizes[-1]
    with open(f'{output_path_prefix}_train.log', 'w') as logfile:
        print(f'train() called: {model.name}, learning rate={learning_rate}, '
            f'weight decay={weight_decay}, batch_size={batch_size}, '
            f'z-size={z_size}, seed={seed}', file=logfile, flush=True)

    epoch_str_wd = len(str(epochs))
    device = utils.get_device()
    #Load in optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = NegativeBinomialLoss()

    ###TEST SPACE###
    #predict = torch.ones(500, 100)
    #target  = torch.zeros(500, 100)
    #sf      = torch.ones(1, 500)
    #print(criterion(mean, dispersion, batch_sf, batch_raw))
    ###TEST SPACE OVER###

    #If model uses sequences load them and unsqueeze to allow convolution
    if seq_data:
        seq_tensor = torch.unsqueeze(seq_data[0:len(seq_data)][0],3)
        seq_tensor = seq_tensor.to(device)
    else:
        seq_tensor = False

    #Split count data into test, train and val datasets and get initial loss 
    test_data, train_data, val_data = count_data.test_train_subset(test_fraction=0.25, val_fraction=0.10, seed=seed)

    train_dl = DataLoader(train_data, batch_size, shuffle=True) 
    test_dl =  DataLoader(test_data, batch_size, shuffle=True) 
    val_dl =  DataLoader(val_data, batch_size, shuffle=True) 
    model.train()
    train_loss_init = calc_loss_no_train(model, train_dl, len(train_data), criterion, optimizer, seq_tensor, device)
    model.eval()
    test_loss_init  = calc_loss_no_train(model, test_dl, len(test_data), criterion, optimizer, seq_tensor, device)
    with open(f'{output_path_prefix}_train.log', 'a') as logfile:
        print(f'Epoch {0:{epoch_str_wd}}/{epochs:{epoch_str_wd}}, train loss: {train_loss_init:>8.2f}, test loss: {test_loss_init:>8.2f}', file=logfile, flush=True)
    
    #Start logging tool and early stop tool
    early_stop_tracker = model_checkpoint(model, early_stop, test_loss_init, output_path_prefix, 0)
    loss_history = loss_tracker(test_loss_init, train_loss_init, epochs)
    
    #Start training
    start_time_sec = time.time()
    for epoch in range(1, epochs+1):
        ### Train and evaluate on training set ###
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            batch_counts = batch[0].to(device)
            batch_sf     = batch[1].to(device)
            batch_raw    = batch[2].to(device)
            batch_sf     = batch_sf.view(batch_sf.shape[0], 1) #Convert size factors to column vector for multiplication

            if seq_data:
                dispersion, mean = model(batch_counts, seq_tensor)  
            else:
                dispersion, mean = model(batch_counts)
            loss = criterion(mean, dispersion, batch_sf, batch_raw)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            train_loss += loss.item()
        train_loss = train_loss / len(train_data)

        ### Evaluate on test dataset
        model.eval()
        test_loss = 0.0

        for batch in test_dl:
            batch_counts = batch[0].to(device)
            batch_sf     = batch[1].to(device)
            batch_raw    = batch[2].to(device)
            batch_sf     = batch_sf.view(batch_sf.shape[0], 1) #Convert size factors to column vector for multiplication
            with torch.no_grad():
                if seq_data:
                    dispersion, mean = model(batch_counts, seq_tensor)  
                else:
                    dispersion, mean = model(batch_counts)
                loss = criterion(mean, dispersion, batch_sf, batch_raw)
                test_loss += loss.item()


        test_loss = test_loss / len(test_data)
        early_stop_now = early_stop_tracker(model, test_loss, epoch)
        loss_history.log_loss(epoch, test_loss, train_loss)
        #If no improvement for #epochs break training loop
        if early_stop_now:
            with open(f'{output_path_prefix}_train.log', 'a') as logfile:
                print(f'Epoch {epoch:{epoch_str_wd}}/{epochs:{epoch_str_wd}}, train loss: {train_loss:>8.2f}, test loss: {test_loss:>8.2f}', file=logfile, flush=True)
            break           

        #Print status to terminal
        if epoch == 1 or epoch % (epochs//20) == 0:
            with open(f'{output_path_prefix}_train.log', 'a') as logfile:
                print(f'Epoch {epoch:{epoch_str_wd}}/{epochs:{epoch_str_wd}}, train loss: {train_loss:>8.2f}, test loss: {test_loss:>8.2f}', file=logfile, flush=True)


    ###End of training
    #Calculate time per epochs
    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epoch

    #Calculate validation error on the best model
    model = torch.load(early_stop_tracker.checkpoint_path, map_location=device)
    val_loss = calc_loss_no_train(model, val_dl, len(val_data), criterion, optimizer, seq_tensor, device)
    #Write last to logfile
    with open(f'{output_path_prefix}_train.log', 'a') as logfile:
        print(f'Time total: {total_time_sec:5.2f} sec',file=logfile,flush=True)
        print(f'Time per epoch: {time_per_epoch_sec:5.2f} sec',file=logfile,flush=True)
        print(f'\nBest model:',file=logfile,flush=True)
        print(f'Epoch {early_stop_tracker.best_epoch:{epoch_str_wd}}/{epochs:{epoch_str_wd}}, test loss: {early_stop_tracker.best_test_loss:>8.2f}', file=logfile, flush=True)
        print(f'Validation loss({len(val_data)} obs): {val_loss:>8.2f}',file=logfile,flush=True)

    #Save the loss history to a csv file
    loss_df = loss_history.save_log(output_path_prefix, epoch)

    #Predict on full dataset and write to output
    if write_output:
        model.eval()
        full_counts = torch.from_numpy(count_data.adata.X).to(device)
        with torch.no_grad():
            if seq_data:
                dispersion, mean = model(full_counts, seq_tensor)  
            else:
                dispersion, mean = model(full_counts)

        predicted_counts = mean
        predicted_counts = predicted_counts.cpu().detach().numpy()
        count_data.adata.layers['predicted_counts'] = predicted_counts
        count_data.adata.uns['dispersion_estimate'] = dispersion.cpu().detach().numpy()

        count_data.write_adata(f'{output_path_prefix}_predict.h5')

    return loss_df

class model_checkpoint():
    #Class to checkpoint best model and early stop if necessary
    def __init__(self, model, early_stop, init_test_loss, output_path_prefix, early_stop_threshold):
        self.max_no_improv      = early_stop
        self.best_test_loss     = init_test_loss
        self.checkpoint_path    = f'{output_path_prefix}_model.pt'
        self.best_epoch         = 0
        self.no_improv_counter  = 0
        self.early_stop_threshold = early_stop_threshold
        torch.save(model, self.checkpoint_path)

    def __call__(self, model, test_loss, epoch):
        if test_loss > (self.best_test_loss - (self.early_stop_threshold*self.best_test_loss)):
            self.no_improv_counter += 1
            if self.no_improv_counter >= self.max_no_improv:
                return True
            else:
                return False
        #If better than best model, save current model and reset counter
        else:
            self.best_test_loss     = test_loss
            self.best_epoch         = epoch 
            self.no_improv_counter  = 0
            torch.save(model, self.checkpoint_path)
            return False

def calc_loss_no_train(model, data_dl, data_len, criterion, optimizer, seq_tensor, device):
    with torch.no_grad():
        running_loss = 0
        for batch in data_dl:
            batch_counts = batch[0].to(device)
            batch_sf     = batch[1].to(device)
            batch_raw    = batch[2].to(device)
            batch_sf     = batch_sf.view(batch_sf.shape[0], 1) #Convert size factors to column vector for multiplication
            if torch.is_tensor(seq_tensor):
                dispersion, mean = model(batch_counts, seq_tensor)  
            else:
                dispersion, mean = model(batch_counts)
            loss = criterion(mean, dispersion, batch_sf, batch_raw)
            running_loss += loss.item()

        return running_loss / data_len
   
class loss_tracker():
    def __init__(self, init_test_loss, init_train_loss, epochs):
        self.loss_df = np.empty([epochs+1,3], dtype=float)
        self.log_loss(0, init_test_loss, init_train_loss)
    
    def log_loss(self, epoch, test_loss, train_loss):
        self.loss_df[epoch,:] = [epoch, test_loss, train_loss]
    
    def save_log(self, output_path_prefix, epoch):
        #Save loss log as csv and remove unused lines
        n_unint = self.loss_df.shape[0]-(epoch+1)
        if n_unint != 0:
            self.loss_df = self.loss_df[:-n_unint]
        self.loss_df = pd.DataFrame(data=self.loss_df, columns=['Epoch', 'Test_loss', 'Train_loss'])
        self.loss_df["Epoch"] = pd.to_numeric(self.loss_df["Epoch"], downcast = 'integer')
        self.loss_df.to_csv(f'{output_path_prefix}_loss.csv', index=False, sep='\t')
        return self.loss_df


