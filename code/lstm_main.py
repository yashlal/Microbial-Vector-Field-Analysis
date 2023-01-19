import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from torch.utils.data import Dataset
import math
import collections
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset, ConcatDataset
import matplotlib.cm as cm
import pickle
import data_parsing
import lstm_funcs
import msda

# This is just the main script if you want to run the model one-off or something

# species names and device setup
blastT_labels = ['L. crispatus',
 'L. iners',
 'L. gasseri',
 'L. jensenii',
 'L. other',
 'Gardnerella spp',
 'Streptococcus spp', 
 'Atopobium spp',
 'Prevotella spp',
 'Bergeyella spp',
 'Corynebacterium spp',
 'Finegoldia spp',
 'Sneathia spp',
 'Dialister spp',
 'Other']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

#Loading training data
with open('data/train_test_sequences.pickle','rb') as f:
    all_training_sequences, all_testing_sequences = pickle.load(f)

# Pick testing traj and remove training ones w overlap to avoid overfit
# Also do scaling and tensor stuff (see data_parsing)
testing_traj_ind = 130
batch_size = 32
trainingData2, test_in, tensor_true_test_in, tensor_true_test_data = data_parsing.setup_testing(all_training_sequences, all_testing_sequences, testing_traj_ind, device=device)
msda_data = msda.standard_cutmix(trainingData2)
trainingDataAll = ConcatDataset([trainingData2, msda_data])

ensemble_size = 1 #Number of restarts
num_channels = 15
num_timesteps = 14
train_window = 14

total_tloss = 0
for z in range(ensemble_size):
    
    hparamdict = {'hsize': 20,
   'layers': 1,
   'batch_size': 32,
   'num_epochs': 1000,
   'lr': 0.01,
   'dropout': 0.1}

    hsize = hparamdict['hsize']
    layers = hparamdict['layers']
    LR = hparamdict['lr']
    batch_size = hparamdict['batch_size']
    num_epochs = hparamdict['num_epochs']
    dropout = hparamdict['dropout']
    
    print('Hyperparameters:', hparamdict)
    
    train_dataloader = DataLoader(trainingData2, batch_size=batch_size, shuffle=True)
    best_prediction, bpred_epoch, best_model, test_loss = lstm_funcs.full_train(num_channels,hsize,layers,dropout,batch_size,LR,
                                         num_epochs,train_dataloader,test_in,
                                         tensor_true_test_data, device=device)
    total_tloss += test_loss
    print(f'Run: {z+1}    Test Loss: {test_loss}    Restart: {z+1}    Best Epoch: {bpred_epoch}')
    if z == ensemble_size:
        avg_tloss = total_tloss/(z+1)
        print('Average Test Loss:', avg_tloss)

base_data_color = ['black', 'red', 'orange', 'green', 'blue', 'gold', 'purple', 'pink', 'grey', 'turquoise', 'cyan', 'crimson', 'indigo', 'olive', 'saddlebrown']
pred_color = base_data_color
best_fit_idx = list(range(4))
full_test_data = torch.cat((torch.clone(tensor_true_test_in),torch.clone(tensor_true_test_data)))
lstm_funcs.plot_best_fit(full_test_data, best_prediction, pred_color, base_data_color, best_fit_idx, blastT_labels, False, None, full_test_data)