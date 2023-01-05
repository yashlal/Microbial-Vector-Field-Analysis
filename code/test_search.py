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
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.cm as cm
import pickle
import data_parsing
import lstm_funcs
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Loading training data
with open('data/train_test_sequences.pickle','rb') as f:
    all_training_sequences, all_testing_sequences = pickle.load(f)

def test_wrapper(ind):
    # Pick testing traj and remove training ones w overlap to avoid overfit
    # Also do scaling and tensor stuff (see data_parsing)
    testing_traj_ind = ind
    batch_size = 32
    trainingData2, test_in, tensor_true_test_in, tensor_true_test_data = data_parsing.setup_testing(all_training_sequences, all_testing_sequences, testing_traj_ind, device=device)

    ensemble_size = 2 #Number of restarts
    num_channels = 15
    big_arr = []
    for z in range(ensemble_size):
        print((z+1, ind))
        hparamdict = {'hsize': 51,
        'layers': 3,
        'batch_size': 32,
        'num_epochs': 1000,
        'lr': 0.0001,
        'dropout': 0.6}

        hsize = hparamdict['hsize']
        layers = hparamdict['layers']
        LR = hparamdict['lr']
        batch_size = hparamdict['batch_size']
        num_epochs = hparamdict['num_epochs']
        dropout = hparamdict['dropout']
        
        print('Hyperparameters:', hparamdict)
        
        train_dataloader = DataLoader(trainingData2, batch_size=batch_size, shuffle=True)
        loss_arr = lstm_funcs.all_loss_train(num_channels,hsize,layers,dropout,batch_size,LR,
                                            num_epochs,train_dataloader,test_in,
                                            tensor_true_test_data, device=device, return_best=False)

        big_arr.append((z+1, ind, loss_arr))
    with open(f'test_saves/{ind}.pickle') as f:
        pickle.dump(big_arr, f)
        print(f"Saved {ind}")

if __name__=='__main__':
    with multiprocessing.get_context('spawn').Pool() as pool:
        pool.map(test_wrapper, range(0,187))