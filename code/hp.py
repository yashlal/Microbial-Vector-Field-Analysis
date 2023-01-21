import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset, ConcatDataset
import matplotlib.cm as cm
import pickle
import data_parsing
import lstm_funcs
import torch.multiprocessing as mp
import random
import itertools
from tqdm import tqdm
import msda

# This file is for the HP grid search we ran (batch size was accidentally frozen at 32)
# I use this when doing runs, currently it is coded for experimenting w MSDA
# The commented out section was for the parallelizing and pickle saving

# seeded 10 test trajs for reprod.
random.seed(0)
test_inds = list(sorted(random.sample(range(187), 10)))
print(test_inds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading training data
with open('data/train_test_sequences.pickle','rb') as f:
    all_training_sequences, all_testing_sequences = pickle.load(f)

search_space = list(itertools.product([5,10,20,50,100],[1,2,3,5],[0.1, 0.3, 0.5, 0.7], [64, 32, 16, 4, 1], [0.01, 0.001, 0.0001]))

def hp_wrapper(config):
    hidden_layer_size = config[0]
    num_layers = config[1]
    dropout = config[2]
    batch_size = config[3]
    LR = config[4]

    n_ensembles = len(test_inds)
    all_ensemble_loss = []
    en = 0
    for test_ind in test_inds:
        en += 1
        print(f'{config}:{en}/10')
        testing_traj_ind = test_ind
        batch_size = 64
        trainingData2, test_in, tensor_true_test_in, tensor_true_test_data = data_parsing.setup_testing(all_training_sequences, all_testing_sequences, testing_traj_ind, device=device)
        # Does the desired MSDA startegy and makes the big dataloader
        train_plus_msda = [trainingData2]
        # 4 can be changed to wtvr u want and so can which MSDA function u use
        for i in range(9):
            train_plus_msda.append(msda.custom_cutmix(trainingData2))

        allTensorData = ConcatDataset(train_plus_msda)

        train_dataloader = DataLoader(allTensorData, batch_size=batch_size, shuffle=True)
        best_epoch, ensemble_indiv_loss = lstm_funcs.all_loss_train(num_channels=15, hsize=hidden_layer_size, layers=num_layers, dropout=dropout, batch_size=batch_size, LR=LR, num_epochs=1000, train_dataloader=train_dataloader, test_in=test_in, tensor_true_test_data=tensor_true_test_data, device=device, return_best=True)
        all_ensemble_loss.append(ensemble_indiv_loss)
        print((test_ind, ensemble_indiv_loss))
    
    # The array has all 10 loss values from the ensemble and then the avg at the last position
    avg_ensemble_loss = sum(all_ensemble_loss) / en
    all_ensemble_loss.append(avg_ensemble_loss)
    return(all_ensemble_loss)

    # pickle_obj = (config, avg_ensemble_loss)
    # f_path = f"hp_saves/{hidden_layer_size}_{num_layers}_{dropout}_{batch_size}_{LR}.pickle"
    # with open(f_path, 'wb') as f:
    #     pickle.dump(pickle_obj, f)

if __name__=='__main__':
    # mp.set_start_method('spawn')
    # with mp.Pool(processes=12) as pool:
    #     pool.map(hp_wrapper, search_space[0:12])
    one_config = [20, 1, 0.1, 32, 0.01]
    print(hp_wrapper(one_config))
    