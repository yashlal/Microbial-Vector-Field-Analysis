import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.cm as cm
import pickle
import data_parsing
import lstm_funcs
import torch.multiprocessing as mp
import random
import itertools
from tqdm import tqdm

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
    total_ensemble_loss = 0
    en = 0
    for test_ind in test_inds:
        en += 1
        print(f'{config}:{en}')
        testing_traj_ind = test_ind
        batch_size = 32
        trainingData2, test_in, tensor_true_test_in, tensor_true_test_data = data_parsing.setup_testing(all_training_sequences, all_testing_sequences, testing_traj_ind, device=device)

        train_dataloader = DataLoader(trainingData2, batch_size=batch_size, shuffle=True)
        best_epoch, ensemble_indiv_loss = lstm_funcs.all_loss_train(num_channels=15, hsize=hidden_layer_size, layers=num_layers, dropout=dropout, batch_size=batch_size, LR=LR, num_epochs=1000, train_dataloader=train_dataloader, test_in=test_in, tensor_true_test_data=tensor_true_test_data, device=device, return_best=True)
        total_ensemble_loss += ensemble_indiv_loss
    
    avg_ensemble_loss = total_ensemble_loss / n_ensembles
    return (config, avg_ensemble_loss)

if __name__=='__main__':
    with mp.get_context('spawn').Pool() as pool:
        l = pool.map(hp_wrapper, search_space)
    with open('hp_save.pickle', 'wb') as f:
        pickle.dump(l ,f)
    