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
import os
import sys
import pickle
import zipfile

def train_model(model, train_dataloader, train_loss, optimizer, device):

    # Puts model in training mode
    model.train()
    sum_loss = 0

    # Use first batch to get batch size
    batch_size = list(train_dataloader)[0][0].shape[0]
    # Number of items in dataloader
    n_batches = len(train_dataloader)

    # Iter through dataloader
    for i in range(0, n_batches):
        seq,labels = next(iter(train_dataloader))

        # Float the tensors, add CUDA is applicable
        seq=seq.float().to(device)
        labels=labels.float().to(device)

        # Check for correct batch size
        # TO:DO SAVE THE 29 WASTED SEQUENCES!
        if(len(seq)==batch_size):
            # Get num of LSTM layers
            n_layers = model.num_layers

            # Training loop
            model.hidden = (torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(device),
                            torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(device))
            y_pred = model(seq)
            single_loss = train_loss(y_pred, labels)
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()
            sum_loss += single_loss.item()

    avg_loss = sum_loss / n_batches
    return avg_loss

def test_model(model, test_inputs, fut_pred, train_window, device):
    # Testing mode for model
    model.eval()
    new_test_inputs = test_inputs.copy()

    # Get model layers
    n_layers = model.num_layers

    # Predict out by fut_pred timesteps
    for j in range(fut_pred):
        # Shaping
        seq = torch.FloatTensor(new_test_inputs[-train_window:]).to(device)
        seq = seq.reshape(1,len(seq),len(seq[0]))
        
        # Speeds up comp by telling torch to not worry about grads
        with torch.no_grad():
            model.hidden = (torch.zeros(n_layers,1,model.hidden_layer_size).to(device),
                               torch.zeros(n_layers,1,model.hidden_layer_size).to(device))
            nextstep=model(seq).tolist()
            new_test_inputs.append(nextstep[0])

    # Return predicted sequence
    output = new_test_inputs[-fut_pred:]
    return torch.FloatTensor(output)