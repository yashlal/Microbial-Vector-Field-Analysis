import torch
import pickle
import data_parsing
import random
import numpy as np
import scipy.stats as sps
from torch.utils.data import DataLoader,TensorDataset

# File for data augm strategies

# Generate a mask of the form {0,1}^n similar to cutmix but not in a continuous segment
# First selects a lambda in U(0,1) then uses that in a bernoulli way to generate the mask
# Returns the new data with masks applied
def custom_cutmix(data):
    inds_ = list(range(len(data)))
    random.shuffle(inds_)

    tensor_training_data = torch.empty((len(data), data[0][0].shape[0], data[0][0].shape[1]),dtype=float)
    tensor_label_data = torch.empty((len(data), data[0][0].shape[1]),dtype=float)

    for i in range(len(data)):
        x = torch.zeros(data[0][0].shape)
        # select lambda
        l = np.random.random()
        # generate mask
        r = sps.bernoulli.rvs(l, size=14)
        # weighting for labels
        w = sum(r) / len(r)

        inds = [i for i in range(len(r)) if r[i]==1]
        
        x[inds, :] = 1
        d1, l1 = data[i]
        d2, l2 = data[inds_[i]]
        # hadamard product to apply mask
        d1_mod = torch.mul(d1, x)
        d2_mod = torch.mul(d2, 1-x)
        # new data and label
        new_d = d1_mod+d2_mod
        new_l = (w*l1)+((1-w)*l2)
        tensor_training_data[i,:,:] = torch.FloatTensor(new_d.float())
        tensor_label_data[i, :] = (torch.FloatTensor(new_l.float()))

    return TensorDataset(tensor_training_data, tensor_label_data)

# Generate a mask of the form {0,1}^n similar to cutmix but with out discrete data
# First generates a start point for the mask uniformly
# then generates an end point for the mask unfiormly
# Return the new data with masks applied
def standard_cutmix(data):
    inds_ = list(range(len(data)))
    random.shuffle(inds_)

    tensor_training_data = torch.empty((len(data), data[0][0].shape[0], data[0][0].shape[1]),dtype=float)
    tensor_label_data = torch.empty((len(data), data[0][0].shape[1]),dtype=float)

    for i in range(len(data)):
        x = torch.zeros(data[0][0].shape)
        # Generate start and end for mask
        start_ind = np.random.randint(0, x.shape[0]-1)
        end_ind = np.random.randint(start_ind, x.shape[0])+1

        x[start_ind:end_ind, :] = 1
        w = (end_ind-start_ind) / (x.shape[0])

        d1, l1 = data[i]
        d2, l2 = data[inds_[i]]

        # hadamard product to apply mask
        d1_mod = torch.mul(d1, x)
        d2_mod = torch.mul(d2, 1-x)
        # new data and label
        new_d = d1_mod+d2_mod
        new_l = (w*l1)+((1-w)*l2)
        tensor_training_data[i,:,:] = torch.FloatTensor(new_d.float())
        tensor_label_data[i, :] = (torch.FloatTensor(new_l.float()))

    return TensorDataset(tensor_training_data, tensor_label_data)

# Interpolation with weighted avg using w in [0,1)
def mixup(data):
    inds_ = list(range(len(data)))
    random.shuffle(inds_)

    tensor_training_data = torch.empty((len(data), data[0][0].shape[0], data[0][0].shape[1]),dtype=float)
    tensor_label_data = torch.empty((len(data), data[0][0].shape[1]),dtype=float)

    for i in range(len(data)):
        # Generate lambda in [0,1)
        w = np.random.random()

        d1, l1 = data[i]
        d2, l2 = data[inds_[i]]

        new_d = (w*d1) + ((1-w)*d2) 
        new_l = (w*l1) + ((1-w)*l2) 

        tensor_training_data[i,:,:] = torch.FloatTensor(new_d.float())
        tensor_label_data[i, :] = torch.FloatTensor(new_l.float())

    return TensorDataset(tensor_training_data, tensor_label_data)    
