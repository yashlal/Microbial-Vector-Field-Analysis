import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pdb

# Step 1: read the datafile
# We need metadata for later (removing sequences with large time gaps, interpolation, prevent crossing patients)
# only run once
# Results alr saved in np_data.pickle
def read_data(filename, pickle_name):
    all_data = pd.read_excel(filename)
    blast_data_and_metadata = all_data.iloc[:,1:19]
    np_data_and_metadata = np.array(blast_data_and_metadata)

    with open(pickle_name,'wb') as f:
        pickle.dump(np_data_and_metadata, f)

# Step 2: Data modifications
# Clear rows of all nan values (some weird data error)
# Clear duplicate rows
# Average in 1 blank gaps
# Again, only run once and then just store in binary
def data_modifications(data_array):
    new_data_array = data_array.copy()
    # remove all rows with nan values
    # isnan only works on floats and ints so we first take those values and then check for nans
    row_f = lambda r: not (np.isnan([el for el in r if (type(el)==float or type(el)==int)])).any()
    new_data_array = np.array(list(filter(row_f, new_data_array)))

    # remove duplicated rows (no idea where these came from in the dataset)
    nonduplicated_row_indices = []
    for ind in range(len(new_data_array)-1):
        if (not (all(new_data_array[ind, 0:3]==new_data_array[ind+1, 0:3]))):
            nonduplicated_row_indices.append(ind)
    # extra check on last row
    if (not (all(new_data_array[-1, 0:3]==new_data_array[-2, 0:3]))):
        nonduplicated_row_indices.append(-1)
    new_data_array = new_data_array[nonduplicated_row_indices, :]

    # we iterate through all but the last row because we access the following row in each loop and otherwise we will get index error
    insertion_indices_array = [] #where to insert rows
    insertion_values_array = [] #actual rows to insert
    for ind in range(len(new_data_array)-1):
        # check rows for same metadata(year and patient number) and if rows differ by a 1 gap
        if ((new_data_array[ind, 0:2]==new_data_array[ind+1, 0:2]).all()) and (new_data_array[ind, 2]==new_data_array[ind+1, 2]-2):

            insertion_indices_array.append(ind+1)
            
            insertion_values_array_el = new_data_array[ind, 0:2].tolist()
            insertion_values_array_el += [int(new_data_array[ind, 2]+1)]
            averaged_bacterial_values = ((new_data_array[ind, 3:]+new_data_array[ind+1, 3:])/2).tolist()
            insertion_values_array_el += averaged_bacterial_values
            insertion_values_array.append(insertion_values_array_el)

    # insert averaged rows
    new_data_array = np.insert(new_data_array, insertion_indices_array, insertion_values_array, 0)
    return new_data_array

# Step 3: Making training sequences out of our data
# Make sequences of 15 (14 in, 1 out) continguous time points (with same metadata: patient number and year)
# Make sequences of 28 (14 ,14) contigous time points (with same metadata) for testing
def generate_training_and_testing_sequences(data_array):
    training_seqs = []

    for ind in range(len(data_array)-14):
        seq = data_array[ind:ind+15, :]
        # metadata check (same patient # and year)
        if all([x==seq[0, 0] for x in seq[:, 0]]) and (all([x==seq[0, 1] for x in seq[:, 1]])):
            #print(seq[:,0], seq[0,0], all([x==seq[0, 0]] for x in seq[:, 0]))
            # check for contigous sequence
            if (seq[0, 2] == (seq[-1, 2]-14)):
                training_seqs.append(seq)

    testing_seqs = []

    for ind2 in range(len(data_array)-27):
        seq = data_array[ind2:ind2+28, :]
        # metadata check (same patient # and year)
        if all([x==seq[0, 0] for x in seq[:, 0]]) and (all([x==seq[0, 1] for x in seq[:, 1]])):
            x = seq[0,2]
            y = seq[-1,2]
            # check for contigous sequence
            if (seq[0, 2] == (seq[-1, 2]-27)):
                testing_seqs.append(seq)

    return training_seqs, testing_seqs

# Step 4: Pick testing sequence and throw out training sequences with overlap
# Only function to actually run on each run of NN (will be run in the lstm_main.py file)
# Also applies MinMax scaler and sets up data in tensorized format
def setup_testing(training_seqs, testing_seqs, testing_traj_index, device):
    testing_seq = testing_seqs[testing_traj_index]
    test_IDs = []
    for i in range(len(testing_seq)):
        temp = ''.join(testing_seq[i,0:2])
        test_IDs.append(temp+str(testing_seq[i,2]))
    filtered_training_seqs = []
    for training_seq in training_seqs:
        flag = True
        train_IDs = []
        for i in range(len(training_seq)):
            temp = ''.join(training_seq[i,0:2])
            train_IDs.append(temp+str(training_seq[i,2]))
        overlap = list(set(train_IDs).intersection(set(test_IDs)))
        if len(overlap)==0:
            filtered_training_seqs.append(training_seq)

    # return filtered_training_seqs, testing_seq

    print("Training set:", len(filtered_training_seqs), "trajectories")

    # remove metadata cols
    np_filtered_training_seqs = np.array(filtered_training_seqs)[:, :, 3:]
    testing_seq = np.array(testing_seq[:, 3:], dtype=float)

    # Flatten to 2d array, rescale, unflatten
    # Shape is (n_trajectories, n_timepoints, n_features)
    np_filtered_training_seqs_flatten = np.reshape(np_filtered_training_seqs, ((len(filtered_training_seqs))*15, 15))
    scaler = MinMaxScaler()
    np_filtered_training_seqs_flatten = scaler.fit_transform(np_filtered_training_seqs_flatten)
    np_filtered_training_seqs = np.reshape(np_filtered_training_seqs_flatten, ((len(filtered_training_seqs)), 15, 15))# Now create Dataloader
    num_traj, num_timesteps, num_inputs, = np_filtered_training_seqs.shape
    # last timestep is label data
    num_timesteps += -1
    tensor_training_data = torch.empty((num_traj, num_timesteps, num_inputs),dtype=float)
    tensor_label_data = torch.empty((num_traj, num_inputs),dtype=torch.float)

    for i in range(num_traj):
        tensor_training_data[i, :, :] = torch.FloatTensor(np_filtered_training_seqs[i, :-1, :])
        tensor_label_data[i] = torch.FloatTensor(np_filtered_training_seqs[i, -1, :])

    print('Training Size', tensor_training_data.shape)
    print('Training Label Size:', tensor_label_data.shape)

    # Make dataloader and testing tensors
    trainingData = TensorDataset(tensor_training_data, tensor_label_data)
    test_in = testing_seq[0:14, :].tolist()
    tensor_true_test_in = torch.FloatTensor(testing_seq[0:14, :]).to(device)
    tensor_true_test_data = torch.FloatTensor(testing_seq[14:, :]).to(device)

    return trainingData, test_in, tensor_true_test_in, tensor_true_test_data

if __name__=='__main__':
    # # running first function for data reading
    # read_data("VMBData_clean.xlsx", 'np_data.pickle')

    # # running second function for data modifs
    # with open('np_data.pickle', 'rb') as f:
    #     np_data_and_metadata = pickle.load(f)
    # np_data_and_metadata = data_modifications(np_data_and_metadata)
    
    # with open('np_data_modified.pickle','wb') as f:
    #     pickle.dump(np_data_and_metadata, f)

    # # running third function for training seqs
    # with open('np_data_modified.pickle', 'rb') as f:
    #     np_data_and_metadata = pickle.load(f)
    # all_training_sequences, all_testing_sequences = generate_training_and_testing_sequences(np_data_and_metadata)
    # p = [all_training_sequences, all_testing_sequences]
    
    # with open('train_test_sequences.pickle','wb') as f:
    #     pickle.dump(p, f)
    pass
