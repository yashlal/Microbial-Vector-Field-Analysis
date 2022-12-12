import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from torch.utils.data import Dataset
import math
import pdb
import random
import collections
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import shap
import os
from torch.utils.data import DataLoader,TensorDataset
import pickle

def create_inout_sequences3(iners_col, tw, p_labels):
    train_inout_seq = []
    train_indices = []
    for i in range(len(iners_col)-tw):
        in_l = iners_col[i:i+tw]
        out_val = iners_col[i+tw:i+tw+1][0]

        check_l = p_labels[i:i+tw+1]
        if all([check_l[i]==check_l[0] for i in range(len(check_l))]):
            if all([not(math.isnan(k1[0])) for k1 in in_l]) and all([not(math.isnan(k2)) for k2 in out_val]):
                train_inout_seq.append((in_l, out_val))
                train_indices.append(i+tw)

    test_seqs = []
    test_indices = []
    for j in range(len(iners_col)-2*tw):
        iners_l2 = iners_col[j:j+2*tw]
        check_l2 = p_labels[j:j+2*tw]
        if all([not(math.isnan(k3[0])) for k3 in iners_l2]) and all([check_l2[q]==check_l2[0] for q in range(len(check_l2))]):
            test_seq = (iners_col[j:j+tw], iners_col[j+tw:j+2*tw])
            test_seqs.append(test_seq)
            test_indices.append(j+tw)

    return train_inout_seq, test_seqs, (train_indices, test_indices)

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=1, dropout=0.5, batch_size=1, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.SM = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(-1,len(input_seq[0]),len(input_seq[0][0])), self.hidden)
        pred = self.linear(lstm_out.view(-1,len(lstm_out[0]),len(lstm_out[0][0])))[:,-1]
        pred = self.SM(pred)
        return pred

def train_model(model, train_dataloader, loss_function1, optimizer):
    model.train()
    sum_loss = 0
    for i in range(0,len(trainingData),batch_size):
        seq,labels = next(iter(train_dataloader))
        seq=seq.float().to(device)
        labels=labels.float().to(device)
        if(len(seq)==batch_size):
            model.hidden = (torch.zeros(layers, batch_size, model.hidden_layer_size).to(device),
                            torch.zeros(layers, batch_size, model.hidden_layer_size).to(device))
            y_pred = model(seq)
            single_loss = loss_function1(y_pred, labels)
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()
            sum_loss += single_loss.item()
    avg_loss = sum_loss/len(trainingData)
    return avg_loss

def test_model(model_in, test_inputs, fut_pred):
    model_in.eval()
    new_test_inputs = test_inputs.copy()
    for j in range(fut_pred): #'fut_pred' for only predictions
        seq = torch.FloatTensor(new_test_inputs[-train_window:]).to(device)
        seq = seq.reshape(1,len(seq),len(seq[0]))
        with torch.no_grad():
            model_in.hidden = (torch.zeros(layers,1,model_in.hidden_layer_size).to(device),
                               torch.zeros(layers,1,model_in.hidden_layer_size).to(device))
            nextstep=model_in(seq).tolist()
            new_test_inputs.append(nextstep[0])

    output = new_test_inputs[-fut_pred:]
    return torch.FloatTensor(output)

def full_train(num_channels,hsize,layers,dropout,batch_size,LR,num_epochs,train_dataloader,test_in,tensor_true_test_data):
    print(f'hsize:{hsize}       layers:{layers}       num_epochs:{num_epochs}       dropout:{dropout}      batch_size:{batch_size}       lr:{LR}')
    tensor_test_in = torch.FloatTensor(test_in).view(-1,num_channels).to(device)
    test_in_local = copy.deepcopy(test_in)
    tensor_test_in_local = torch.FloatTensor(test_in_local).view(-1,num_channels).to(device)
    model = LSTM(input_size=num_channels,hidden_layer_size=hsize,num_layers=layers,output_size=num_channels,dropout=dropout,batch_size=batch_size).to(device)
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    optimizer.state = collections.defaultdict(dict)
    fut_pred = len(tensor_true_test_data)
    best_loss = 9999.9

    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param,0.0)
            start = int(len(param)/4)
            end = int(len(param)/2)
            param.data[start:end].fill_(1.)
        elif 'weight' in name:
            nn.init.orthogonal_(param)

    loss_function1 = nn.BCELoss().to(device)

    for i in range(num_epochs):
        avg_loss = train_model(model, train_dataloader, loss_function1, optimizer)
        if i%25 == 1 or i==num_epochs-1:
            ap = test_model(model, test_in_local, fut_pred).view(-1,num_channels).to(device)
            test_loss = loss_function1(ap, tensor_true_test_data)
            #print(f'Run:{z}       Epoch:{i}       TestLoss1:{test_loss}      TrainLoss:{avg_loss}')
            if test_loss.item() < best_loss and i>300:
                best_loss = test_loss.item()
                best_i = i
                best_ap = ap.detach().clone()
                model_clone = LSTM(input_size=num_channels,hidden_layer_size=hsize,num_layers=layers,output_size=num_channels,dropout=dropout,batch_size=batch_size).to(device)
                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))
    #if (i+1) == num_epochs:
    #    print(f'Run{z+1}       Epoch:{i+1}       TestLoss1:{best_loss}      TrainLoss:{avg_loss}')
    best_prediction = best_ap#.tolist()

    del loss_function1, optimizer, model
    torch.cuda.empty_cache()

    return best_prediction, best_i, model_clone, best_loss#,results1#,results2, results3

def shap_val(model, train_X, test_X):
    #For our use case: test_X is expected to be None because the training data is the one being explained
    #train_X should be a tensor, DoubleTensor or FloatTensor are both fine, of the training data of shape (#samples, #timesteps, #features)
    #model is the model
    #test_X should be None (unless explaining a different data or a full training data subset, then it should be of shape (#samples, #timesteps, #features) as a double or float tensor)
    test_model = model.eval()
    train_X = train_X.type(torch.FloatTensor).to(device) #Ensure data type is float tensor
    test_model.hidden = (torch.zeros(layers,train_X.shape[0],model.hidden_layer_size).to(device),
                         torch.zeros(layers,train_X.shape[0],model.hidden_layer_size).to(device))

    explainer = shap.DeepExplainer(test_model, train_X)
    if test_X is not None:
        test_X = test_X.type(torch.FloatTensor).to(device)

        test_model.hidden = (torch.zeros(layers,train_X.shape[0] * 2,model.hidden_layer_size).to(device),
                            torch.zeros(layers,train_X.shape[0] * 2,model.hidden_layer_size).to(device))
        # explaining each prediction requires 2 * background dataset size runs
        test_model.train()
        shap_values = explainer.shap_values(test_X)

    else:
        test_model.hidden = (torch.zeros(layers,train_X.shape[0] * 2,model.hidden_layer_size).to(device),
                            torch.zeros(layers,train_X.shape[0] * 2,model.hidden_layer_size).to(device))
        test_model.train()
        shap_values = explainer.shap_values(train_X)
    return explainer, shap_values #These should be the 2 things you need to plot anything, however, for this code only shap_values will be used

def setup_testing(training_seqs, testing_seqs, testing_traj_index):
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

    return filtered_training_seqs, testing_seq

blastT_labels = ['blast_L_crispatus','blast_L_iners', 'blast_L_gasseri', 'blast_L_jensenii', 'blast_L_other', 'blast_Gardnerella_spp', 'blast_Streptococcus_spp', 'blast_Atopobium_spp', 'blast_Prevotella_spp', 'blast_Bergeyella_spp', 'blast_Corynebacterium_spp', 'blast_Finegoldia_spp', 'blast_Sneathia_spp', 'blast_Dialister_spp', 'blast_Other']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
with open('train_test_sequences.pickle','rb') as f:
    all_training_sequences, all_testing_sequences = pickle.load(f)

# Pick testing traj and remove training ones w ovrlap to avoid overfit
testing_traj_ind = 0
filtered_training_seqs, testing_seq = setup_testing(all_training_sequences, all_testing_sequences, testing_traj_ind)

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

ensemble_size = 1 #Number of restarts, but best model will be the best model of the last restart to happen
num_channels = 15
num_timesteps = num_timesteps
train_window = 14
take = '_01'
#save_dir = 'lstm_shap_h'+str(hsize)+'_l'+str(layers)+'_b'+str(batch_size)+'_d'+str(dropout)+str(take)

best_predictions = []
shap_results = []
enum = 0
trainingData = tensor_training_data
trainingData2 = TensorDataset(tensor_training_data, tensor_label_data)

test_in = testing_seq[0:14, :].tolist()
tensor_true_test_in = torch.FloatTensor(testing_seq[0:14, :]).to(device)
tensor_true_test_data = torch.FloatTensor(testing_seq[14:, :]).to(device)

total_tloss = 0
finalbest_result_list = []
for z in range(ensemble_size):

    temp1_list = []
    temp2_list = []
    hparamdict = {'hsize': 51,
   'layers': 3,
   'batch_size': 32,
   'num_epochs': 700,
   'lr': 0.0001,
   'dropout': 0.6}
    # #print(f'H:{hsize} L:{layers} D:{dropout} E:{num_epochs} LR:{LR}')
    hsize = hparamdict['hsize']
    layers = hparamdict['layers']
    LR = hparamdict['lr']
    batch_size = hparamdict['batch_size']
    num_epochs = hparamdict['num_epochs']
    dropout = hparamdict['dropout']
    print(hparamdict)

    temp2_list.append(hparamdict)
    #for w in range(1):
    train_dataloader = DataLoader(trainingData2, batch_size=batch_size, shuffle=True)
    best_prediction, bpred_epoch, best_model, test_loss = full_train(num_channels,hsize,layers,dropout,batch_size,LR,
                                         num_epochs,train_dataloader,test_in,
                                         tensor_true_test_data)
    total_tloss += test_loss
    temp1_list.append(best_prediction)
    temp1_list.append(bpred_epoch)
    print(f'Run: {z+1}    Test Loss: {test_loss}    Restart: {z+1}    Best Epoch: {bpred_epoch}')
    avg_tloss = total_tloss/(z+1)
    temp2_list.append(avg_tloss)
    temp2_list.append(temp1_list)
    finalbest_result_list.append(temp2_list)
    print('Average Test Loss:', avg_tloss)

copy_model = copy.deepcopy(best_model.eval()) #best_model can be replaced by any model
print(trainingData.type())
explainer, shap_values = shap_val(copy_model, trainingData, None) #Gets the shap values of all training examples to show general relations the model learned
all_shap_val = np.array(shap_values) #The basic shap values that are needed is something of shape (#output_features(15), #data samples, #time steps(14), #input_features(15))
#Warning: unrecognized nn.Module: LSTM will be stated repeatedly, however, it will run

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm


def top_shap_feature_selection(all_shap_values, num_exp_features, feature_output_idx):
    assert str(type(all_shap_values)) == "<class 'list'>" or str(type(all_shap_values)) == "<class 'numpy.ndarray'>"

    if type(all_shap_values) == list:
        all_shap_values = np.array(all_shap_values)

    shap_shape = all_shap_values.shape

    all_shap_values = np.mean(np.abs(all_shap_val)[feature_output_idx], axis=0)
    idx = np.argsort(all_shap_values.flatten())[::-1][0:int(num_exp_features)]
    flat_shap = all_shap_values.flatten()
    idxT_step = (np.divide(idx + 1, shap_shape[-1])) - 1
    idxFeature = idx % shap_shape[-1]

    return idx.astype('int64'), idxT_step.astype('int64'), idxFeature.astype('int64')

def budget_beeswarm(all_shap_values, input_data, num_exp_features, labels, feature_output_idx, path):
    #all_shap_values should be a ndarray or list of ndarrays of shape (#outputs, #samples, #timesteps, #input features)
    #input_data expects a numpy array of the inputs to the model of shape (#samples, #timesteps, #input features)
    #labels is a list of the labels, ex. blastT_labels
    #num_exp_features is an integer, the number of output features wanted to explain the desired output feature (using the num_exp_features with sorting by absolute value average shap value score)
    #feature_output_idx is the index of the desired feature output from the outputs of the model, in this case 0 to 14 for each species
    #path is the folder saving path, if it is None, then it is not saved
    assert str(type(all_shap_values)) == "<class 'list'>" or str(type(all_shap_values)) == "<class 'numpy.ndarray'>"
    assert str(type(input_data)) == "<class 'numpy.ndarray'>"

    if type(all_shap_values) == list:
        all_shap_values = np.array(all_shap_values)

    shap_shape = all_shap_values.shape
    num_total_values = int(shap_shape[0] * shap_shape[1] * shap_shape[2] * shap_shape[3])
    fig, ax = plt.subplots()

    assert len(shap_shape) == 4
    assert shap_shape[2] == input_data.shape[1]

    idx, idxT_step, idx_Feature = top_shap_feature_selection(all_shap_values, num_exp_features, feature_output_idx)

    #Shap Values to be plotted on x axis
    rx = all_shap_values[feature_output_idx, :, idxT_step, idx_Feature].flatten()
    cx = input_data[:, idxT_step, idx_Feature].flatten()

    #Color bar min, max, feature data to color by, and red to blue coloring setup
    rvmin = np.min(input_data)
    rvmax = np.max(input_data)
    #cx = np.tile(flat_input_data, int(rx.shape[0]/flat_input_data.shape[0]))

    #cm = mcol.LinearSegmentedColormap.from_list("BlueRed",["b","w", "r"])
    vcenter = np.mean(cx[idx])
    #print(vcenter)

    #Potentially fix with a better solution, this is only a patch for cx[idx] == [0., 0., 0., 0., ...]
    if vcenter <= rvmin:
        vcenter = 0.5
        rvmin = 0.0
    normalize = mcol.TwoSlopeNorm(vcenter=vcenter, vmin=rvmin, vmax=rvmax)
    colormap = cm.coolwarm_r

    #y axis timestep plotting
    #timestepY = np.array(range(shap_shape[2])) + 1
    #ry = np.reshape(np.tile(timestepY, int(num_total_values/(shap_shape[2]*shap_shape[0]))), (-1, 15, 14))
    #ry = ry[:, idxT_step, idx_Feature].flatten()
    timestepY = np.array(range(num_exp_features)) + 1

    ry = np.tile(timestepY, int(shap_shape[1])).flatten()

    #Plotting

    sc = plt.scatter(rx, ry, c=cx, vmin=rvmin, vmax=rvmax, s=20, cmap=colormap)#, cmap=cm
    pts = ax.collections[0]
    pts.set_offsets(pts.get_offsets() + np.c_[np.zeros(cx.shape[0]), np.random.uniform(-.3, .3, cx.shape[0])])
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(cx)
    fig.colorbar(scalarmappaple)
    #cbar = plt.colorbar(sc)
    plt.ylabel('Top 10 Impact Features')
    plt.xlabel('Shap Values')

    #features = np.char.add(np.char.add('Time Step ', idxT_step.astype('str')), np.char.add(' Species ', idx_Feature.astype('str')))
    features = np.char.add(np.char.add('Time Step: ', (idxT_step + 1).astype('str')), np.char.add(' | Species: ', np.array(labels)[idx_Feature.astype('int64')]))

    ax.set_yticks(list(range(1, num_exp_features+1)))
    ax.set_yticklabels(features)
    plt.rcParams['figure.figsize'] = [20, 10]
    fig.set_label('Feature Value')#cbar.set_label
    plt.title('Explaining ' + str(labels[feature_output_idx]))
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

for idx in range(15):
    #path = '/content/drive/MyDrive/Colab_Data/Visual_' + str(blastT_labels[idx]) + '.png'
    budget_beeswarm(all_shap_val, trainingData.numpy(), 10, blastT_labels, idx, None)
