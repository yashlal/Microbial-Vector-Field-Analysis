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
import utils.data_parsing as data_parsing
import pickle
import pdb

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

        #self.hidden = (torch.zeros(self.batch_size,self.num_layers,self.hidden_layer_size),
        #               torch.zeros(self.batch_size,self.num_layers,self.hidden_layer_size))
        self.SM = nn.Sigmoid() #Tanh()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(-1,len(input_seq[0]),len(input_seq[0][0])), self.hidden)
        #pdb.set_trace()
        pred = self.linear(lstm_out.view(-1,len(lstm_out[0]),len(lstm_out[0][0])))[:,-1]
        pred = self.SM(pred)
        return pred

def train1(model, train_dataloader, loss_function1, optimizer):
    model.train()
    sum_loss = 0
    for i in range(0,len(trainingData),batch_size):
        seq,labels = next(iter(train_dataloader))
        seq=seq.float().to(device)
        labels=labels.float().to(device)
        if(len(seq)==batch_size):
            #pdb.set_trace()
            model.hidden = (torch.zeros(layers, batch_size, model.hidden_layer_size).to(device),
                            torch.zeros(layers, batch_size, model.hidden_layer_size).to(device))
            y_pred = model(seq)
            #pdb.set_trace()
            single_loss = loss_function1(y_pred, labels)
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()
            sum_loss += single_loss.item()
    avg_loss = sum_loss/len(trainingData)
    return avg_loss

#add dataloader
def test1(model_in, test_inputs, fut_pred):
    model_in.eval()
    new_test_inputs = test_inputs.copy()
    for j in range(fut_pred): #'fut_pred' for only predictions
        #pdb.set_trace()
        seq = torch.FloatTensor(new_test_inputs[-train_window:]).to(device)
        seq = seq.reshape(1,len(seq),len(seq[0]))
        with torch.no_grad():
            model_in.hidden = (torch.zeros(layers,1,model_in.hidden_layer_size).to(device),
                               torch.zeros(layers,1,model_in.hidden_layer_size).to(device))
            #pdb.set_trace()
            nextstep=model_in(seq).tolist()
            new_test_inputs.append(nextstep[0])
    #print(new_test_inputs[0:train_window]==test_inputs[0:train_window])
    #print("new: ", new_test_inputs[0:train_window])
    #print("old: ", test_inputs)
    output = new_test_inputs[-fut_pred:]
    return torch.FloatTensor(output)

def full_train(num_channels,hsize,layers,dropout,batch_size,LR,num_epochs,train_dataloader,test_in,tensor_true_test_data):
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
            param.data[start:end].fill_(1.) #initialize forget_gate to
        elif 'weight' in name:
            nn.init.orthogonal_(param)

    #starting index
    index = 13
    loss_function1 = nn.BCELoss().to(device)

    for i in range(num_epochs):
        avg_loss = train1(model, train_dataloader, loss_function1, optimizer)
        #avg_loss = train2(model, train_data, optimizer, index)
        if i%25 == 1 or i==num_epochs-1:
            ap = test1(model, test_in_local, fut_pred).view(-1,num_channels).to(device)
            test_loss = loss_function1(ap, tensor_true_test_data)
            print(f'Run:{z}       Epoch:{i}       TestLoss1:{test_loss}      TrainLoss:{avg_loss}')
            if(0):
                plot_init_select(tensor_true_test_data,ap,blastT_labels[index],index)
            #pdb.set_trace()
            if test_loss.item() < best_loss and i>300:
                best_loss = test_loss.item()
                best_i = i
                best_ap = ap.detach().clone()
                #torch.save(model,'best_ap.pt')
                model_clone = LSTM(input_size=num_channels,hidden_layer_size=hsize,num_layers=layers,output_size=num_channels,dropout=dropout,batch_size=batch_size).to(device)
                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))

    best_prediction = best_ap#.tolist()

    del loss_function1, optimizer, model
    torch.cuda.empty_cache()

    return best_prediction, model_clone#,results1#,results2, results3

#Plot real
def plot_init_select(true_data,best_p,label_text,index):
    plt.close()
    fig1, ax1 = plt.subplots()
    title = 'Best Fits'
    ax1.set_title(title)
    ylabel = 'Relative Abundance Run'
    ax1.set_ylabel(ylabel)
    ax1.plot(true_data[:,index].tolist(), label=label_text)
    array_bestp = np.array(best_p.cpu())
    ax1.plot(array_bestp[:,index], ':')
    ax1.legend()
    plt.show(block=False)
    plt.pause(0.5)
    return

def plot_lacto(test_in,true_data,best_prediction,label_text,save_dir):
    test_all = torch.cat((test_in,tensor_true_test_data))
    fig1, ax1 = plt.subplots()
    title = 'Best Fits'
    ax1.set_title(title)
    ylabel = 'Relative Abundance Run'
    ax1.set_ylabel(ylabel)
    ax1.plot(test_all[:,0:3].tolist(), label=label_text[0:3])
    for best_p in best_predictions:
        array_bestp = np.array(best_p.cpu())
        ax1.plot(range(14,28),array_bestp[:,0:3], '--')
    ax1.legend()
    fn = '3_Lacto_species.png'
    path = os.path.join(save_dir,fn)
    plt.savefig(path)
    return

def plot_others(test_in,tensor_true_test_data,best_prediction,blastT_labels,save_dir):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig,axs = plt.subplots(3,4)
    test_all = torch.cat((test_in,tensor_true_test_data))
    blastT_labels_other = blastT_labels[3:]
    for i in range(3):
        for j in range(4):
            axs[i,j].set_title(blastT_labels_other[4*i+j])
            axs[i,j].plot(test_all[:,4*i+j].tolist())
            axs[i,j].plot(range(14,28),best_prediction[:,4*i+j].tolist(), '--')

        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='rel abund')

        for ax in axs.flat:
            ax.label_outer()

    fn = 'Other_species.png'
    path = os.path.join(save_dir,fn)
    plt.savefig(path)
    return

def shap_val(model, train_X, test_X):
    test_model = model
    test_model.eval()
    train_X = train_X.type(torch.FloatTensor).to(device) #For some reason your trainingData ends up as torch.DoubleTensor
    test_model.hidden = (torch.zeros(layers,train_X.shape[0],model.hidden_layer_size).to(device),
                         torch.zeros(layers,train_X.shape[0],model.hidden_layer_size).to(device))

    explainer = shap.DeepExplainer(best_model, train_X)
    test_X = torch.FloatTensor(test_X).view(-1,num_channels).to(device) #Change from list to FloatTensor
    test_X = test_X[None] #Just adding an axis to make it shape [1, 14, 15]
    # print(test_X.shape)

    best_model.hidden = (torch.zeros(layers,train_X.shape[0] * 2,model.hidden_layer_size).to(device),
                         torch.zeros(layers,train_X.shape[0] * 2,model.hidden_layer_size).to(device))
    # "explaining each prediction requires 2 * background dataset size runs", apparently
    test_model.train()
    shap_values = explainer.shap_values(test_X)

    return explainer, shap_values #These should be the 2 things you need to plot stuff

blastT_labels = ['blast_L_crispatus','blast_L_iners', 'blast_L_gasseri', 'blast_L_jensenii', 'blast_L_other', 'blast_Gardnerella_spp', 'blast_Streptococcus_spp', 'blast_Atopobium_spp', 'blast_Prevotella_spp', 'blast_Bergeyella_spp', 'blast_Corynebacterium_spp', 'blast_Finegoldia_spp', 'blast_Sneathia_spp', 'blast_Dialister_spp', 'blast_Other']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train_test_sequences.pickle','rb') as f:
    all_training_sequences, all_testing_sequences = pickle.load(f)

# Pick testing traj and remove training ones w ovrlap to avoid overfit
testing_traj_ind = 0
filtered_training_seqs, testing_seq = data_parsing.setup_testing(all_training_sequences, all_testing_sequences, testing_traj_ind)

print("Training set:", len(filtered_training_seqs), "trajectories")

# remove metadata cols
np_filtered_training_seqs = np.array(filtered_training_seqs)[:, :, 3:]
testing_seq = np.array(testing_seq[:, 3:], dtype=np.float)

# Flatten to 2d array, rescale, unflatten
# Shape is (n_trajectories, n_timepoints, n_features)
np_filtered_training_seqs_flatten = np.reshape(np_filtered_training_seqs, ((len(filtered_training_seqs))*15, 15))
scaler = MinMaxScaler()
np_filtered_training_seqs_flatten = scaler.fit_transform(np_filtered_training_seqs_flatten)
np_filtered_training_seqs = np.reshape(np_filtered_training_seqs_flatten, ((len(filtered_training_seqs)), 15, 15))

# Now create Dataloader
num_traj, num_timesteps, num_inputs, = np_filtered_training_seqs.shape
# last timestep is label data
num_timesteps += -1
tensor_training_data = torch.empty((num_traj, num_timesteps, num_inputs),dtype=float)
tensor_label_data = torch.empty((num_traj, num_inputs),dtype=torch.float)

for i in range(num_traj):
    tensor_training_data[i, :, :] = torch.FloatTensor(np_filtered_training_seqs[i, :-1, :])
    tensor_label_data[i] = torch.FloatTensor(np_filtered_training_seqs[i, -1, :])

print(tensor_training_data.shape)

#Params
hsize=30
layers=3
batch_size=len(filtered_training_seqs)
num_epochs=350
LR=0.001
dropout=0.3
ensemble_size = 1
num_channels = 15
num_timesteps = num_timesteps
train_window = 14
take = '_02'
save_dir = 'lstm_shap_h'+str(hsize)+'_l'+str(layers)+'_b'+str(batch_size)+'_d'+str(dropout)+str(take)

best_predictions = []
shap_results = []
enum = 0
trainingData = tensor_training_data
trainingData2 = TensorDataset(tensor_training_data, tensor_label_data)
train_dataloader = DataLoader(trainingData2, batch_size=batch_size, shuffle=True)
test_in = testing_seq[0:14, :].tolist()
tensor_true_test_in = torch.FloatTensor(testing_seq[0:14, :]).to(device)
tensor_true_test_data = torch.FloatTensor(testing_seq[14:, :]).to(device)

#pdb.set_trace()

for z in range(ensemble_size):
    print(f'H:{hsize} L:{layers} D:{dropout} E:{num_epochs} LR:{LR}')
    best_prediction, best_model = full_train(num_channels,hsize,layers,dropout,batch_size,LR,
                                             num_epochs,train_dataloader,test_in,
                                             tensor_true_test_data)
    best_predictions.append(best_prediction)

    ##run classifier
    ##pass loss to grad decsent

    #test permutations with avg as replacement
    loss_function = nn.BCELoss(reduction='none').to(device)

    #shap
    explainer, shap_vals = shap_val(best_model, trainingData, test_in)
    shap_results.append(shap_vals)
    # pdb.set_trace()

#plot last example only
os.mkdir(save_dir)
plot_lacto(tensor_true_test_in,tensor_true_test_data,best_prediction,blastT_labels,save_dir)
plot_others(tensor_true_test_in,tensor_true_test_data,best_prediction,blastT_labels,save_dir)

#plot shap all
for i in range(len(blastT_labels)):
    shap_array = np.array(shap_results[-1][i][0])
    shap_df = pd.DataFrame(shap_array,columns=blastT_labels)
    figs,axs = plt.subplots()
    axs.set_title(str(blastT_labels[i]))
    axs = sns.heatmap(shap_df,linewidth=0.5)
    plt.tight_layout()
    fn = str(blastT_labels[i]) + '.png'
    path = os.path.join(save_dir,fn)
    plt.savefig(path)
