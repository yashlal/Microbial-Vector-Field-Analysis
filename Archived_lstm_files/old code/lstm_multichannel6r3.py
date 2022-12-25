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

#needs to be made multidimensional - channel 0 = predictor
def create_inout_sequences2(all_data, tw, p_labels):
    train_inout_seq = []
    train_indices = []
    for i in range(len(all_data[0])-tw):
        out_val = all_data[0][i+tw:i+tw+1]
        check_l = p_labels[i:i+tw+1]
        in_l = []
        for j in range(i,i+tw):
            timeslice = []
            for k in range(len(all_data)):
                timeslice.append(all_data[k][j])
                #print(i,j,k,all_data[k][j],timeslice)
            in_l.append(timeslice)
            
        if all([check_l[i]==check_l[0] for i in range(len(check_l))]):
            if all([k1>=0 for k1 in all_data[0][i:i+tw]]) and all([k2>=0 for k2 in out_val]):
                train_inout_seq.append((in_l, out_val))
                train_indices.append(i+tw)

    test_seqs = []
    test_indices = []
    for j in range(len(all_data[0])-2*tw):
        check_l2 = p_labels[j:j+2*tw]
        all_l2 = []
        for i in range(j,j+2*tw):
            timeslice = []
            for k in range(1,len(all_data)):
                timeslice.append(all_data[k][i])
            all_l2.append(timeslice)
        if all([k3>=0 for k3 in all_data[0][j:j+2*tw]]) and all([check_l2[q]==check_l2[0] for q in range(len(check_l2))]):    
            test_seq = (all_l2[j:j+tw],all_l2[j+tw:j+2*tw])#hw:make multidimensional
            test_seqs.append(test_seq)
            test_indices.append(j+tw)

    return train_inout_seq, test_seqs, (train_indices, test_indices)

#needs to be made multidimensional
def create_inout_sequences(iners_col, tw, p_labels):
    train_inout_seq = []
    train_indices = []
    for i in range(len(iners_col)-tw):
        in_l = iners_col[i:i+tw]
        out_val = iners_col[i+tw:i+tw+1]

        check_l = p_labels[i:i+tw+1]

        if all([check_l[i]==check_l[0] for i in range(len(check_l))]):
            if all([k1>=0 for k1 in in_l]) and all([k2>=0 for k2 in out_val]):
                train_inout_seq.append((in_l, out_val))
                train_indices.append(i+tw)

    test_seqs = []
    test_indices = []
    for j in range(len(iners_col)-2*tw):
        iners_l2 = iners_col[j:j+2*tw]
        check_l2 = p_labels[j:j+2*tw]
        if all([k3>=0 for k3 in iners_l2]) and all([check_l2[q]==check_l2[0] for q in range(len(check_l2))]):
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

def train1(model, trainingData, labelData, loss_function1, optimizer):
    model.train()
    sum_loss = 0
    for i in range(0,len(trainingData),batch_size):
        seq = trainingData[i:i+batch_size]
        labels = labelData[i:i+batch_size]
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

def one_loss(output,target):
    loss = torch.mean((output-target)**2)
    return loss

def train2(model, train_data, optimizer, index):
    model.train()
    sum_loss = 0
    for seq, labels in train_data:
        seq = torch.FloatTensor(seq).to(device)
        labels = torch.FloatTensor(labels).to(device)
        model.hidden = (torch.zeros(layers, batch_size, model.hidden_layer_size).to(device),
                        torch.zeros(layers, batch_size, model.hidden_layer_size).to(device))

        y_pred = model(seq)[index]
        single_loss = one_loss(y_pred, labels[index])
        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()
        sum_loss += single_loss.item()
    avg_loss = float(sum_loss/len(train_data))
    return avg_loss

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

def simavg(model_in, loss_function, test_in, fut_pred, data_avg):
    test_result_list = []
    avg = copy.deepcopy(data_avg)
    test_in_local = copy.deepcopy(test_in)
    #pdb.set_trace()
    fut_pred_local = fut_pred
    model_in.hidden = (torch.zeros(layers, batch_size, model_in.hidden_layer_size).to(device),
                       torch.zeros(layers, batch_size, model_in.hidden_layer_size).to(device))
    best_ap = test1(model_in,test_in_local,fut_pred_local).to(device)
    for j in range(len(blastT_labels)):
        test_in_local = copy.deepcopy(test_in)
        model_in.hidden = (torch.zeros(layers, batch_size, model_in.hidden_layer_size).to(device),
                           torch.zeros(layers, batch_size, model_in.hidden_layer_size).to(device))
        for i in range(len(test_in_local)):
            #continue
            test_in_local[i][j] = avg[j]#current offending line
        ap_test2 = test1(model_in,test_in_local,fut_pred_local).to(device)#was offending line before
        mse_diff = loss_function(ap_test2, best_ap)
        #mse_diff = loss_function(best_ap, best_ap)
        multi_diff = torch.sum(mse_diff,dim=0) #dim=0-->shape[15]
        row = multi_diff.tolist()
        row2 = [y/z/14 for y, z in zip(row,blastT_avg)]
        test_result_list.append(row2)
    return test_result_list

def simrand(model, loss_function, test_in, fut_pred):
    test_result_list = []
    test_in_local = copy.deepcopy(test_in)
    best_ap = test1(model,test_in_local,fut_pred).to(device)
    for j in range(len(blastT_labels)):
        one_result = []
        test_in_local = copy.deepcopy(test_in)
        for k in range(100):
            noise = np.random.normal(blastT_avg[j],blastT_stds[j],14)
            test_in_local = copy.deepcopy(test_in)
            for i in range(len(test_in_local)):
                test_in_local[i][j] = blastT_avg[j]+noise[i]
            ap_test2 = test1(model,test_in_local,fut_pred).to(device)
            mse_diff = loss_function(ap_test2, best_ap)
            multi_diff = torch.sum(mse_diff,dim=0) #dim=0-->shape[15]
            row = multi_diff.tolist()
            one_result.append(row)
        one_array = np.array(one_result)
        one_avg = np.average(one_array,axis=0)
        row2 = [y/z/14 for y, z in zip(one_avg,blastT_avg)]
        test_result_list.append(row2)
    return test_result_list

def simtime(model, loss_function, test_in, fut_pred):
    test_result_list = []
    test_in_local = copy.deepcopy(test_in)
    best_ap = test1(model,test_in_local,fut_pred).to(device)
    for i in range(len(test_in_local)):
        test_in_local = copy.deepcopy(test_in)
        for j in range(len(blastT_labels)):
            test_in_local[i][j] = blastT_avg[j]
        ap_test2 = test1(model,test_in_local,fut_pred).to(device)
        mse_diff = loss_function(ap_test2, best_ap)
        multi_diff = torch.sum(mse_diff,dim=1) #dim=0-->shape[15]
        row = multi_diff.tolist()
        row2 = [y/14 for y in row]
        test_result_list.append(row2)
    return test_result_list

def full_train(num_channels,hsize,layers,dropout,batch_size,LR,num_epochs,xData,yData,test_in,tensor_true_test_data):
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
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

    #starting index
    index = 13
    loss_function1 = nn.MSELoss().to(device)

    for i in range(num_epochs):
        permutation=list(np.random.permutation(xData.shape[0]))
        shuffled_X = xData[permutation,:]
        shuffled_Y = yData[permutation,:]
        shuffled_X = shuffled_X.type(torch.FloatTensor).to(device)
        shuffled_Y = shuffled_Y.type(torch.FloatTensor).to(device)
        #pdb.set_trace()
        avg_loss = train1(model, shuffled_X, shuffled_Y, loss_function1, optimizer)
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
        if i%500 == -1:
            index += 1
            if index > 14:
                index += -15
    
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

def plot_select(best_prediction,index):
    plt.show(block=False)
    plt.pause(0.05)
    return

def plot_lacto(true_data,best_prediction,label_text):
    index = 12
    fig1, ax1 = plt.subplots()
    title = 'Best Fits'
    ax1.set_title(title)
    ylabel = 'Relative Abundance Run'
    ax1.set_ylabel(ylabel)
    ax1.plot(true_data[:,index:].tolist(), label=label_text[index:])
    for best_p in best_predictions:
        array_bestp = np.array(best_p.cpu())
        ax1.plot(array_bestp[:,index:], '--')
    ax1.legend()
    plt.show(block=False)
    plt.pause(0.05)
    return

def plot_others(tensor_true_test_data,best_prediction,blastT_labels):
    fig,axs = plt.subplots(3,4)
    for i in range(3):
        for j in range(4):
            axs[i,j].set_title(blastT_labels[4*i+j])
            axs[i,j].plot(tensor_true_test_data[:,4*i+j].tolist())
            axs[i,j].plot(best_prediction[:,4*i+j].tolist(), '--')

        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='rel abund')

        for ax in axs.flat:
            ax.label_outer()
    
    plt.show(block=False)
    plt.pause(0.05)
    return

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_data = pd.read_excel("VMBData_clean.xlsx")
blast_data = all_data.filter(regex="^blast*")
blast_list = blast_data.T.reset_index().values.tolist()

#averages to place in for missing data
blast_avg = []
for i in range(len(blast_list)):
    temp_array = np.array(blast_list[i][1:])
    temp_sum = np.nansum(temp_array)
    temp_len = len(temp_array) - np.count_nonzero(np.isnan(temp_array))
    blast_avg.append(temp_sum/temp_len)
blastT_avg = blast_avg.copy()
blastT_avg.reverse()
    
#interpolation
for j in range(len(blast_list)):
    for i in range(1,len(blast_list[j])):
        if (blast_list[j][i]>=0)==False:
            if ((blast_list[j][i+1]>=0)==True) and ((blast_list[j][i-1]>=0)==True):
                blast_list[j][i] = (blast_list[j][i+1]+blast_list[j][i-1])/2

#big matrix & rotation
blastT_list = [list(r) for r in zip(*blast_list[::-1])]
blastT_labels = blastT_list[0]
blastT_nums = blastT_list[1:]
blastT_stds = []
#pdb.set_trace()
for i in range(len(blast_list)):
    blastT_stds.append(np.nanstd(np.array(blast_list[i][1:],dtype=float)))
blastT_stds.reverse()
#print(all_data)

#rescale input data
if(1):
    x = np.array(blastT_nums)
    if(1):
        scaler = MinMaxScaler() #goes with nn.Sigmoid()
        arr_norm = scaler.fit_transform(x)
    if(0):
        m = np.nanmean(x,axis=0,dtype=float,keepdims=True)
        s = np.nanstd(x,axis=0,dtype=float,keepdims=True)
        x -= m
        x /= s*3
        arr_norm = x
    x2 = arr_norm.data.tolist()
    print("Data Rescaled")
if(0):
    x2 = blastT_nums
    print("Data NOT Rescaled")
#pdb.set_trace()
p_labels = all_data['meta_Participant'].values.tolist()

train_window = 14
train_inout_seq, test_seqs, indices_tuple = create_inout_sequences3(x2, train_window, p_labels)

#test trajectory number
#this controls which test trajectory you are using
ttn = 51 #5,10,51,69
test_seq = test_seqs[ttn]

test_ind = indices_tuple[1][ttn]
train_indices_to_remove = list(range(test_ind, test_ind+train_window))

clean_train_inout_seq = []
for seq_ind in range(len(train_inout_seq)):
    if indices_tuple[0][seq_ind] not in train_indices_to_remove:
        clean_train_inout_seq.append(train_inout_seq[seq_ind])

#fullTrainingData = np.array(clean_train_inout_seq)[:, 0]
#fullTestingData = np.array(clean_train_inout_seq)[:, 0]

num_traj = len(clean_train_inout_seq)
num_timesteps = len(clean_train_inout_seq[0][0])
num_inputs = len(clean_train_inout_seq[0][0][0])
trainingData = torch.empty((num_traj,num_timesteps,num_inputs),dtype=float)
labelData = torch.empty((num_traj,num_inputs),dtype=torch.float)
#pdb.set_trace()
for i in range(num_traj):
    trainingData[i,:] = torch.FloatTensor(clean_train_inout_seq[i][0])
    labelData[i,:] = torch.FloatTensor(clean_train_inout_seq[i][1])
#pdb.set_trace()
test_in = test_seq[0]
test_out = test_seq[1]
#define testing block
fut_pred = 14
num_channels = len(test_in[0])
tensor_true_test_data = torch.FloatTensor(test_out).view(-1,num_channels).to(device)
tensor_true_test_data2 = tensor_true_test_data.detach().clone()

#Params
hsize=30
layers=3
batch_size=800 #always 1 because input isn't set up otherwise
num_epochs=800
LR=0.001
dropout=0.3
ensemble_size = 2

simavg_results = []
simrand_results = []
simtime_results = []
best_predictions = []
enum = 0

for z in range(ensemble_size):
    print(f'H:{hsize} L:{layers} D:{dropout} E:{num_epochs} LR:{LR}')
    best_prediction, best_model = full_train(num_channels,hsize,layers,dropout,batch_size,LR,
                                             num_epochs,trainingData,labelData,test_in,
                                             tensor_true_test_data)
    best_predictions.append(best_prediction)
    #test permutations with avg as replacement
    loss_function = nn.MSELoss(reduction='none').to(device)
    results1 = simavg(best_model, loss_function, test_in, fut_pred, blastT_avg)
    simavg_results.append(results1)
    #test avg + rand permutations
    results2 = simrand(best_model, loss_function, test_in, fut_pred)
    simrand_results.append(results2)
    #test time replaced with avg values
    results3 = simtime(best_model, loss_function, test_in, fut_pred)
    simtime_results.append(results3)
    #pdb.set_trace()

#plot last example only
plot_lacto(tensor_true_test_data,best_prediction,blastT_labels)
plot_others(tensor_true_test_data,best_prediction,blastT_labels)

#plot simavg
array_simavg = np.array(simavg_results)
simavg_avg =  np.average(array_simavg,axis=0)
simavg_df = pd.DataFrame(simavg_avg,index=blastT_labels,columns=blastT_labels)

#pdb.set_trace()

fign,axn = plt.subplots()
axn.set_title('Species avg')
axn = sns.heatmap(simavg_df,linewidth=0.5)#,norm=LogNorm())
plt.tight_layout()
plt.show(block=False)

#plot simrand
array_simrand = np.array(simrand_results)
simrand_avg = np.average(array_simrand,axis=0)
simrand_df = pd.DataFrame(simrand_avg,index=blastT_labels,columns=blastT_labels)

figr,axr = plt.subplots()
axr.set_title('Species avg + rand')
axr = sns.heatmap(simrand_df, linewidth=0.5)
plt.tight_layout()
plt.show(block=False)

#plot removing time points
array_simtime = np.array(simtime_results)
simtime_avg = np.average(array_simtime,axis=0)
simtime_df = pd.DataFrame(simtime_avg,index=range(14),columns=range(14,28))

figt,axt = plt.subplots()
axt.set_title('Time avg')
axt = sns.heatmap(simtime_df, linewidth=0.5)
plt.tight_layout()
plt.show()



