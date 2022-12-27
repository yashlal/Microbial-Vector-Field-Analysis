import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from torch.utils.data import Dataset
import collections
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from MM_LSTM import MM_LSTM_CLASS

def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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

    avg_loss = sum_loss / (n_batches*batch_size)
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

def full_train(num_channels, hsize, layers, dropout, batch_size, LR, num_epochs, train_dataloader, test_in, tensor_true_test_data, device):
    # Print info every 25 epochs
    print(f'hsize:{hsize}       layers:{layers}       num_epochs:{num_epochs}       dropout:{dropout}      batch_size:{batch_size}       lr:{LR}')
    
    test_in_local = copy.deepcopy(test_in)

    # model and optimizer setup
    model = MM_LSTM_CLASS(input_size=num_channels,hidden_layer_size=hsize,num_layers=layers,output_size=num_channels,dropout=dropout,batch_size=batch_size).to(device)
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    optimizer.state = collections.defaultdict(dict)
    fut_pred = len(tensor_true_test_data)
    best_loss = 9999.9

    # Param inits
    # Biases are zero, weights are orthogonal
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
        avg_loss = train_model(model, train_dataloader, loss_function1, optimizer, device=device)
        #Obtain testing loss every 25 epochs and compare to find the best model
        if i%25 == 1 or i==num_epochs-1:

            ap = test_model(model, test_in_local, fut_pred, train_window=14, device=device).view(-1,num_channels).to(device)
            test_loss = loss_function1(ap, tensor_true_test_data)
            print(f'Epoch:{i}       TestLoss1:{test_loss}      TrainLoss:{avg_loss}')

            if test_loss.item() < best_loss and i>300:
                best_loss = test_loss.item()
                best_i = i
                best_ap = ap.detach().clone()
                model_clone = MM_LSTM_CLASS(input_size=num_channels,hidden_layer_size=hsize,num_layers=layers,output_size=num_channels,dropout=dropout,batch_size=batch_size).to(device)
                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))

    best_prediction = best_ap#.tolist()

    del loss_function1, optimizer, model
    torch.cuda.empty_cache()

    return best_prediction, best_i, model_clone, best_loss

def plot_best_fit(true_data, prediction, model_pred_color, true_data_color, idx, labels, save, save_dir, full_test_data):
    #idx is a list of the index of the species
    #idx of the prediction and labels should correspond
    #prediction should be the output of the testing scheme
    #model_pred_color and true_data_color are both lists that should correspond to the color you want for each species plotted
    #true_data should be shape (timesteps, species)
    assert len(model_pred_color) == len(true_data_color) == len(idx), 'Make sure the lists of colors and indexes are the same size'
    assert len(true_data[0]) == len(labels), 'Make sure the size of the features and labels are the same shape'
    plt.close()
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['figure.figsize'] = [8, 6]
    pred_labels = ' prediction'
    best_pred = torch.clone(prediction).detach()
    fig2, ax2 = plt.subplots()
    title = 'LSTM Fits'
    ax2.set_title(title, fontsize=25)
    ylabel = 'Relative Abundance'
    ax2.set_ylabel(ylabel, fontsize=25)
    ax2.set_xlabel('Time Step', fontsize=25)
    cidx = 0
    for index in idx:
        ax2.plot(full_test_data[:,index].tolist(), label=labels[cidx], lw=3, c=model_pred_color[cidx])
        array_bestp1 = np.array(best_pred.cpu())
        ax2.plot(range(14, 28), array_bestp1[:,index], label=labels[cidx] + pred_labels, ls = ':', lw=4, c=true_data_color[cidx])
        cidx += 1
    ax2.tick_params('both', length=10, width=3)
    ax2.legend(fontsize=13, loc='lower left')
    # plt.tight_layout()
    if (save_dir is not None) and save:
        plt.savefig('LSTMFit.png')
    plt.show()