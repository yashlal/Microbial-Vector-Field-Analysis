import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.SM = nn.Sigmoid()

    def forward(self, input_seq):

        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        pred = self.linear(lstm_out.view(len(input_seq), -1))[-1]
        pred = self.SM(pred)
        return pred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vag_data = pd.read_excel("VMBData_clean.xlsx")

all_data = vag_data['blast_L_iners'].values.astype(float).tolist()

for i in range(len(all_data)):
    if (all_data[i]>=0)==False:
        if ((all_data[i+1]>=0)==True) and ((all_data[i-1]>=0)==True):
            all_data[i] = (all_data[i+1]+all_data[i-1])/2

p_labels = vag_data['meta_Participant'].values.tolist()

train_window = 14
train_inout_seq, test_seqs, indices_tuple = create_inout_sequences(all_data, train_window, p_labels)

#test trajectory number
ttn = 0
test_seq = test_seqs[ttn]

test_ind = indices_tuple[1][ttn]
train_indices_to_remove = list(range(test_ind, test_ind+train_window))

clean_train_inout_seq = []
for seq_ind in range(len(train_inout_seq)):
    if indices_tuple[0][seq_ind] not in train_indices_to_remove:
        clean_train_inout_seq.append(train_inout_seq[seq_ind])

test_in = test_seq[0]
test_out = test_seq[1]


def test1(model, test_inputs, fut_pred):
    model.eval()
    new_test_inputs = test_inputs.copy()
    for j in range(fut_pred): #'fut_pred' for only predictions
        seq = torch.FloatTensor(new_test_inputs[-train_window:]).to(device)
        with torch.no_grad():
            model.hidden = (torch.zeros(1,1,model.hidden_layer_size),
                            torch.zeros(1,1,model.hidden_layer_size))
            new_test_inputs.append(model(seq).item())
    output = new_test_inputs[-fut_pred:]

    return torch.FloatTensor(output)

def main(hsize=25, num_epochs=500, LR=0.001):
    #Params

    print(f'H:{hsize} E:{num_epochs} LR:{LR}')

    model = LSTM(hidden_layer_size=hsize).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    epochs = num_epochs

    train_loss_list = []
    test1_loss_list = []
    test2_loss_list = []

    for i in range(epochs):
        train_graph = []
        epoch_loss = []
        for seq, labels in clean_train_inout_seq:
            seq = torch.FloatTensor(seq).to(device)
            labels = torch.FloatTensor(labels).to(device)

            optimizer.zero_grad()
            model.train()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            train_loss = single_loss.item()
            optimizer.step()

            epoch_loss.append(train_loss)
            train_graph.append((y_pred.item(), labels.item()))

        if i%25 == 1 or i==epochs-1:

            #testing block
            fut_pred = 14
            tensor_true_test_data = torch.FloatTensor(test_out).view(-1,1).to(device)

            ap1 = test1(model, test_in, fut_pred).view(-1,1).to(device)
            test_loss1 = loss_function(ap1, tensor_true_test_data).item()

            test1_loss_list.append((i, test_loss1))

            np_train_graph = np.array(train_graph)

            fig1, ax1 = plt.subplots()
            ax1.plot(np_train_graph[:,0]-np_train_graph[:,1], label='True-Train Difference')
            ax1.legend()
            ax1.set_title('L. Iners NN Train')
            ax1.set_ylabel('Relative Abundance')
            plt.show()

            fig2, ax2 = plt.subplots()
            ax2.plot(tensor_true_test_data.tolist(), label='True')
            ax2.plot(ap1.tolist(), label='Test_Pred')
            ax2.legend()
            ax2.set_title('L. Iners NN Test')
            ax2.set_ylabel('Relative Abundance')
            plt.show()

            print(f'Epoch:{i}       TestLoss1:{test_loss1}')

        loss_avg = sum(epoch_loss)/len(epoch_loss)
        train_loss_list.append(loss_avg)

    test1_loss_list = np.array(test1_loss_list)

    fig3, ax3 = plt.subplots()
    ax3.plot(train_loss_list)
    ax3.set_title('Training Loss')
    plt.show()

    fig4, ax4 = plt.subplots()
    ax4.plot(test1_loss_list[:,0], test1_loss_list[:,1], label='Test1')
    ax4.legend()
    ax4.set_title('Testing Loss')
    plt.show()
