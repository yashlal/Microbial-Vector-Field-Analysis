import torch
import torch.nn as nn

class MM_GRU_CLASS(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=1, dropout=0.5, batch_size=1, output_size=1):
        super(MM_GRU_CLASS, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.SM = nn.Sigmoid()

    def forward(self, input_seq):
        gru_out, self.hidden = self.gru(input_seq.view(-1,len(input_seq[0]),len(input_seq[0][0])), self.hidden)
        pred = self.linear(gru_out.view(-1,len(gru_out[0]),len(gru_out[0][0])))[:,-1]
        pred = self.SM(pred)
        return pred