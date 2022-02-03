import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import torch.nn as nn
import torch

df = pd.read_csv('dataset.csv', index_col=0)
species = df.columns.values.tolist()[2:]
traj_nums = list(range(1,190))
trajs = []

for i in traj_nums:
    traj = df.loc[df['Trajectory'] == i].iloc[:, 2:].values.tolist()
    trajs.append(traj)

trajs = list(filter(lambda x: len(x)>4, trajs))

print('74 Trajs with 862 Data Points')

test_traj = trajs[0]
train_traj = trajs[1:]

class LSTM(nn.Module):
    def __init__(self, num_sensors=15, hidden_units=60):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=15)
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to('cuda')
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to('cuda')

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        out = self.softmax(out)
        return out

model = LSTM()
model.to('cuda')
loss_function = nn.KLDivLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_epoch = 1

loss_values = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_test():
    for epoch in range(n_epoch):

        traj = test_traj
        for i in range(len(traj)-1):

            model.zero_grad()

            input = torch.tensor(traj[i]).to(device)
            input = torch.reshape(input, (1,1,15))

            pred = model(input)
            true = torch.tensor(traj[i+1]).to(device)

            loss = (loss_function(pred, true)).to(device)

            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

            # if (epoch%100)==0 and (i==0):
            #     print(f'Epoch {epoch}')
            #     print(f'Loss {loss.item()}')
            #     plt.plot(loss_values)
            #     plt.show()
            #     loss_values.clear()

    return loss_values

lv = train_and_test()
plt.plot(lv)
plt.show()
