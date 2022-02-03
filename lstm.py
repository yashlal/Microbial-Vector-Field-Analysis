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

# print('74 Trajs with 862 Data Points')

test_traj = trajs[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_dim=60, output_size=15):
        super(MyLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.lin(lstm_out)
        out = self.softmax(out)
        return out

model = MyLSTM()
model.to(device)
loss_function = nn.KLDivLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epoch = 15

loss_values = []


def train():
    for epoch in range(n_epoch):

        traj = test_traj
        for i in range(len(traj)-1):

            model.zero_grad()

            input = torch.tensor(traj[i]).to(device)
            input = torch.reshape(input, (1,1,15))

            pred = model(input)
            true = torch.tensor(traj[i+1]).to(device)
            true = torch.reshape(true, (1,1,15))

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

lv = train()
plt.plot(lv)
plt.show()
