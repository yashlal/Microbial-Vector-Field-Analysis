import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('dataset.csv', index_col=0)
species = df.columns.values.tolist()[2:]

traj_nums = list(range(1,190))
trajs = []

for i in traj_nums:
    trajs.append(df.loc[df['Trajectory'] == i].to_numpy())

for j, traj in enumerate(trajs):
    if len(traj)==18:
        print(j)
