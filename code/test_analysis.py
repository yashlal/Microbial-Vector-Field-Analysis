import itertools as it
import random
import time
from tqdm import tqdm
import pickle
import numpy as np

all_vals = []
for i in range(187):
    with open(f'test_saves/indiv_runs/{i}.pickle', 'rb') as f:
        vals = pickle.load(f)[0][2]
        all_vals.append(vals)

all_vals_np = np.array(all_vals)

time_labels = all_vals_np[0, :, 0]
err_vals = all_vals_np[:, :, 1]

all_mean = np.mean(err_vals, 0)

random.seed(0)
print(all_mean)
N_ITER = 10**5
for n in [2,5,10]:
    sampled_means = []
    for i in tqdm(range(N_ITER)):
        curr_c = sorted(random.sample(range(187), n))
        sampled_means.append(np.mean(err_vals[curr_c, :], 0))
        
    np_sampled_means = np.array(sampled_means)
    m_arr = np.mean(np_sampled_means, 0)
    var_arr = np.var(np_sampled_means, 0)

    print(m_arr, var_arr)

    # with open(f'test_saves/means/{n}.pickle', 'wb') as f:
    #     pickle.dump(sampled_means,f)