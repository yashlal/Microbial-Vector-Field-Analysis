import itertools
import pickle
import matplotlib.pyplot as plt

# This file just takes the saves from the HP grid search and outputs them sorted and makes the histogram

search_space = list(itertools.product([5,10,20,50,100],[1,2,3,5],[0.1, 0.3, 0.5, 0.7], [32], [0.01, 0.001, 0.0001]))
vals = []
for config in search_space:
    hidden_layer_size = config[0]
    num_layers = config[1]
    dropout = config[2]
    batch_size = 32
    LR = config[4]

    f_path = f"hp_saves/pickles/{hidden_layer_size}_{num_layers}_{dropout}_{batch_size}_{LR}.pickle"
    with open(f_path, 'rb') as f:
        l = pickle.load(f)
        vals.append(l)
    
print(len(vals))
vals.sort(key=lambda x:x[1])
for c in vals:
    print(c)

v = [x[1] for x in vals]
plt.hist(v)
plt.show()