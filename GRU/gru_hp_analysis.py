import itertools
import pickle
import matplotlib.pyplot as plt

search_space = list(itertools.product([5,10,20,50,100],[1,2,3,5],[0.1, 0.3, 0.5, 0.7], [64, 32, 16, 4, 1], [0.01, 0.001, 0.0001]))
vals = []
for config in search_space:
    hidden_layer_size = config[0]
    num_layers = config[1]
    dropout = config[2]
    batch_size = config[3]
    LR = config[4]

    f_path = f"gru_hp_saves/{hidden_layer_size}_{num_layers}_{dropout}_{batch_size}_{LR}.pickle"
    try:
        with open(f_path, 'rb') as f:
            l = pickle.load(f)
            vals.append(l)
    except FileNotFoundError:
        print(f_path)
    
print(len(vals))
vals.sort(key=lambda x:x[1])

hist_vals = [x[1] for x in vals]
plt.hist(hist_vals)
plt.show()

