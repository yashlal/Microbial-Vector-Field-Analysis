import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import sys
import seaborn as sns

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('dataset.csv', index_col=0)
species = df.columns.values.tolist()[2:]

traj_nums = list(range(1,190))
trajs = []

for i in traj_nums:
    trajs.append(df.loc[df['Trajectory'] == i].iloc[:, 2:].to_numpy())

def main(plots='stack'):
    dt=1
    t_train_multi = []
    traj_train = []
    for traj in trajs:
        if len(traj)>=4:
            t = np.arange(0, len(traj), dt)
            t_train_multi.append(t)
            traj_train.append(traj)

    n_features = 136
    n_targets = 15
    constraint_rhs = np.zeros((136,))
    constraint_lhs = np.zeros((136, n_targets * n_features))
    for i in range(136):
        for j in range(15):
            col_ind = i+(136*j)
            constraint_lhs[i, col_ind] = 1

    # optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, constraint_lhs=constraint_lhs, max_iter=30)
    trapp_opt = ps.TrappingSR3()
    model = ps.SINDy(optimizer=trapp_opt)
    model.fit(traj_train, multiple_trajectories=True, t=t_train_multi)
    model.print()

    test_traj = trajs[0]
    m = len(test_traj)
    t_test_sim = np.arange(m, step=1)
    sim_test = model.simulate(test_traj[0], t=t_test_sim)

    col = sns.color_palette("hls", 15)

    if plots=='stack':
        fig = plt.figure()
        ax = fig.add_subplot(121)
        plt.stackplot(list(range(len(test_traj))), test_traj[:,0], test_traj[:,1], test_traj[:,2], test_traj[:,3], test_traj[:,4], test_traj[:,5], test_traj[:,6], test_traj[:,7], test_traj[:,8],test_traj[:,9], test_traj[:,10], test_traj[:,11], test_traj[:,12], test_traj[:,13], test_traj[:,14], colors=col, labels=species)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc=4)
        plt.xlabel('Time')
        plt.ylabel('Relative Abundance')
        plt.title(f'Testing Trajectory')


        fig.add_subplot(122)
        plt.stackplot(list(range(len(sim_test))), sim_test[:,0], sim_test[:,1], sim_test[:,2], sim_test[:,3], sim_test[:,4], sim_test[:,5], sim_test[:,6], sim_test[:,7], sim_test[:,8],sim_test[:,9], sim_test[:,10], sim_test[:,11], sim_test[:,12], sim_test[:,13], sim_test[:,14], colors=col)
        plt.xlabel('Time')
        plt.ylabel('Relative Abundance')
        plt.title(f'Simulated Trajectory')

        plt.tight_layout()
        plt.show()

    if plots=='error':
        error_mat = sim_test-test_traj
        fig, ax = plt.subplots(3, 5, sharex='col', sharey='row')

        fig.supxlabel('Time')
        fig.supylabel('Predicted-Real in RA')

        for i in range(3):
            for j in range(5):
                ind_spec = 5*i+j
                ax[i,j].plot(error_mat[:,ind_spec])
                ax[i,j].title.set_text(f'{species[ind_spec]}')
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    main(plots='stack')
    main(plots='error')
