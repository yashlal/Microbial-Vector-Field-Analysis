import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import sys
import seaborn as sns

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('dataset.csv', index_col=0)
species = df.columns.values.tolist()[2:]
print(species)
traj_nums = list(range(1,190))
trajs = []

for i in traj_nums:
    trajs.append(df.loc[df['Trajectory'] == i].iloc[:, 2:].to_numpy())

def main(plots='stack', traj_num=0):
    traj_train_multi=[]
    for traj in trajs:
        if len(traj)>=4:
            traj_train_multi.append(traj)

    training_traj = traj_train_multi[traj_num]


    library_functions = [
        lambda x : np.exp(x),
        lambda x,y: np.exp(x+y),
    ]

    library_function_names = [
        lambda x : 'exp(' + x + ')',
        lambda x,y : 'exp(' + x + y + ')',
    ]

    custom_library = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)

    identity_library = ps.IdentityLibrary()
    polynomial_library = ps.PolynomialLibrary(degree=3, include_interaction=True)
    fourier_library = ps.FourierLibrary(n_frequencies=5)

    combined_library = ps.ConcatLibrary([fourier_library, custom_library, identity_library, polynomial_library])

    trapp_opt = ps.TrappingSR3(threshold=0.01)
    model = ps.SINDy(optimizer=trapp_opt, discrete_time=True, feature_library=combined_library)
    model.fit(training_traj)
    model.print()
    print('Model score: %f' % model.score(training_traj, t=len(training_traj)))

    test_traj = training_traj
    m = len(test_traj)
    sim_test = model.simulate(test_traj[0], t=m)

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

    if plots=='test':
        for i in range(len(traj_train_multi)):
            test_traj = traj_train_multi[i]
            m = len(test_traj)
            try:
                sim_test = model.simulate(test_traj[0], t=m)
            except:
                continue

            fig = plt.figure()
            ax = fig.add_subplot(121)
            plt.stackplot(list(range(len(test_traj))), test_traj[:,0], test_traj[:,1], test_traj[:,2], test_traj[:,3], test_traj[:,4], test_traj[:,5], test_traj[:,6], test_traj[:,7], test_traj[:,8],test_traj[:,9], test_traj[:,10], test_traj[:,11], test_traj[:,12], test_traj[:,13], test_traj[:,14], colors=col, labels=species)
            plt.xlabel('Time')
            plt.ylabel('Relative Abundance')
            plt.title(f'Testing Trajectory {i}')


            fig.add_subplot(122)
            plt.stackplot(list(range(len(sim_test))), sim_test[:,0], sim_test[:,1], sim_test[:,2], sim_test[:,3], sim_test[:,4], sim_test[:,5], sim_test[:,6], sim_test[:,7], sim_test[:,8],sim_test[:,9], sim_test[:,10], sim_test[:,11], sim_test[:,12], sim_test[:,13], sim_test[:,14], colors=col)
            plt.xlabel('Time')
            plt.ylabel('Relative Abundance')
            plt.title(f'Simulated Trajectory {i}')

            plt.tight_layout()
            plt.savefig(f'Plots/dump,pfe/{i}.png')
            plt.clf()

    if plots=='solo':
        fig = plt.figure()
        ax = fig.add_subplot(121)
        plt.plot(test_traj)
        plt.xlabel('Time')
        plt.ylabel('Relative Abundance')
        plt.title(f'Testing Trajectory')


        fig.add_subplot(122)
        plt.plot(sim_test)
        plt.xlabel('Time')
        plt.ylabel('Relative Abundance')
        plt.title(f'Simulated Trajectory')

        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    main(plots='stack')
    # main(plots='error')
