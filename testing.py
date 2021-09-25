import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pysindy as ps

myargs = (a,b) = (0.08,0.08)

def ODE(X, t, a, b):
    x, y = X[0], X[1]
    dx = 1*x-2.5*(x*y)
    dy = 2.5*(x*y)-1*y
    return [dx,dy]

def for_plot():

    global myargs

    dt = 0.01
    time_rdxn = 1
    t_min = 0
    t_max = 10

    t_train = np.arange(t_min, t_max, dt)
    x0s_train = [[2,2]]
    x_train_multi = [odeint(ODE, x0_train, t_train, args=myargs)[::time_rdxn] for x0_train in x0s_train]

    model = ps.SINDy(feature_names=['x','y'])
    model.fit(x_train_multi, t=dt, multiple_trajectories=True)
    model.print()

    t_test = np.arange(t_min, t_max, dt)
    x0_test = np.array([1,1])
    x_test = odeint(ODE, x0_test, t_test, args=myargs)

    sc = model.score(x_test, t=dt)
    print(f'Acc is {sc}')
    x_pred = model.simulate(x0_test, t_test)

    return model, x_test, x_pred

mod, x_test_grph, x_pred_grph = for_plot()

t_traj = np.arange(0, 100, 0.01)
x0_trajs = [[1,1],[2,2],[3,3], [0.5,0.5],[4,4]]
trajs_real = [odeint(ODE, x0_traj, t_traj, args=myargs) for x0_traj in x0_trajs]
trajs_pred = [mod.simulate(x0_traj, t_traj) for x0_traj in x0_trajs]

max = 8
xy = []
for i in range(-1,10*max):
    for j in range(-1,10*max):
        xy.append((i/10,j/10))

xy = np.array(xy)
x, y = xy[:,0], xy[:,1]

dx = 1*x-2.5*(x*y)
dy = 2.5*(x*y)-1*y

u_real, v_real = dx,dy
u_pred, v_pred = mod.predict(xy)[:,0], mod.predict(xy)[:,1]

func_x = x_test_grph[:,0]
func_y = x_test_grph[:,1]

fig = plt.figure()

fig.add_subplot(121)
plt.quiver(x,y,u_real, v_real)
for traj in trajs_real:
    plt.plot(traj[:,0],traj[:,1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('True Vector Field')

fig.add_subplot(122)
plt.quiver(x,y,u_pred,v_pred, color='red')
for traj in trajs_pred:
    plt.plot(traj[:,0],traj[:,1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted Vector Field')

plt.tight_layout()
plt.show()
