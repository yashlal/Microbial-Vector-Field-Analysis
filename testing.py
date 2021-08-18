import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pysindy as ps

myargs = (a,b,c,d,n) = (0.0035,-0.001,0.002,-0.002,2)

def ODE(X, t, a, b, c, d, n):
    x, y = X[0], X[1]
    dx = a*(x**n)+b*(y**n)
    dy = c*(x**n)+d*(y**n)
    return [dx,dy]

def for_plot():

    global myargs

    dt = 0.1
    t_min = 0
    t_max = 10

    t_train = np.arange(t_min, t_max, dt)
    x0s_train = [[1,1], [5,5], [12,12], [20,20], [35,35],]
    x_train_multi = [odeint(ODE, x0_train, t_train, args=myargs) for x0_train in x0s_train]

    print(x_train_multi)

    model = ps.SINDy(feature_names=['x','y'])
    model.fit(x_train_multi, t=dt, multiple_trajectories=True)
    model.print()

    t_test = np.arange(t_min, t_max, dt)
    x0_test = np.array([25,25])
    x_test = odeint(ODE, x0_test, t_test, args=myargs)

    sc = model.score(x_test, t=dt)
    print(f'Acc is {sc}')
    x_pred = model.simulate(x0_test, t_test)

    return model, x_test, x_pred

mod, x_test_grph, x_pred_grph = for_plot()

fig = plt.figure()

fig.add_subplot(121)
plt.plot(x_test_grph)
plt.title('True Trajectories')

fig.add_subplot(122)
plt.plot(x_pred_grph)
plt.title('Predicted Trajectories')

plt.tight_layout()
plt.show()

max = 50
xy = []
for i in range(1,max+1):
    for j in range(1,max+1):
        xy.append((i,j))

xy = np.array(xy)
x, y = xy[:,0], xy[:,1]

dx = a*(x**n)+b*(y**n)
dy = c*(x**n)+d*(y**n)

u_real, v_real = dx,dy
u_pred, v_pred = mod.predict(xy)[:,0], mod.predict(xy)[:,1]

fig = plt.figure()

fig.add_subplot(121)
plt.quiver(x,y,u_real, v_real)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('True Vector Field')

fig.add_subplot(122)
plt.quiver(x,y,u_pred,v_pred, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted Vector Field')

plt.tight_layout()
plt.show()
