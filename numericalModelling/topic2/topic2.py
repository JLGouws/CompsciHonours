import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def taylors_method(f, a, b, N, alpha):
  h = (b -a)/N
  Y = [(a, alpha)]
  for i in range (1, N + 1):
    Y += [(a + i * h, Y[-1][1] + h * f(Y[-1][0], Y[-1][1]))]
  return Y

def midpoint_method(f, a, b, N, alpha):
  h = (b -a)/N
  Y = [(a, alpha), (a + h, alpha + h * f(a, alpha))]
  for i in range (2, N + 1):
    Y += [(a + i * h, Y[-2][1] + 2 * h * f(Y[-1][0], Y[-1][1]))]
  return Y

def runge_kutta2(f, a, b, N, w):
  h = (b - a) / N
  Y = [(a, w)]
  for i in range(1, N + 1):
    K1 = h * f(a + (i - 1) * h, Y[-1][1])
    K2 = h * f(a + i * h, Y[-1][1] + K1)
    Y += [(a + i * h, Y[-1][1] + (K1 + K2) / 2)]
  return Y

logistic = lambda t, u : 1 * u * (1 - u)

################################################################################
#1a.
points = np.array(taylors_method(logistic, 0, 20, 2000, 0.05))
time = points[:, 0]
solution = points[:, 1]

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(time, solution, lw = 8, c = "hotpink")
fig.savefig("figs/1a.pdf")

################################################################################
#1b.
points = np.array(midpoint_method(logistic, 0, 20, 2000, 0.05))
time = points[:, 0]
solution = points[:, 1]

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(time, solution, 'o', c = "lime")
fig.savefig("figs/1b.pdf")

################################################################################
#1c.
points = np.array(runge_kutta2(logistic, 0, 20, 2000, 0.05))
time = points[:, 0]
solution = points[:, 1]

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(time, solution, c = "aqua")
fig.savefig("figs/1c.pdf")

################################################################################
#2.

u0 = np.round(np.linspace(-0.1, 2.0, 22).T, 2)
t0 = np.zeros(22).T
tf = np.full((22, 1),7)[0].T
points = np.array(runge_kutta2(logistic, t0, tf, 2000, u0))

time = points[:, 0]
solution = points[:, 1]

df = pd.DataFrame(data = {'time' : np.ravel(time, order = 'F'), '$\\nu$' : np.ravel(solution, order = 'F'), 'u_0' : np.round(np.repeat(np.linspace(-0.1, 2.0, 22).T, 2001), 2)})

fig= plt.figure(figsize=(10,10))
ax = sns.lineplot(data = df, x = 'time', y = '$\\nu$', hue = 'u_0', palette = 'icefire', legend = 'full')
ax.set(ylim = [-.75, 2.2], xlim = [0.0, 7.0])
ax.legend(loc = 'upper right', title = '$u_0$')
fig.savefig("figs/2.pdf")

################################################################################
#3.
#a.
udot = lambda t, u: u - np.exp(t / 2) * np.sin(5 * t) + 5 * np.exp(t / 2) * np.cos(5 * t)

fig, ax = plt.subplots(3, 1, figsize=(10,15))
for i, (method, name) in enumerate([(taylors_method, "Euler's Method"), 
                                    (midpoint_method, "Midpoint Method"), 
                                    (runge_kutta2, "Runge-Kutta Second Order Method")]):
    points = np.array(method(udot, 0, 5, 63, 0))
    time = points[:, 0]
    solution = points[:, 1]

    ax[i].plot(time, solution, label = '$\Delta t = 0.08$', alpha = 0.5, c = 'r')

    points = np.array(method(udot, 0, 5, 125, 0))
    time = points[:, 0]
    solution = points[:, 1]

    ax[i].plot(time, solution, label = '$\Delta t = 0.04$', alpha = 0.6, c = 'b')

    ax[i].legend(title = 'Time step')

    points = np.array(method(udot, 0, 5, 500, 0))
    time = points[:, 0]
    solution = points[:, 1]

    ax[i].plot(time, solution, 'k-', label = '$\Delta t = 0.01$', dashes = (3, 7))

    ax[i].legend(title = 'Time step')
    ax[i].set(title = name, xlabel = 'time', ylabel = '$\\nu$')

fig.savefig("figs/3a.pdf")
