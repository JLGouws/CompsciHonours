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

analytic = lambda t : 0.05 * np.exp(t) / (0.05  * (np.exp(t) - 1) + 1)

################################################################################
#1a.
fig, ax = plt.subplots(1, 2, figsize=(10,5))
time = []
solution = []
u = []
E = []
for i, res in enumerate([1000, 2000, 4000]):
    points = np.array(taylors_method(logistic, 0, 10, res, 0.05))
    time += [points[:, 0]]
    solution += [points[:, 1]]
    u += [analytic(time[-1])]
    E += [u[-1] - solution[-1]]
    ax[0].plot(time[-1], E[-1])
    ax[1].semilogy(time[-1], np.abs(E[-1]))
    #ax[2].plot(time, solution)
#ax[0].plot(time, u)
fig.savefig("1a.pdf")
################################################################################
V_r = (solution[-2] - 2 * solution[-1][::2]) / (1 - 2)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(time[-2], V_r)
fig.savefig("1b.pdf")
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].plot(time[-2], E[-2], linewidth = 8)
ax[0].plot(time[-2], V_r - solution[-2], linewidth = 3)
ax[1].plot(time[-2], (V_r - solution[-2])/E[-2])
fig.savefig("1c.pdf")

################################################################################
#2a.
fig, ax = plt.subplots(1, 2, figsize=(10,5))
time = []
solution = []
u = []
E = []
for i, res in enumerate([1000, 2000, 4000]):
    points = np.array(runge_kutta2(logistic, 0, 10, res, 0.05))
    time += [points[:, 0]]
    solution += [points[:, 1]]
    u += [analytic(time[-1])]
    E += [u[-1] - solution[-1]]
    ax[0].plot(time[-1], E[-1])
    ax[1].semilogy(time[-1], np.abs(E[-1]))
    #ax[2].plot(time, solution)
#ax[0].plot(time, u)
fig.savefig("2a.pdf")

V_r = (solution[-2] - 2 **2 * solution[-1][::2]) / (1 - 2 ** 2)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(time[-2], V_r)
fig.savefig("2b.pdf")
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].plot(time[-2], E[-2], lw = 8)
ax[0].plot(time[-2], V_r - solution[-2], lw = 3)
ax[1].plot(time[-2], (V_r - solution[-2])/E[-2])
fig.savefig("2c.pdf")

#V_r = (solution[-2] - 2**2 * solution[-1][::2]) / (1 - 2**2)
#fig, ax = plt.subplots(figsize=(10,5))
#ax.plot(time[-2], V_r)
#fig.savefig("2b.pdf")
################################################################################
#4.
fig, ax = plt.subplots(1, figsize=(10,10))
time = []
solution = []
u = []
E = []
L2Euler = []
dtEuler = []
for i, res in enumerate([125, 250, 500, 1000, 2000, 4000]):
    points = np.array(taylors_method(logistic, 0, 10, res, 0.05))
    time += [points[:, 0]]
    solution += [points[:, 1]]
    u += [analytic(time[-1])]
    E += [u[-1] - solution[-1]]
    L2Euler += [np.sqrt(10 / res * np.sum(E[-1] ** 2))]
    dtEuler += [10 / res]
    #ax[2].plot(time, solution)
#ax[0].plot(time, u)
ax.loglog(dtEuler, L2Euler, 'o-', label = 'Euler')
time = []
solution = []
u = []
E = []
L2RK2 = []
dtRK2 = []
for i, res in enumerate([125, 250, 500, 1000, 2000, 4000]):
    points = np.array(runge_kutta2(logistic, 0, 10, res, 0.05))
    time += [points[:, 0]]
    solution += [points[:, 1]]
    u += [analytic(time[-1])]
    E += [u[-1] - solution[-1]]
    L2RK2 += [np.sqrt(10 / res * np.sum(E[-1] ** 2))]
    dtRK2 += [10 / res]
    #ax[2].plot(time, solution)
#ax[0].plot(time, u)
ax.loglog(dtRK2, L2RK2, 'o-', label = 'Runge-Kutta 2')
ax.legend()
fig.savefig("4.pdf")
