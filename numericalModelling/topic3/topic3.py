import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, physics}'

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
colours = ["teal", "yellowgreen", "fuchsia"]
for i, res in enumerate([1000, 2000, 4000]):
    points = np.array(taylors_method(logistic, 0, 10, res, 0.05))
    time += [points[:, 0]]
    solution += [points[:, 1]]
    u += [analytic(time[-1])]
    E += [u[-1] - solution[-1]]
    ax[0].plot(time[-1], E[-1], label = "$\Delta t = $" + str(10 / res), c = colours[i])
    ax[1].semilogy(time[-1], np.abs(E[-1]), label = "$\Delta t = $" + str(10 / res), c = colours[i])
    #ax[2].plot(time, solution)
ax[0].legend()
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$|\\nu^n - u(t^n)|$")
ax[1].legend()
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("$|\\nu^n - u(t^n)|$")
fig.savefig("figs/1a.pdf")
################################################################################

fig, ax = plt.subplots(1, 2, figsize=(10,5), tight_layout = True)
colours = ["teal", "yellowgreen", "fuchsia"]
for i, res in enumerate([1000, 2000, 4000]):
    ax[0].plot(time[i], 2 ** i * E[i], ls = (12 * i, [12,24]), label = "$ " + (str(2 ** i) if 2 ** i != 1 else "") + " E_{ " + str(10 / res) + " }$", c = colours[i])
    ax[1].semilogy(time[i], np.abs(2 ** i * E[i]), ls = (12 * i, [12,24]), label = "$ " + (str(2 ** i) if 2 ** i != 1 else "") + " |E_{ " + str(10 / res) + " }|$" , c = colours[i])
    #ax[2].plot(time, solution)
ax[0].legend(handlelength = 5)
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$|\\nu^n - u(t^n)|$")
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("$|\\nu^n - u(t^n)|$")
ax[1].legend(handlelength = 5)
fig.savefig("figs/1b.pdf")
V_r = (solution[-2] - 2 * solution[-1][::2]) / (1 - 2)
#fig, ax = plt.subplots(figsize=(10,5))
#ax.plot(time[-2], V_r)
#fig.savefig("figs/1b.pdf")
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].semilogy(time[-2], np.abs(E[-2]), linewidth = 8, label = "$E^n_{0.05}$", c = "yellowgreen")
ax[0].semilogy(time[-2], np.abs(V_r - solution[-2]), linewidth = 2, label = "$\\tilde E^n_{0.05}$", c = "firebrick")
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$|\\nu^n - \\nu^n_R|$")
ax[1].plot(time[-2], (V_r - solution[-2])/E[-2], label = "$\\tilde E^n_{0.05} / E^n_{0.05}$", c = "chocolate", lw = 2)
ax[1].set_ylim([0.5, 1.5])
ax[1].legend()
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("$|\\nu^n - \\nu^n_R|$")
ax[0].legend()
fig.savefig("figs/1c.pdf")

################################################################################
#2a.
fig, ax = plt.subplots(1, 2, figsize=(10,5))
time = []
solution = []
u = []
E = []
colours = ["cyan", "grey", "firebrick"]
for i, res in enumerate([1000, 2000, 4000]):
    points = np.array(runge_kutta2(logistic, 0, 10, res, 0.05))
    time += [points[:, 0]]
    solution += [points[:, 1]]
    u += [analytic(time[-1])]
    E += [u[-1] - solution[-1]]
    ax[0].plot(time[-1], E[-1], c = colours[i], label = "$\Delta t = $" + str(10 / res))
    ax[1].semilogy(time[-1], np.abs(E[-1]), c = colours[i], label = "$\Delta t = $" + str(10 / res))
ax[0].legend()
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$|\\nu^n - u(t^n)|$")
ax[1].legend()
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("$|\\nu^n - u(t^n)|$")
fig.savefig("figs/2a.pdf")

#2b.
fig, ax = plt.subplots(1, 2, figsize=(10,5), tight_layout = True)
for i, res in enumerate([1000, 2000, 4000]):
    ax[0].plot(time[i], 4 ** i * E[i], ls = (12 * i, [12,24]), c = colours[i], label = "$ " + (str(4 ** i) if 2 ** i != 1 else "") + " E_{ " + str(10 / res) + " }$")
    ax[1].semilogy(time[i], np.abs(4 ** i * E[i]), ls = (12 * i, [12,24]), c = colours[i], label = "$ " + (str(4 ** i) if 2 ** i != 1 else "") + " E_{ " + str(10 / res) + " }$")
ax[0].legend(handlelength = 5)
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$|\\nu^n - u(t^n)|$")
ax[1].legend(handlelength = 5)
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("$|\\nu^n - u(t^n)|$")
fig.savefig("figs/2b.pdf")

V_r = (solution[-2] - 2 **2 * solution[-1][::2]) / (1 - 2 ** 2)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(time[-2], V_r)
fig.savefig("vrRK2.pdf")

fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].semilogy(time[-2], np.abs(E[-2]), lw = 8, label = "$E^n_{0.05}$", c = 'grey')
ax[0].semilogy(time[-2], np.abs(V_r - solution[-2]), lw = 2, label = "$\\tilde E^n_{0.05}$", c = "crimson")
ax[1].plot(time[-2], (1000 * (V_r - solution[-2])/(1000 * E[-2])), label = "$\\tilde E^n_{0.05} / E^n_{0.05}$", c = "darkseagreen", lw = 2)
ax[1].set_ylim([0.5, 1.5])
ax[0].legend()
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$|\\nu^n - u(t^n)|$")
ax[1].legend()
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("$|\\nu^n - u(t^n)|$")
fig.savefig("figs/2c.pdf")


#V_r = (solution[-2] - 2**2 * solution[-1][::2]) / (1 - 2**2)
#fig, ax = plt.subplots(figsize=(10,5))
#ax.plot(time[-2], V_r)
#fig.savefig("2b.pdf")
################################################################################
#3.

def derivative(f, h):
    return np.concatenate(([(-25 * f[0] / 12 + 4 * f[1] - 3 * f[2] + 4 * f[3] / 3 - f[4] / 4) / h, 
              (-25 * f[1] / 12 + 4 * f[2] - 3 * f[3] + 4 * f[4] / 3 - f[5] / 4) / h], 
              (f[:-4]/12 - 2 * f[1:-3] / 3 + 2 * f[3:-1] / 3 - f[4:] / 12) / h, 
              [(25 * f[-2] / 12 - 4 * f[-3] + 3 * f[-4] - 4 * f[-5] / 3 + f[-6] / 4) / h, 
               (25 * f[-1] / 12 - 4 * f[-2] + 3 * f[-3] - 4 * f[-4] / 3 + f[-5] / 4) / h]))

fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot(time[-2], 1.5e5 * E[-2], label = "$E_{0.05}$", c = "salmon")
ax.plot(time[-2], derivative(V_r, 10. / 2000.), label = "$\\frac{d}{d t}\\nu_{R_{0.05}}$", c = "dodgerblue")
ax.legend()
ax.set_xlabel("$t$")
ax.set_ylabel("$|\\nu^n - u(t^n)|$")
fig.savefig("figs/3.pdf")
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
ax.loglog(dtEuler, L2Euler, 'o-', label = 'Euler', c = "slateblue")
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
ax.loglog(dtRK2, L2RK2, 'o-', label = 'Runge-Kutta 2', c = "springgreen")
ax.legend()
ax.set_xlabel("$\Delta t$", fontsize = 20)
ax.set_ylabel("$\\norm{E_{\Delta t}}$", fontsize = 20)
fig.savefig("figs/4.pdf")
