import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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

def runge_kutta4(f, a, b, w, h = 0.1, N = None):
  if N is not None:
    h = (b - a) / N
  else:
    N = round((b - a) / h)
  t, Y = [a], [w]
  for i in range(1, N + 1):
    K1 = h * f(t[-1], Y[-1])
    K2 = h * f(t[-1] + 0.5 * h, Y[-1] + K1 / 2)
    K3 = h * f(t[-1] + 0.5 * h, Y[-1] + K2 / 2)
    K4 = h * f(t[-1] + h, Y[-1] + K3)
    Y += [Y[-1] + (K1 + 2 * K2 + 2 * K3 + K4) / 6]
    t += [t[-1] + h]
  return t, Y

logistic = lambda t, u : 1 * u * (1 - u)

analytic = lambda t : 0.05 * np.exp(t) / (0.05  * (np.exp(t) - 1) + 1)

################################################################################
#1a.
def lotka_volterra(a):
    def F (t, x):
        return np.array([0.8 * x[0] + a * x[0] ** 2 - x[0] * x[1], -x[1] + x[0] * x[1]])
    return F

fig, ax = plt.subplots(figsize=(10,10))
points = runge_kutta4(logistic, 0, 25, 0.05, h = 0.05)
ax.plot(points[0], points[1])
fig.savefig('figs/logistic.pdf')

solution = []
time = []
solution2 = []
time2 = []
Lt = []
L0 = []
Lt2 = []
L02 = []
fig, ax = plt.subplots(1, 2, figsize=(11,5), tight_layout = True)
res = 0.05
colours = ["indianred", "sandybrown", "gold", "yellowgreen", "deepskyblue", "blueviolet", "deeppink"]
for i, x0 in enumerate([(1,4), (1,3), (1,2), (1, 1.5), (1, 1.2), (1, 1), (1, 0.8)]):
    points = runge_kutta4(lotka_volterra(0), 0, 25, np.array(x0), h = res)
    time += [np.array(points[0])]
    solution += [np.array(points[1])]
    L0 += [x0[0] - np.log(x0[0]) + x0[1] - 0.8 * np.log(x0[1])]
    Lt += [solution[-1][:,0] - np.log(solution[-1][:,0]) + solution[-1][:,1] - 0.8 * np.log(solution[-1][:,1])]
    ax[0].plot(solution[-1][:, 0], solution[-1][:, 1])
    points = runge_kutta4(lotka_volterra(0), 0, 25, np.array(x0), h = res / 2)
    time2 += [np.array(points[0])]
    solution2 += [np.array(points[1])]
    L02 += [x0[0] - np.log(x0[0]) + x0[1] - 0.8 * np.log(x0[1])]
    Lt2 += [solution2[-1][:,0] - np.log(solution2[-1][:,0]) + solution2[-1][:,1] - 0.8 * np.log(solution2[-1][:,1])]
    ax[0].plot(solution[-1][:, 0], solution[-1][:, 1], c = colours[i], label = "$\\vec{x}_0 = ( "  + str(x0[0]) + ", " + str(x0[1]) + ")$")
    ax[0].scatter(x0[0], x0[1], c = colours[i], s = 13)
ax[0].legend()
ax[0].set_xlabel("$x$", fontsize = 15)
ax[0].set_ylabel("$y$", fontsize = 15)
ax[1].plot(time[2], solution[2][:, 0], c = "tan", label = "x")
ax[1].plot(time[2], solution[2][:, 1], '--', c = "crimson", label = "y")
ax[1].legend()
ax[1].set_ylabel("population", fontsize = 15)
ax[1].set_xlabel("$t$", fontsize = 15)
fig.savefig('figs/Lotka_Volterra.pdf')

#fig, ax = plt.subplots(len(Lt) // 2 + 1, 2, figsize=(20,5 * len(Lt)), tight_layout = True)
fig = plt.figure(figsize=(10,int(2.5 * len(Lt))), tight_layout = True)
gs = GridSpec(len(Lt) // 2 + 1, 4, figure = fig)
x0s = [(1,4), (1,3), (1,2), (1, 1.5), (1, 1.2), (1, 1), (1, 0.8)]
for i, L in enumerate(zip(Lt, L0)):
#    ax = fig.add_subplot(len(Lt) // 2 + 1, 2, i + 1);
    if i != len(Lt) - 1:
        ax = fig.add_subplot(gs[i // 2, 2 * (i % 2 != 0): 2 + 2 * (i % 2 != 0)]);
    else:
        ax = fig.add_subplot(gs[i // 2, 2 - (i % 2 == 0):4 - (i % 2 == 0)]);

    ax.plot(time[i], L[0] - L[1], c = 'teal', label = "$L(t) - L(0)$, full time step")#[i % (len(Lt) // 2 + 1)][i // (len(Lt) // 2 + 1)].plot(time[i], L[0] - L[1])
    ax.plot(time2[i], Lt2[i] - L02[i], c = 'olivedrab', label = "$L(t) - L(0)$, half time step")#[i % (len(Lt) // 2 + 1)][i // (len(Lt) // 2 + 1)].plot(time2[i], Lt2[i] - L02[i])
    ax.plot(time2[i], 2**4 * (Lt2[i] - L02[i]), linestyle = '--', c = 'red', dashes = (3,5), label = "$16 \\times [L(t) - L(0)]$, half time step")
    ax.set_title(f"$\\vec{{x}}_0 = ({x0s[i][0]}, {x0s[i][1]})$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$L(t) - L(0)$")
    if i == 1:
        ax.legend()
fig.savefig('figs/LVError.pdf')

solution = []
time = []
Lt = []
L0 = []
fig, ax = plt.subplots(1, 2, figsize=(10,5), tight_layout = True)
colours = ["mediumaquamarine", "sandybrown", "gold", "yellowgreen", "deepskyblue", "blueviolet", "deeppink"]
for i, x0 in enumerate([(1,2)]):
    points = runge_kutta4(lotka_volterra(0.1), 0, 25, np.array(x0), h = 0.05)
    time += [np.array(points[0])]
    solution += [np.array(points[1])]
    L0 += [x0[0] - np.log(x0[0]) + x0[1] - 0.8 * np.log(x0[1])]
    Lt += [solution[-1][:,0] - np.log(solution[-1][:,0]) + solution[-1][:,1] - 0.8 * np.log(solution[-1][:,1])]
#    print(solution[-1].shape)
    ax[0].plot(solution[-1][:, 0], solution[-1][:, 1], c = colours[i], label = f"$\\vec{{x}}_0 = ({x0[0]}, {x0[1]})$")
    ax[0].scatter(x0[0], x0[1], c = colours[i], s = 13)
ax[0].legend()
ax[0].set_xlabel("$x$", fontsize = 15)
ax[0].set_ylabel("$y$", fontsize = 15)
ax[1].plot(time[0], solution[0][:, 0], c = "orchid", label = "x")
ax[1].plot(time[0], solution[0][:, 1], '--', c = "darkcyan", label = "x")
ax[1].legend()
ax[1].set_ylabel("population", fontsize = 15)
ax[1].set_xlabel("$t$", fontsize = 15)
fig.savefig('figs/Lotka_Volterra0.1.pdf')

################################################################################
#2.

def SIR(beta, gamma, sigma):
    def F(t, x):
        return np.array([-beta * x[0] * x[1] + sigma * x[2], beta * x[0] * x[1] - gamma * x[1], gamma * x[1] - sigma * x[2]])
    return F

def findLess(a, val, start):
    for i, e in enumerate(a):
        if i < start: continue
        if e < val: return i
    return -1

fig, ax = plt.subplots(1, 1, figsize=(5,5))
points = runge_kutta4(SIR(0.8, 1 / 3, 0), 0, 110, np.array([(8e6 - 1)/ 8e6, 1 / 8e6, 0]), h = 0.05)
timeSIR = np.array(points[0])
solutionSIR = np.array(points[1])
maxI = np.argmax(solutionSIR[:, 1])
end = findLess(solutionSIR[:, 1], 1 / 8e6, maxI)
f = open("writeup/2ai.tex", "w")
f.write(f"$t = {timeSIR[maxI] : .04f}$")
f.close()
f = open("writeup/2aii.tex", "w")
f.write(f"$\max\limits_{{t}}{{I(t)}} = {round(solutionSIR[:, 1][maxI] * 8e6)}$")
f.close()
f = open("writeup/2aiii.tex", "w")
f.write(f"$t = {timeSIR[end] : .01f}$")
f.close()
f = open("writeup/2aiiii.tex", "w")
f.write(f"$p = {1 - solutionSIR[end, 0]: .01f}$")
f.close()
ax.plot(timeSIR, solutionSIR[:, 0], label = 'S', c = "orange", alpha = 0.8)
ax.plot(timeSIR, solutionSIR[:, 1], label = 'I', c = "red", alpha = 0.7)
ax.plot(timeSIR, solutionSIR[:, 2], label = 'R', c = "lime", alpha = 0.6)
ax.set_xlabel("$t$")
ax.set_ylabel("population fraction")
#ax.text(y = 0, x = timeSIR[maxI], s = str(timeSIR[maxI]))
ax.legend()
fig.savefig('figs/SIR.pdf')

fig, ax = plt.subplots(1, 1, figsize=(5,5))
points = runge_kutta4(SIR(0.4, 1 / 3, 0), 0, 420, np.array([(8e6 - 1)/ 8e6, 1 / 8e6, 0]), h = 0.05)
timeSIR = np.array(points[0])
solutionSIR = np.array(points[1])
maxI = np.argmax(solutionSIR[:, 1])
end = findLess(solutionSIR[:, 1], 1 / 8e6, maxI)
f = open("writeup/2bi.tex", "w")
f.write(f"$t = {timeSIR[maxI] : .04f}$")
f.close()
f = open("writeup/2bii.tex", "w")
f.write(f"$\max\limits_{{t}}{{I(t)}} = {round(solutionSIR[:, 1][maxI] * 8e6)}$")
f.close()
f = open("writeup/2biii.tex", "w")
f.write(f"$t = {timeSIR[end] : .01f}$")
f.close()
f = open("writeup/2biiii.tex", "w")
f.write(f"$p = {1 - solutionSIR[end, 0]: .01f}$")
f.close()
ax.plot(timeSIR, solutionSIR[:, 0], label = 'S', c = "goldenrod", alpha = 1)
ax.plot(timeSIR, solutionSIR[:, 1], label = 'I', c = "crimson", alpha = 0.8)
ax.plot(timeSIR, solutionSIR[:, 2], label = 'R', c = "forestgreen", alpha = 0.6)
#ax.text(y = 0, x = timeSIR[maxI], s = str(timeSIR[maxI]))
ax.set_xlabel("$t$")
ax.set_ylabel("population fraction")
#ax.text(y = 0, x = timeSIR[maxI], s = str(timeSIR[maxI]))
ax.legend()
fig.savefig('figs/SIRbetaHalved.pdf')


def lorenz(sigma, b, r):
    def F(t, x):
        return np.array([sigma * (x[1] - x[0]), r * x[0]  - x[1] - x[0] * x[2], x[0] * x[1] - b * x[2]])
    return F

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([2, 0, 0]), h = 0.005)
timelorenz = np.array(points[0])
solutionlorenz = np.array(points[1])

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([2.0001, 0, 0]), h = 0.005)
solutionlorenz0001 = np.array(points[1])

fig = plt.figure(figsize=(10,8), constrained_layout = True)
gs = GridSpec(4, 4, figure = fig)
ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')
ax.plot(solutionlorenz[:, 0], solutionlorenz[:, 1], solutionlorenz[:, 2], alpha = 0.7, c = 'maroon', label = "$\\vec{x}_0 = [2, 0, 0]$")
ax.set(xlabel = '$x$', ylabel = '$y$', zlabel = '$z$')
ax.legend()

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([-2, 0, 0]), h = 0.005)
timelorenz2 = np.array(points[0])
solutionlorenz2 = np.array(points[1])
ax2 = fig.add_subplot(gs[0:2, 2:4], projection='3d')
ax2.plot(solutionlorenz2[:, 0], solutionlorenz2[:, 1], solutionlorenz2[:, 2], alpha = 0.7, c = 'teal', label = "$\\vec{x}_0 = [-2, 0, 0]$")
ax2.set(xlabel = '$x$', ylabel = '$y$', zlabel = '$z$')
ax2.legend()

ax3 = fig.add_subplot(gs[2:4, 1:3], projection='3d')
ax3.plot(solutionlorenz[:, 0], solutionlorenz[:, 1], solutionlorenz[:, 2], alpha = 0.7, c = 'maroon', label = "$\\vec{x}_0 = [2, 0, 0]$")
ax3.plot(solutionlorenz2[:, 0], solutionlorenz2[:, 1], solutionlorenz2[:, 2], alpha = 0.7, c = 'teal', label = "$\\vec{x}_0 = [-2, 0, 0]$")
ax3.set(xlabel = '$x$', ylabel = '$y$', zlabel = '$z$')
ax3.legend()
fig.savefig('figs/lorenz.pdf')

fig, ax = plt.subplots(1)
ax.plot(timelorenz, solutionlorenz[:, 0], alpha = 0.7, c = 'r', label = "$\\vec{x}_0 = [2, 0, 0]$")
ax.plot(timelorenz, solutionlorenz0001[:, 0], '--',  alpha = 0.7, c = 'b', label = "$\\vec{x}_0 = [2.001, 0, 0]$")
ax.set_xlabel("$t$")
ax.set_ylabel("$x(t)$")
ax.legend()
fig.savefig('figs/xvtLorenz.pdf')

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([2, 0, 0]), h = 0.01)
timelorenzdt01 = np.array(points[0])
solutionlorenzdt01 = np.array(points[1])

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([2, 0, 0]), h = 0.02)
timelorenzdt02 = np.array(points[0])
solutionlorenzdt02 = np.array(points[1])

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([2, 0, 0]), h = 0.005)
timelorenzdt005 = np.array(points[0])
solutionlorenzdt005 = np.array(points[1])

ax = [] 
fig = plt.figure(figsize=(10.7,10), tight_layout = True)
gs = GridSpec(4, 4, figure = fig)
ax += [fig.add_subplot(gs[0:2, 1:3])]
ax += [fig.add_subplot(gs[2:4, 0:2])]
ax += [fig.add_subplot(gs[2:4, 2:4])]
ax[0].plot(timelorenzdt02, solutionlorenzdt02[:, 0], '--',  alpha = 0.7, c = 'g', label = '$\Delta t = 0.02$')
ax[0].plot(timelorenzdt01, solutionlorenzdt01[:, 0], alpha = 0.7, c = 'r', label = '$\Delta t = 0.01$')
ax[0].plot(timelorenzdt005, solutionlorenzdt005[:, 0], '--',  alpha = 0.7, c = 'b', label = '$\Delta t = 0.005$')
ax[0].set_title("$x(t)$ for different time steps")
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$x(t)$")
ax[0].legend()
ax[1].semilogy(timelorenzdt02, np.abs(solutionlorenzdt01[:, 0][::2] - solutionlorenzdt02[:, 0]),alpha = 0.8, c = 'orange', label = '$|x_{0.01} - x_{0.02}|$')
ax[1].semilogy(timelorenzdt01, np.abs(solutionlorenzdt01[:, 0] - solutionlorenzdt005[:, 0][::2]), '--',  alpha = 0.7, c = 'magenta', label = '$|x_{0.01} - x_{0.05}|$')
ax[1].semilogy(timelorenzdt02, np.abs(solutionlorenzdt02[:, 0] - solutionlorenzdt005[:, 0][::4]), '--',  alpha = 0.7, c = 'cyan', label = '$|x_{0.02} - x_{0.05}|$')
ax[1].set_title("Log of absolute difference in $x(t)$ for different time steps")
ax[1].legend()
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("$|x_{res_1} - x_{res_2}|$")
ax[2].semilogy(timelorenzdt02, np.abs(solutionlorenzdt01[:, 1][::2] - solutionlorenzdt02[:, 1]), alpha = 0.8, c = 'orange', label = '$|y_{0.01} - y_{0.02}|$')
ax[2].semilogy(timelorenzdt01, np.abs(solutionlorenzdt01[:, 1] - solutionlorenzdt005[:, 1][::2]), '--',  alpha = 0.7, c = 'magenta', label = '$|y_{0.01} - y_{0.05}|$')
ax[2].semilogy(timelorenzdt02, np.abs(solutionlorenzdt02[:, 1] - solutionlorenzdt005[:, 1][::4]), '--',  alpha = 0.7, c = 'cyan', label = '$|y_{0.02} - y_{0.05}|$')
ax[2].set_title("Log of absolute difference in $y(t)$ for different time steps")
ax[2].set_xlabel("$t$")
ax[2].set_ylabel("$|y_{res_1} - y_{res_2}|$")
ax[2].legend()
fig.savefig('figs/lorenzChaosTimeStep.pdf')
