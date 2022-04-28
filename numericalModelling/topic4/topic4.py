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
fig.savefig('logistic.pdf')

solution = []
time = []
solution2 = []
time2 = []
Lt = []
L0 = []
Lt2 = []
L02 = []
fig, ax = plt.subplots(1, 2, figsize=(10,10))
res = 0.05
for x0 in [(1,4), (1,3), (1,2), (1, 1.5), (1, 1.2), (1, 1), (1, 0.8)]:
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
    ax[0].plot(solution[-1][:, 0], solution[-1][:, 1])
ax[1].plot(time[2], solution[2][:, 0])
ax[1].plot(time[2], solution[2][:, 1], '--')
fig.savefig('Lotka_Volterra.pdf')

fig, ax = plt.subplots(len(Lt), 1, figsize=(10,5 * len(Lt)))
for i, L in enumerate(zip(Lt, L0)):
    ax[i].plot(time[i], L[0] - L[1])
    ax[i].plot(time2[i], Lt2[i] - L02[i])
fig.savefig('error.pdf')

solution = []
time = []
Lt = []
L0 = []
fig, ax = plt.subplots(1, 2, figsize=(10,10))
for x0 in [(1,2)]:
    points = runge_kutta4(lotka_volterra(0.1), 0, 25, np.array(x0), h = 0.05)
    time += [np.array(points[0])]
    solution += [np.array(points[1])]
    L0 += [x0[0] - np.log(x0[0]) + x0[1] - 0.8 * np.log(x0[1])]
    Lt += [solution[-1][:,0] - np.log(solution[-1][:,0]) + solution[-1][:,1] - 0.8 * np.log(solution[-1][:,1])]
#    print(solution[-1].shape)
    ax[0].plot(solution[-1][:, 0], solution[-1][:, 1])
ax[1].plot(time[0], solution[0][:, 0])
ax[1].plot(time[0], solution[0][:, 1], '--')
fig.savefig('Lotka_Volterra0.1.pdf')

################################################################################
#2.

def SIR(beta, gamma, sigma):
    def F(t, x):
        return np.array([-beta * x[0] * x[1] + sigma * x[2], beta * x[0] * x[1] - gamma * x[1], gamma * x[1] - sigma * x[2]])
    return F

fig, ax = plt.subplots(1, 1, figsize=(10,10))
points = runge_kutta4(SIR(0.8, 1 / 3, 0), 0, 100, np.array([(8e6 - 1)/ 8e6, 1 / 8e6, 0]), h = 0.05)
timeSIR = np.array(points[0])
solutionSIR = np.array(points[1])
ax.plot(timeSIR, solutionSIR[:, 0], label = 'S')
ax.plot(timeSIR, solutionSIR[:, 1], label = 'I')
ax.plot(timeSIR, solutionSIR[:, 2], label = 'R')
ax.legend()
fig.savefig('SIR.pdf')


def lorenz(sigma, b, r):
    def F(t, x):
        return np.array([sigma * (x[1] - x[0]), r * x[0]  - x[1] - x[0] * x[2], x[0] * x[1] - b * x[2]])
    return F

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([2, 0, 0]), h = 0.005)
timelorenz = np.array(points[0])
solutionlorenz = np.array(points[1])
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot(solutionlorenz[:, 0], solutionlorenz[:, 1], solutionlorenz[:, 2])

points = runge_kutta4(lorenz(10, 8 / 3, 28), 0, 40, np.array([-2, 0, 0]), h = 0.005)
timelorenz2 = np.array(points[0])
solutionlorenz2 = np.array(points[1])
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot(solutionlorenz2[:, 0], solutionlorenz2[:, 1], solutionlorenz2[:, 2])

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot(solutionlorenz[:, 0], solutionlorenz[:, 1], solutionlorenz[:, 2], alpha = 0.6)
ax3.plot(solutionlorenz2[:, 0], solutionlorenz2[:, 1], solutionlorenz2[:, 2], alpha = 0.6)
fig.savefig('lorenz.pdf')
