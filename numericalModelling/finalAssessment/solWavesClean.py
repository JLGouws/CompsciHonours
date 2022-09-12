import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def runge_kutta4step(f, t, w, h = 0.1): #rk4 step
  K1 = h * f(t, w)
  K2 = h * f(t + 0.5 * h, w + K1 / 2)
  K3 = h * f(t + 0.5 * h, w + K2 / 2)
  K4 = h * f(t + h, w + K3)
  return [t + h], [w + (K1 + 2 * K2 + 2 * K3 + K4) / 6]

def setKdV(dx):
    def KdV(t, u):
        uxxx = thirdDerivativePeriodic(u, dx)
        ux = derivativePeriodic(u, dx)
        return - 6 * u * ux - uxxx
    return  KdV

def derivativePeriodic(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(fx[1] - fx[-2]) * .5 / h], #first derivative this is 4th order accurate?
        (fx[2:] - fx[:-2]) * .5 / h, [(fx[1] - fx[-2]) * .5 / h]))

def thirdDerivativePeriodic(fx, h):
    fx = np.array(fx)#third derivative excuse the unreadability
    return np.concatenate(([(-0.5 * fx[-3] + fx[-2] - fx[1] + 0.5 * fx[2]) / h**3, (-0.5 * fx[-2] + fx[-1] - fx[2] + 0.5 * fx[3]) / h**3], 
        ( - 0.5 * fx[:-4] + fx[1: -3] - fx[3: -1] + 0.5 * fx[4:]) / h ** 3, 
        [(-0.5 * fx[-4] + fx[-3] - fx[0] + 0.5 * fx[1]) / h**3, (-0.5 * fx[-3] + fx[-2] - fx[1] + 0.5 * fx[2]) / h**3]))

def evolvePde(F, u0, x, ti, tf, dt):
    t = [ti]
    solution = [u0]
    while t[-1] <= tf:
        ti, y = runge_kutta4step(F, t[-1], solution[-1], dt) #step forward in time
        solution += y
        t += ti
    return np.array(solution), np.array(t)

def integrateMass(u, dx): #integrating the mass
    u = np.array(u)
    return np.sum(u * dx, axis = 1);

x = np.linspace(- np.pi, np.pi, 151)
dx = x[1] - x[0]
dt = dx / 800
fx = 2 * np.cosh(x )**(-2) #set up grid and initial conditions

totalTime = dt * 19002

KdV = setKdV(dx)

solution, t = evolvePde(KdV, fx, x, 0, totalTime, dt) #evolve pde


mass = integrateMass(solution, dx)
t = np.array(t)

fig1, ax1 = plt.subplots(1)

ax1.set_ylim(0, 18);
                                                                                
ax1.plot(t, mass, c = "gold", label = "$\Delta t = \\frac{\Delta x}{800}$", ls = (20, [10, 20]))
ax1.set_xlabel("$t$")
ax1.set_ylabel("$M(t)$")


ax1.legend(handlelength = 5)

plt.show()

fig1, ax1 = plt.subplots(4, figsize = (5.7, 12), tight_layout = True)

ax1[0].plot(x, solution[3000], c = "tab:purple")
ax1[0].set_title(f"$t = {t[3000]:.04f}$")
ax1[0].set_xlabel("$x$")
ax1[0].set_ylabel("$u(x,t)$")

ax1[1].plot(x, solution[8000], c = "tab:purple")
ax1[1].set_title(f"$t = {t[8000]:.04f}$")
ax1[1].set_xlabel("$x$")
ax1[1].set_ylabel("$u(x,t)$")

ax1[2].plot(x, solution[13000], c = "tab:purple")
ax1[2].set_title(f"$t = {t[13000]:.04f}$")
ax1[2].set_xlabel("$x$")
ax1[2].set_ylabel("$u(x,t)$")

ax1[3].plot(x, solution[18000], c = "tab:purple")
ax1[3].set_title(f"$t = {t[18000]:.04f}$")
ax1[3].set_xlabel("$x$")
ax1[3].set_ylabel("$u(x,t)$")

plt.show()

x = np.linspace(- np.pi, np.pi, 200)
dx = x[1] - x[0]
dt = dx / 1000

fx = (12 * (3 + 4 * np.cosh(2 * x) + np.cosh(4 * x)))  / ((3 * np.cosh(x) + np.cosh(3 * x))**2)

KdV = setKdV(dx)

solution, t = evolvePde(KdV, fx, x, 0, 19002 * dt, dt)

fps = 1000

mass = integrateMass(solution, dx)
t = np.array(t)
                                                                                
fig1, ax1 = plt.subplots(1)                         

ax1.set_ylim(0, 18);                                                       

ax1.plot(t, mass, c = "darkcyan")
ax1.set_xlabel("$t$")
ax1.set_ylabel("$M(t)$")

plt.show()

fig1, ax1 = plt.subplots(4, figsize = (5.7, 12), tight_layout = True)

ax1[0].plot(x, solution[7000], c = "mediumseagreen")
ax1[0].set_title(f"$t = {t[7000]:.04f}$")
ax1[0].set_xlabel("$x$")
ax1[0].set_ylabel("$u(x,t)$")

ax1[1].plot(x, solution[10100], c = "mediumseagreen")
ax1[1].set_title(f"$t = {t[10100]:.04f}$")
ax1[1].set_xlabel("$x$")
ax1[1].set_ylabel("$u(x,t)$")

ax1[2].plot(x, solution[14500], c = "mediumseagreen")
ax1[2].set_title(f"$t = {t[14500]:.04f}$")
ax1[2].set_xlabel("$x$")
ax1[2].set_ylabel("$u(x,t)$")

ax1[3].plot(x, solution[16000], c = "mediumseagreen")
ax1[3].set_title(f"$t = {t[16000]:.04f}$")
ax1[3].set_xlabel("$x$")
ax1[3].set_ylabel("$u(x,t)$")

plt.show()

quit()
fig, ax = plt.subplots(1)                         
                                                                                
fps = 1000

ax.set_ylim(-2, 10);                                                             
                                                                                
def update(frame, solution, ax):                                              
    if 5 * frame < len(solution):
        ax.clear()
        ax.set_ylim(-2 , 10);                                                       
        ax.plot(x, solution[5 * frame])
        fig.savefig(f"figs/kdvframes2/{frame}.pdf")
                                                                                
ax.plot(x, solution[0])
                                                                                
ani = FuncAnimation(fig, update, len(t), fargs=(solution, ax),                
                    interval=1000/fps)                                          
plt.show()  
quit()
