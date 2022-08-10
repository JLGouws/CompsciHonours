import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation

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

def runge_kutta4step(f, t, w, h = 0.1):
  K1 = h * f(t, w)
  K2 = h * f(t + 0.5 * h, w + K1 / 2)
  K3 = h * f(t + 0.5 * h, w + K2 / 2)
  K4 = h * f(t + h, w + K3)
  return [t + h], [w + (K1 + 2 * K2 + 2 * K3 + K4) / 6]

def runge_kutta4stepDirichlet(f, t, w, h = 0.1, a = 0, b = 0):
  K1 = h * f(t, w)
  K2 = h * f(t + 0.5 * h, w + K1 / 2)
  K3 = h * f(t + 0.5 * h, w + K2 / 2)
  K4 = h * f(t + h, w + K3)
  return [t + h], [np.concatenate(([a], (w + (K1 + 2 * K2 + 2 * K3 + K4) / 6)[1:-1], [b]))]

def taylor_step(f, t, w, h = 0.1):
  return [t + h], [w + h * f(t, w)]

def taylor_stepDirichlet(f, t, w, h = 0.1, a = 0, b = 0):
  return [t + h], [np.concatenate(([a], (w + h * f(t, w))[1:-1], [b]))]


def setAdvection(k, dx):
    def advection(t, u):
        ux = derivative(u, dx)
        return k * ux 
    return advection

def setAdvectionPeriodic(k, dx):
    def advection(t, u):
        ux = derivativePeriodic(u, dx)
        return k * ux 
    return advection

def setHeat(k, dx):
    def heat(t, u):
        uxx = derivative(derivative(u, dx), dx)
        return k * uxx
    return heat

def setHeatDirichlet(k, dx):
    def heat(t, u):
        u[0], u[-1] = 0, 0
        uxx = secondDerivativePeriodic(u, dx)
        return k * uxx
    return heat

def setWave(k, dx):
    def wave(t, U):
        uxx = derivative(derivative(U[0], dx), dx)
        return np.array([U[1], 1 / k * uxx])
    return wave

def setWavePeriodic(k, dx):
    def wave(t, U):
        q1 = U[0]
        q2 = U[1]
        dq1 = derivative(q1, dx)
        dq2 = derivative(q2, dx)
        dq2[0] = (q2[1] + q1[1]) * .5 / dx
        dq1[-1] = (-q2[-2] - q1[-2]) * .5 / dx
        return np.array([k * dq1, - k * dq2])
    return wave

def setWavePeriodic2D(k, dx, dy):
    def wave(t, U):
        uxx = xDerivativePeriodic(xDerivativePeriodic(U[0], dx), dx)
        uyy = yDerivativePeriodic(yDerivativePeriodic(U[0], dy), dy)
        return np.array([U[1], k * (uxx + uyy)])
    return wave

def derivative(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(-3 * fx[0] + 4 * fx[1] - fx[2]) * .5 / h], 
        (fx[2:] - fx[:-2]) * .5 / h, [(3 * fx[-1] - 4 * fx[-2] + fx[-3]) * .5 / h]))

def secondDerivativePeriodic(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(fx[-2] - 2 * fx[0] + fx[1]) / h**2], 
        (fx[2:] - 2 * fx[1:-1] + fx[:-2])/ h**2, [(fx[-2] - 2 * fx[0] + fx[1]) / h**2]))

def derivativePeriodic(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(fx[1] - fx[-2]) * .5 / h], 
        (fx[2:] - fx[:-2]) * .5 / h, [(fx[1] - fx[-2]) * .5 / h]))

def xDerivativePeriodic(fx, h):
    fx = np.array(fx)
    return np.concatenate((np.array([(fx[1,:] - fx[-2,:]) * .5 / h]), 
        (fx[2:,:] - fx[:-2,:]) * .5 / h, np.array([(fx[1,:] - fx[-2,:]) * .5 / h])))

def yDerivativePeriodic(fx, h):
    fx = np.array(fx)
    #print(np.array([(fx[:,1] - fx[:,:-2]) * .5 / h]).T)
    return np.concatenate((np.array([(fx[:,1] - fx[:,-2]) * .5 / h]).T, 
        (fx[:,2:] - fx[:,:-2]) * .5 / h, np.array([(fx[:,1] - fx[:,-2]) * .5 / h]).T),
        axis = 1)

def derivative2(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(fx[-1] - 2 * fx[0] + fx[1]) / h**2], 
        (fx[2:] - 2 * fx[1:-1] + fx[:-2]) / h**2, [(fx[-2] - 2 * fx[-1] + fx[0]) / h**2]))

def evolvePde(F, u0, x, ti, tf, dt):
    t = [ti]
    solution = [u0]
    while t[-1] <= tf:
        ti, y = runge_kutta4step(F, t[-1], solution[-1], dt)
        solution += y
        t += ti
    return np.array(solution), np.array(t)

def evolvePdeDirichlet(F, u0, x, ti, tf, dt):
    t = [ti]
    solution = [u0]
    while t[-1] <= tf:
        ti, y = runge_kutta4stepDirichlet(F, t[-1], solution[-1], dt, 0, 0)
        solution += y
        t += ti
    return np.array(solution), np.array(t)

def evolvePdeEuler(F, u0, x, ti, tf, dt):
    t = [ti]
    solution = [u0]
    while t[-1] <= tf:
        ti, y = taylor_step(F, t[-1], solution[-1], dt)
        solution += y
        t += ti
    return np.array(solution), np.array(t)

def evolvePdeEulerDirichlet(F, u0, x, ti, tf, dt, a = 0, b = 0):
    t = [ti]
    solution = [u0]
    while t[-1] <= tf:
        ti, y = taylor_stepDirichlet(F, t[-1], solution[-1], dt, a, b)
        solution += y
        t += ti
    return np.array(solution), np.array(t)

x = np.linspace(- np.pi, np.pi, 101)
dx = x[1] - x[0]
dt = dx
fx = np.cos(x / 2) ** 8
fx[0], fx[-1] = 0, 0

heatEquation = setHeatDirichlet(1, dx)

solution, t = evolvePdeEulerDirichlet(heatEquation, fx, x, 0, np.pi / 2, dt)

fig, ax = plt.subplots(1)                         
ax.plot(x, solution[-1], c = "brown")
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,t)$")
fig.savefig("figs/heatEulerdx.pdf")

x = np.linspace(- np.pi, np.pi, 101)
dx = x[1] - x[0]
dt = 0.01 * dx
fx = np.cos(x / 2) ** 8
fx[0], fx[-1] = 0, 0

heatEquation = setHeatDirichlet(1, dx)

solution, t = evolvePdeEulerDirichlet(heatEquation, fx, x, 0, 8 + dt, dt)

fig, ax = plt.subplots(1)                         
ax.plot(x, solution[0], label = f"$t = {t[0]:.04f}$", c = "firebrick")
ax.plot(x, solution[round(0.1 / dt)], label = f"$t = {t[round(0.1 / dt)]:.04f}$", c = "salmon")
ax.plot(x, solution[round(0.5 / dt)], label = f"$t = {t[round(0.5 / dt)]:.04f}$", c = "peru")
ax.plot(x, solution[round(1 / dt)]  , label = f"$t = {t[round(1 / dt)]:.04f}$"  , c = "darkorange")
ax.plot(x, solution[round(2 / dt)]  , label = f"$t = {t[round(2 / dt)]:.04f}$"  , c = "goldenrod")
ax.plot(x, solution[round(4 / dt)]  , label = f"$t = {t[round(4 / dt)]:.04f}$"  , c = "seagreen")
ax.plot(x, solution[round(8 / dt)]  , label = f"$t = {t[round(8 / dt)]:.04f}$"  , c = "deepskyblue")
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,t)$")
fig.savefig("figs/heatEuler01dx.pdf")

x = np.linspace(- np.pi, np.pi, 101)
dx = x[1] - x[0]
mulFac = 0.03153
dt = mulFac * dx
fx = np.cos(x / 2) ** 8
fx[0], fx[-1] = 0, 0

heatEquation = setHeatDirichlet(1, dx)

solution, t = evolvePdeEulerDirichlet(heatEquation, fx, x, 0, 8 + dt, dt)

fig, ax = plt.subplots(1)                         
ax.text(x = -3, y = 1, s = "$\Delta t = " + str(mulFac) + "\Delta x $")
ax.plot(x, solution[0], label = f"$t = {t[0]:.04f}$", c = "red")
ax.plot(x, solution[round(0.1 / dt)], label = f"$t = {t[round(0.1 / dt)]:.04f}$", c = "saddlebrown")
ax.plot(x, solution[round(0.5 / dt)], label = f"$t = {t[round(0.5 / dt)]:.04f}$", c = "darkgoldenrod")
ax.plot(x, solution[round(1 / dt)]  , label = f"$t = {t[round(1 / dt)]:.04f}$"  , c = "olive")  
ax.plot(x, solution[round(2 / dt)]  , label = f"$t = {t[round(2 / dt)]:.04f}$"  , c = "palegreen")   
ax.plot(x, solution[round(4 / dt)]  , label = f"$t = {t[round(4 / dt)]:.04f}$"  , c = "dodgerblue")    
ax.plot(x, solution[round(8 / dt)]  , label = f"$t = {t[round(8 / dt)]:.04f}$"  , c = "rebeccapurple") 
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,t)$")
ax.legend()
fig.savefig("figs/heatEulerMaxdx.pdf")

x = np.linspace(- np.pi, np.pi, 101)
dx = x[1] - x[0]
mulFac = 0.04386
dt = mulFac * dx
fx = np.cos(x / 2) ** 8
fx[0], fx[-1] = 0, 0

heatEquation = setHeatDirichlet(1, dx)

solution, t = evolvePdeDirichlet(heatEquation, fx, x, 0, 8 + dt, dt)

fig, ax = plt.subplots(1)                         
ax.text(x = -3, y = 1, s = "$\Delta t = " + str(mulFac) + "\Delta x $")
ax.plot(x, solution[0], label = f"$t = {t[0]:.04f}$", c = "maroon")
ax.plot(x, solution[round(0.1 / dt)], label = f"$t = {t[round(0.1 / dt)]:.04f}$", c = "lightcoral")
ax.plot(x, solution[round(0.5 / dt)], label = f"$t = {t[round(0.5 / dt)]:.04f}$", c = "orange")
ax.plot(x, solution[round(1 / dt)]  , label = f"$t = {t[round(1 / dt)]:.04f}$"  , c = "yellowgreen")
ax.plot(x, solution[round(2 / dt)]  , label = f"$t = {t[round(2 / dt)]:.04f}$"  , c = "turquoise")
ax.plot(x, solution[round(4 / dt)]  , label = f"$t = {t[round(4 / dt)]:.04f}$"  , c = "slateblue")
ax.plot(x, solution[round(8 / dt)]  , label = f"$t = {t[round(8 / dt)]:.04f}$"  , c = "fuchsia")
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,t)$")
ax.legend()
fig.savefig("figs/heatRK4Maxdx.pdf")

quit()

fps = 1000
                                                                                
fig, ax = plt.subplots(1)                         
                                                                                
ax.set_ylim(-0.8, 0.8);                                                             
                                                                                
def update(frame, solution, ax):                                              
    ax.clear()
    ax.set_ylim(-0.8, 0.8);                                                             
    ax.plot(x, solution[frame])
                                                                                
ax.plot(x, solution[0])
                                                                                
ani = FuncAnimation(fig, update, len(t), fargs=(solution, ax),                
                    interval=1000/fps)                                          
plt.show()  
quit()
