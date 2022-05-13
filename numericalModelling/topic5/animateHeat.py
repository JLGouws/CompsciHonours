import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from matplotlib.animation import FuncAnimation


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

def setAdvection(k, dx):
    def advection(t, u):
        ux = derivative(u, dx)
        return k * ux 
    return advection

def setHeat(k, dx):
    def heat(t, u):
        uxx = derivative(derivative(u, dx), dx)
        return k * uxx
    return heat

def setWave(k, dx):
    def wave(t, U):
        uxx = derivative(derivative(U[0], dx), dx)
        return np.array([U[1], 1 / k * uxx])
    return wave

def setWavePeriodic(k, dx):
    def wave(t, U):
        uxx = derivativePeriodic(derivativePeriodic(U[0], dx), dx)
        return np.array([U[1], 1 / k * uxx])
    return wave

def setWavePeriodic2D(k, dx, dy):
    def wave(t, U):
        uxx = xDerivativePeriodic(xDerivativePeriodic(U[0], dx), dx)
        uyy = yDerivativePeriodic(yDerivativePeriodic(U[0], dy), dy)
        return np.array([U[1], k * (uxx + uyy)])
    return wave

def setHeatPeriodic2D(k, dx, dy):
    def wave(t, U):
        uxx = xDerivativePeriodic(xDerivativePeriodic(U, dx), dx)
        uyy = yDerivativePeriodic(yDerivativePeriodic(U, dy), dy)
        return k * (uxx + uyy)
    return wave

def derivative(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(-3 * fx[0] + 4 * fx[1] - fx[2]) * .5 / h], 
        (fx[2:] - fx[:-2]) * .5 / h, [(3 * fx[-1] - 4 * fx[-2] + fx[-3]) * .5 / h]))

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

#def evolvePde2nd(F, u0, udot0, x, ti, tf, dt):
#    t = [ti]
#    solution = [u0]
#    ti, y = runge_kutta4step(lambda t, x: udot0, ti, u0, dt)
#    t += ti
#    solution += y
#    while t[-1] <= tf:
#        ti, q = runge_kutta4step(F, t[-1], solution[-1], dt)
#        ti, y = runge_kutta4step(lambda t, x: q[0], t[-1], solution[-1], dt)
#        solution += y
#        t += ti
#    return np.array(solution), np.array(t)

#
#x = np.linspace(- 2.5 * np.pi, 1.5 * np.pi, 1000)
#dx = x[1] - x[0]
#dt = dx / 200
#fx = np.exp(- 0.5 * x ** 2)
#
#advectionEquation = setAdvection(2, dx)
#
#solution, t = evolvePde(advectionEquation, fx, x, 0, 50000 * dt, dt)
#
#fig, ax = plt.subplots(1)
#ax.plot(x, solution[0])
#ax.plot(x, solution[25000])
#ax.plot(x, solution[50000])
#fig.savefig("advectionSteps.pdf")
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
#X, Y = np.meshgrid(x, t)
#
#surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter('{x:.02f}')
#
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#fig.savefig("advection.pdf")
#
#x = np.linspace(- 2 * np.pi, 2 * np.pi, 1000)
#dx = x[1] - x[0]
#dt = dx / 300
#fx = np.exp(- 0.5 * x ** 2)
#
#heatEquation = setHeat(3, dx)
#
#solution, t = evolvePde(heatEquation, fx, x, 0, 50000 * dt, dt)
#
#fig, ax = plt.subplots(1)
#ax.plot(x, solution[0])
#ax.plot(x, solution[25000])
#ax.plot(x, solution[50000])
#fig.savefig("heatSteps.pdf")
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
#X, Y = np.meshgrid(x, t)
#
#surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter('{x:.02f}')
#
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#fig.savefig("heat.pdf")
#
#x = np.linspace(- 2 * np.pi, 2 * np.pi, 1000)
#dx = x[1] - x[0]
#dt = dx / 300
#fx = np.exp(- x ** 2)
#gx = np.zeros_like(x)
#
#waveEquation = setWave(0.5, dx)
#
#solution, t = evolvePde(waveEquation, np.array([fx, gx]), x, 0, 50000 * dt, dt)
#
#solution = solution[:,0,:]
#
#fig, ax = plt.subplots(1)
#ax.plot(x, solution[0])
#ax.plot(x, solution[50000])
##ax.plot(x, solution[50000])
#fig.savefig("waveSteps.pdf")
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
#X, Y = np.meshgrid(x, t)
#
#surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter('{x:.02f}')
#
#fig.colorbar(surf, shrink=0.5, aspect=5)
##plt.show()
#fig.savefig("waveEq.pdf")
#
#x = np.linspace(- 4 * np.pi, 4 * np.pi, 1000)
#dx = x[1] - x[0]
#dt = dx / 400
#fx = np.exp(- x ** 2)
#gx = np.cos(x)
#
#waveEquation = setWavePeriodic(0.25, dx)
#
#solution, t = evolvePde(waveEquation, np.array([fx, gx]), x, 0, 50000 * dt, dt)
#
#solution = solution[:,0,:]
#
#fig, ax = plt.subplots(1)
#ax.plot(x, solution[0])
#ax.plot(x, solution[50000])
##ax.plot(x, solution[50000])
#fig.savefig("waveStepsPeriodic.pdf")
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
#X, Y = np.meshgrid(x, t)
#
#surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter('{x:.02f}')
#
#fig.colorbar(surf, shrink=0.5, aspect=5)
##plt.show()
#fig.savefig("waveEqPeriodic.pdf")
#
x = np.linspace(- np.pi, np.pi, 100)
y = np.linspace(- np.pi, np.pi, 100)
dx = x[1] - x[0]
dy = y[1] - y[0]
x, y = np.meshgrid(x, y)
dt = dx / 50
fx = np.exp(- x ** 2 - y ** 2)

heatEquation = setHeatPeriodic2D(2, dx, dy)

solution, t = evolvePde(heatEquation, fx, x, 0, 900 * dt, dt)
#
#fig, ax = plt.subplots(1)
#ax.plot(x, solution[0])
#ax.plot(x, solution[50000])
##ax.plot(x, solution[50000])
#fig.savefig("waveStepsPeriodic.pdf")
#

fps = 1000

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_zlim(-1, 1);

def update(frame, solution, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, solution[frame], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, vmin = -0.25, vmax = 1)

plot = [ax.plot_surface(x, y, solution[0], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, vmin = -0.25, vmax = 1)]

ani = FuncAnimation(fig, update, len(t), fargs=(solution, plot),
                    interval=1000/fps)
plt.show()
#fn = 'waveEquation'
#ani.save(fn+'.mp4',writer='ffmpeg',fps=60)
#ani.save(fn+'.gif',writer='imagemagick',fps=60)
#fig.savefig("waveEqPeriodic.pdf")
#
