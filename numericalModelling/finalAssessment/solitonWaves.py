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

def taylor_step(f, t, w, h = 0.1):
  return [t + h], [w + h * f(t, w)]

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

def setAdvectionUpwindPeriodic(k, dx):
    def advection(t, u):
        ux = derivativeUpwindPeriodic(u, dx)
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

def setKdV(dx):
    def KdV(t, u):
        uxxx = thirdDerivativePeriodic(u, dx)
        #uxxx = derivativePeriodic(derivativePeriodic(derivativePeriodic(u, dx), dx), dx)
        ux = derivativePeriodic(u, dx)
        return - 6 * u * ux - uxxx
    return  KdV

def derivative(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(-3 * fx[0] + 4 * fx[1] - fx[2]) * .5 / h], 
        (fx[2:] - fx[:-2]) * .5 / h, [(3 * fx[-1] - 4 * fx[-2] + fx[-3]) * .5 / h]))

def derivativeUpwind(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(-3 * fx[0] + 4 * fx[1] - fx[2]) * .5 / h], 
        (fx[2:] - fx[:-2]) * .5 / h, [(3 * fx[-1] - 4 * fx[-2] + fx[-3]) * .5 / h]))

def derivativePeriodic(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(fx[1] - fx[-2]) * .5 / h], 
        (fx[2:] - fx[:-2]) * .5 / h, [(fx[1] - fx[-2]) * .5 / h]))

def thirdDerivativePeriodic(fx, h):
    fx = np.array(fx)
    return np.concatenate(([(-0.5 * fx[-3] + fx[-2] - fx[1] + 0.5 * fx[2]) / h**3, (-0.5 * fx[-2] + fx[-1] - fx[2] + 0.5 * fx[3]) / h**3], 
        ( - 0.5 * fx[:-4] + fx[1: -3] - fx[3: -1] + 0.5 * fx[4:]) / h ** 3, 
        [(-0.5 * fx[-4] + fx[-3] - fx[0] + 0.5 * fx[1]) / h**3, (-0.5 * fx[-3] + fx[-2] - fx[1] + 0.5 * fx[2]) / h**3]))

def derivativeUpwindPeriodic(fx, h):
    fx = np.array(fx)
    return np.concatenate((
        (fx[1:] - fx[:-1]) / h, [(fx[1] - fx[0]) / h]))

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

def evolvePdeEuler(F, u0, x, ti, tf, dt):
    t = [ti]
    solution = [u0]
    while t[-1] <= tf:
        ti, y = taylor_step(F, t[-1], solution[-1], dt)
        solution += y
        t += ti
    return np.array(solution), np.array(t)

def integrateMass(u, dx):
    u = np.array(u)
    return np.sum(u * dx, axis = 1);

x = np.linspace(- np.pi, np.pi, 100)
dx = x[1] - x[0]
dt = dx / 500
#fx = 2 * np.cosh(x )**(-2)
fx = (12 * (3 + 4 * np.cosh(2 * x) + np.cosh(4 * x)))  / ((3 * np.cosh(x) + np.cosh(3 * x))**2)

KdV = setKdV(dx)

solution, t = evolvePde(KdV, fx, x, 0, 5000 * dt, dt)


fps = 1000

mass = integrateMass(solution, dx)
t = np.array(t)
                                                                                
fig1, ax1 = plt.subplots(1)                         

ax1.set_ylim(0, 18);                                                       

ax1.plot(t, mass)

plt.show()

fig, ax = plt.subplots(1)                         
                                                                                
ax.set_ylim(-2, 10);                                                             
                                                                                
def update(frame, solution, ax):                                              
    if 5 * frame < len(solution):
        ax.clear()
        ax.set_ylim(-2 , 10);                                                       
        ax.plot(x, solution[5 * frame])
                                                                                
ax.plot(x, solution[0])
                                                                                
ani = FuncAnimation(fig, update, len(t), fargs=(solution, ax),                
                    interval=1000/fps)                                          
plt.show()  
quit()

fig, ax = plt.subplots(1)
ax.plot(x, solution[0])
ax.plot(x, solution[10000])
#ax.plot(x, solution[48000])
ax.plot(x, solution[50000])
#ax.plot(x, solution[50000])
fig.savefig("waveStepsPeriodic.pdf")

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(x, t)

surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')

fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
fig.savefig("waveEqPeriodic.pdf")
