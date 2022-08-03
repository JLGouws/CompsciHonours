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

def taylorStepGhost(f, t, w, h = 0.1):
    return [t + h], [w[1:-1] + h * f(t, w)]

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

def set2DHeatNeumann(D, dx, dy, vx1 = 0, vx2 = 0, vy1 = 0, vy2 = 0):
    def heat(t, u):
        uxx = xSecondDerivativeNeuman(u, dx, vx1, vx2)
        uyy = ySecondDerivativeNeuman(u, dy, vy1, vy2)
        return D * (uxx + uyy) 
    return heat

def set1DHeatNeumannGhost(D, dx, vx1 = 0, vx2 = 0):
    def heat(t, u):
        uxx = xSecondDerivativeGhost1D(u, dx, vx1, vx2)
        return D * uxx
    return heat

def set1DHeatNeumann(D, dx, vx1 = 0, vx2 = 0):
    def heat(t, u):
        uxx = xSecondDerivativeNeuman1D(u, dx, vx1, vx2)
        return D * uxx 
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

def xSecondDerivativeGhost1D(fx, h, v1, v2):
    fx = np.array(fx)
    return (fx[:-2] - 2 * fx[1:-1] + fx[2:]) / h ** 2
    
def xSecondDerivativeNeuman1D(fx, h, v1, v2):
    fx = np.array(fx)
    return np.concatenate(([0, (fx[0] / 12 - 2 * fx[1] / 3 + 2 * fx[3] / 3 - fx[4] / 12 - h * v1) * 0.5 / h **2 ], 
        (fx[1:-3] - 2 * fx[2:-2] + fx[3:-1]) / h ** 2, 
        [( h * v2 - fx[-5] / 12 + 2 * fx[-4] / 3 - 2 * fx[-2] / 3 + fx[-1] / 12) * 0.5 / h **2 , 0]))

def xSecondDerivativeNeuman(fx, h, v1, v2):
    fx = np.array(fx)
    return np.concatenate(([np.zeros_like(fx[0,:]), (fx[0,:] / 12 - 2 * fx[1,:] / 3 + 2 * fx[3, :] / 3 - fx[4,:] / 12 - h * v1) * 0.5 / h **2 ], 
        (fx[1:-3,:] - 2 * fx[2:-2,:] + fx[3:-1,:]) / h ** 2, 
        [( h * v2 - fx[-5,:] / 12 + 2 * fx[-4,:] / 3 - 2 * fx[-2, :] / 3 + fx[-1,:] / 12) * 0.5 / h **2 , np.zeros_like(fx[0,:])]))

def ySecondDerivativeNeuman(fx, h, v1, v2):
    fx = np.array(fx)
    return np.concatenate((np.concatenate((np.array([np.zeros_like(fx[:,0])]).T, np.array([fx[:,0] / 12 - 2 * fx[:,1] / 3 + 2 * fx[:,3] / 3 - fx[:,4] / 12 - h * v1]).T * 0.5 / h **2), axis = 1), 
        (fx[:,1:-3] - 2 * fx[:,2:-2] + fx[:,3:-1]) / h ** 2, 
        np.concatenate((np.array([h * v2 - fx[:,-5] / 12 + 2 * fx[:,-4] / 3 - 2 * fx[:,-2] / 3 + fx[:,-1] / 12]).T * 0.5 / h **2 , np.array([np.zeros_like(fx[:,0])]).T), axis = 1)), axis = 1)

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

def evolvePdeGhost(F, u0, x, ti, tf, dt):
    t = [ti]
    u0[0] = u0[1]
    u0[-2] = u0[-1]
    solution = [u0]
    while t[-1] <= tf:
        ti, y = taylorStepGhost(F, t[-1], solution[-1], dt)
        y = [np.concatenate(([y[0][0]], y[0], [y[0][-1]]))]
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

x = np.linspace(-0.005, 1.005, 203)
print(x[:3])
print(x[-3:])
dx = x[1] - x[0]
dt = dx / 400
#fx = 2 * np.cosh(x )**(-2)
sigma = 0.005
fx = 10 * np.exp(- 0.5 * ((x - 0.5) ** 2 ) / sigma)

heat = set1DHeatNeumannGhost(1, dx)

solution, t = evolvePdeGhost(heat, fx, x, 0, 5000 * dt, dt)

fig, ax = plt.subplots(1)

T = []

for s in solution:
    T += [np.sum(s[1:-1]**2) * dx]
T = np.array(T)

ax.semilogy(t, T)
plt.show()
quit()

fps = 1000

fig, ax = plt.subplots(1)
                                                                                
ax.set_ylim(-0.5, 1.5); 


#print(T[-1])
                                                                                
def update(frame, solution, ax):
    ax.clear()
    ax.set_ylim(-0.5 , 1.5);
    ax.plot(x, solution[frame])

ax.plot(x[1:-1], fx[1:-1])

ani = FuncAnimation(fig, update, len(t), fargs=(solution, ax), 
                    interval=1000/fps)                                          
plt.show()  

quit()

fig, ax = plt.subplots(1)
T = []

for s in solution:
    T += [np.sum(s**2, (0, 1)) * dx * dy]
T = np.array(T)

ax.plot(t, T)
plt.show()
print(T[-1])
fig.savefig("figs/T.pdf")
quit()

x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = dx / 1000
#fx = 2 * np.cosh(x )**(-2)
X, Y = np.meshgrid(x, y)
fx = np.exp(- 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / sigma)

heat = set2DHeatNeumann(1, dx, dy)#, 10, 10, 10, 10)

solution, t = evolvePde(heat, fx, x, 0, 0.02, dt)

T = []

for s in solution:
    T += [np.sum(s**2, (0, 1)) * dx * dy]
T = np.array(T)

ax.plot(t, T)

print(T[-1])

plt.show()
