import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

def set2DHeatNeumann(D, dx, dy, vx1 = 0, vx2 = 0, vy1 = 0, vy2 = 0):
    def heat(t, u):
        uxx = xSecondDerivativeNeuman(u, dx, vx1, vx2)
        uyy = ySecondDerivativeNeuman(u, dy, vy1, vy2)
        return D * (uxx + uyy) 
    return heat

def set2DHeatNeumannSecond(D, dx, dy, vx1 = 0, vx2 = 0, vy1 = 0, vy2 = 0):
    def heat(t, u):
        uxx = xSecondDerivativeNeumanSecond(u, dx, vx1, vx2)
        uyy = ySecondDerivativeNeumanSecond(u, dy, vy1, vy2)
        return D * (uxx + uyy) 
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

def xSecondDerivativeNeuman(fx, h, v1, v2):
    fx = np.array(fx)
    v1 = np.array(v1)
    v2 = np.array(v2)
    xi = [fx[1, :] - 2 * h * v1]
    xf = [2 * h * v2 + fx[-1,:]]
    fx = np.concatenate((xi, fx, xf))
    return (fx[:-2,:] - 2 * fx[1:-1,:] + fx[2:,:]) / h ** 2

def ySecondDerivativeNeuman(fx, h, v1, v2):
    fx = np.array(fx)
    v1 = np.array(v1)
    v2 = np.array(v2)
    yi = np.array([fx[:, 1] - 2 * h * v1]).T
    yf = np.array([2 * h * v2 + fx[:,-1]]).T
    fx = np.concatenate((yi, fx, yf), axis = 1)
    return (fx[:, :-2] - 2 * fx[:,1:-1] + fx[:,2:]) / h ** 2

def xSecondDerivativeNeumanSecond(fx, h, v1, v2):
    fx = np.array(fx)
    v1 = np.array(v1)
    v2 = np.array(v2)
    vxx0 = [(-3 * h * v1 + (-fx[0,:] / 12 - 9 * fx[1,:] + 16 * fx[2,:] - 38 * fx[3,:] / 3 + 49 * fx[4,:] / 12 - fx[5,:]) ) * 0.5 / h **2]
    vxxN = [(3 * h * v1 + (-fx[-1,:] / 12 - 9 * fx[-2,:] + 16 * fx[-3,:] - 38 * fx[-4,:] / 3 + 49 * fx[-5,:] / 12 - fx[-6,:]) ) * 0.5 / h **2]
    return np.concatenate((vxx0, (fx[:-2,:] - 2 * fx[1:-1,:] + fx[2:,:]) / h ** 2, vxxN))

def ySecondDerivativeNeumanSecond(fx, h, v1, v2):
    fx = np.array(fx)
    v1 = np.array(v1)
    v2 = np.array(v2)
    vyy0 = [(-3 * h * v1 + (-fx[:,0] / 12 - 9 * fx[:,1] + 16 * fx[:,2] - 38 * fx[:,3] / 3 + 49 * fx[:,4] / 12 - fx[:,5]) ) * 0.5 / h **2]
    vyyN = [(3 * h * v1 + (-fx[:,-1] / 12 - 9 * fx[:,-2] + 16 * fx[:,-3] - 38 * fx[:,-4] / 3 + 49 * fx[:,-5] / 12 - fx[:,-6]) ) * 0.5 / h **2]
    return np.concatenate((vyy0, (fx[:,:-2] - 2 * fx[:,1:-1] + fx[:,2:]) / h ** 2, vyyN))

def ySecondDerivativeNeumanSecond(fx, h, v1, v2):
    fx = np.array(fx)
    v1 = np.array(v1)
    v2 = np.array(v2)
    yi = np.array([fx[:, 1] - 2 * h * v1]).T
    yf = np.array([2 * h * v2 + fx[:,-1]]).T
    fx = np.concatenate((yi, fx, yf), axis = 1)
    return (fx[:, :-2] - 2 * fx[:,1:-1] + fx[:,2:]) / h ** 2

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

fig, ax = plt.subplots(1)

x = np.linspace(0, 1, 81)
y = np.linspace(0, 1, 81)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = dx / 250
X, Y = np.meshgrid(x, y)
sigma = 0.001
fx = np.exp(- 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / sigma)
heat = set2DHeatNeumann(1, dx, dy)
solution, t = evolvePde(heat, fx, x, 0, 0.1, dt)

T = []

for s in solution:
    T += [np.sum(s**2, (0, 1)) * dx * dy]
T = np.array(T)
ax.semilogy(t, T, c = "firebrick", label = "$\Delta t = \\frac{\Delta x}{300}$", ls = (0, [10, 20]))

x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = dx / 500
X, Y = np.meshgrid(x, y)
sigma = 0.001
fx = np.exp(- 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / sigma)
heat = set2DHeatNeumann(1, dx, dy)
solution, t = evolvePde(heat, fx, x, 0, 0.1, dt)
T = []

for s in solution:
    T += [np.sum(s**2, (0, 1)) * dx * dy]
T = np.array(T)
ax.semilogy(t, T, c = "darkgreen", label = "$\Delta t = \\frac{\Delta x}{500}$", ls = (10, [10, 20]))

x = np.linspace(0, 1, 131)
y = np.linspace(0, 1, 131)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = dx / 750
X, Y = np.meshgrid(x, y)
sigma = 0.001
fx = np.exp(- 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / sigma)
heat = set2DHeatNeumann(1, dx, dy)
solution, t = evolvePde(heat, fx, x, 0, 0.1, dt)
T = []

for s in solution:
    T += [np.sum(s**2, (0, 1)) * dx * dy]

f = open("tforT.tex", "w")
for i, d in enumerate(T):
    if d < 1e-4:
        f.write(f"$t = {t[i]:.05f}$")
        break
f.close();
quit()

T = np.array(T)
ax.semilogy(t, T, c = "steelblue", label = "$\Delta t = \\frac{\Delta x}{750}$", ls = (20, [10, 20]))

ax.legend(handlelength = 5)
ax.set_xlabel("$t$")
ax.set_ylabel("$T(t)$")
fig.savefig("figs/T.pdf")

fig, ax = plt.subplots(1)
ax.plot(np.array(t)[:2000], T[:2000], c = "orangered")
ax.set_xlabel("$t$")
ax.set_ylabel("$T(t)$")
fig.savefig("figs/T1.pdf")

fps = 1000

fig = plt.figure(figsize = (10, 15), tight_layout=True)
gs = GridSpec(6, 4, figure = fig)

ax = fig.add_subplot(gs[0:2, 0:2], projection = "3d")
ax.plot_surface(X, Y, solution[0], cmap=cm.coolwarm,
     linewidth = 0, antialiased=False, label = f"$t = {t[0]:.04f}$")
ax.set_zlim(-0.5, 1.05); 
ax.set_title(f"$t = {t[0]:.04f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x, y)$")

ax = fig.add_subplot(gs[0:2, 2:4], projection = "3d")
ax.plot_surface(X, Y, solution[100], cmap=cm.coolwarm,
        linewidth = 0, antialiased=False, label = f"$t = {t[1000]:.04f}$")
ax.set_zlim(-0.5, 1.05); 
ax.set_title(f"$t = {t[100]:.04f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x, y)$")

ax = fig.add_subplot(gs[2:4, 0:2], projection = "3d")
ax.plot_surface(X, Y, solution[300], cmap=cm.coolwarm,
     linewidth = 0, antialiased=False, label = f"$t = {t[3000]:.04f}$")
ax.set_zlim(-0.5, 1.05); 
ax.set_title(f"$t = {t[300]:.04f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x, y)$")

ax = fig.add_subplot(gs[2:4, 2:4], projection = "3d")
ax.plot_surface(X, Y, solution[1000], cmap=cm.coolwarm,
     linewidth = 0, antialiased=False, label = f"$t = {t[5000]:.04f}$")
ax.set_zlim(-0.5, 1.05); 
ax.set_title(f"$t = {t[1000]:.04f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x, y)$")

ax = fig.add_subplot(gs[4:6, 1:3], projection = "3d")
ax.plot_surface(X, Y, solution[3000], cmap=cm.coolwarm,
     linewidth = 0, antialiased=False, label = f"$t = {t[7000]:.04f}$")
ax.set_zlim(-0.5, 1.05); 
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x, y)$")
ax.set_title(f"$t = {t[3000]:.04f}$")
fig.savefig("figs/heatSurface.pdf")

x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = dx / 500
X, Y = np.meshgrid(x, y)

sigma = 0.01
fx = np.exp(- 0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / sigma)
heat = set2DHeatNeumann(1, dx, dy)
solution, t = evolvePde(heat, fx, x, 0, 0.1, dt)
fig, ax = plt.subplots(1)
ax.plot(X[0,:], solution[0][51,:], label = f"$t = {t[0] : .04f}$", c = "darkred")
ax.plot(X[0,:], solution[-1][51,:], label = f"$t = {t[-1] : .04f}$", c = "cadetblue")
ax.legend()
fig.savefig("figs/profile.pdf")

quit()
                                                                                
def update(frame, solution, ax):
    ax.clear()
    ax.set_zlim(-0.5 , 1.5);
    ax.plot_surface(X, Y, solution[frame], cmap=cm.coolwarm,
                                linewidth = 0, antialiased=False)

ax.plot_surface(X, Y, solution[0], cmap=cm.coolwarm,
                       linewidth = 0, antialiased=False)
                                                                                
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

ax.semilogy(t, T)

print(T[-1])

plt.show()
