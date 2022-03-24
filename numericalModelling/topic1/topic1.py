import matplotlib.pyplot as plt
import numpy as np

def logisticMap(r, x0, n):
  for i in range (0, n):
    x0 =  r * x0 * (1 - x0)
  return x0

print(logisticMap(2, 0.5, 10))
print(logisticMap(2, 0.5, 20))
print(logisticMap(2, 0.5, 100))

print(logisticMap(2, 0.01, 100))
print(logisticMap(2, 0.01, 200))
print(logisticMap(2, 0.01, 400))

def logisticMapList(r, x0, n):
  xs = [x0]
  for i in range (0, n):
    x0 =  r * x0 * (1 - x0)
    xs += [x0]
  return xs

first20 = logisticMapList(2, 0.01, 20)
fig, ax = plt.subplots(1)
ax.plot(range(0, 21), first20)
ax.set_xlabel("$n$")
ax.set_ylabel("$x_n$")
fig.savefig("figs/first20.pdf")

def logisticMapListTol(r, x0, tol, N):
  xs = [x0, r * x0 * (1 - x0)]
  n = 0
  while abs((xs[-1] - xs[-2]) / xs[-1]) > tol and n <= N:
    xs +=  [r * xs[-1] * (1 - xs[-1])]
    n += 1
  return xs

fig, ax = plt.subplots(1)
for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
  xs = logisticMapListTol(2, x, 0.01, 1000)
  ax.plot(range(0, len(xs)), xs)
ax.set_xlabel("$n$")
ax.set_ylabel("$x_n$")
fig.savefig("figs/multipleConvergences.pdf")

rs = [0.05 * x for x in range(0, 60)]

def logisticMapTol(r, x0, tol, N):
  xs = [x0, r * x0 * (1 - x0)]
  n = 0
  while abs((xs[-1] - xs[-2]) ) > abs(xs[-1]) * tol and n <= N:
    xs +=  [r * xs[-1] * (1 - xs[-1])]
    n += 1
  return (xs[-1] + xs[-2])/2

fig, ax = plt.subplots(1)
xstar = []
for r in rs:
  xstar += [logisticMapTol(r, 0.5, 10e-6, 1000)]
ax.plot(rs, xstar)
ax.set_xlabel("$r$")
ax.set_ylabel("$x^*$")
fig.savefig("figs/equilibriumPoint.pdf")

q6 = logisticMapList(3.2, 0.65, 80)
fig, ax = plt.subplots(1)
ax.plot(range(0, 81), q6)
ax.set_xlabel("$n$")
ax.set_ylabel("$x_n$")
fig.savefig("figs/q6.pdf")


rs = [0.005 * x for x in range(0, 690)]
rnew =[]

fig, ax = plt.subplots(1)
xstar = []
for r in rs:
  xstar += logisticMapList(r, 0.5, 1000)[-2:]
  rnew += [r, r]
ax.scatter(rnew, xstar)
ax.set_xlabel("$r$")
ax.set_ylabel("$x^*$")
fig.savefig("figs/equilibriumPointsBifurcations.pdf")

rs = [0.005 * x for x in range(690, 800)]

extraPoints = 100
fig, ax = plt.subplots(1)
for r in rs:
  xstar += logisticMapList(r, 0.5, 1000 + extraPoints)[-extraPoints:]
  rnew += [r] * extraPoints
ax.scatter(rnew, xstar, s = 4)
ax.set_xlabel("$r$")
ax.set_ylabel("$x^*$")
fig.savefig("figs/furtherEquilibriumPointsBifurcations.jpeg")
