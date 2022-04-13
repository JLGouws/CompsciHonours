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

xstarnew = []
rnewnew = []

fig, ax = plt.subplots(1)
xstar = []
for r in rs:
  xstar += logisticMapList(r, 0.5, 1000)[-2:]
  xstarnew += [np.flip(np.unique(np.round(xstar[-2:], 3)))]
  rnewnew += [r]
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
  xstarnew += [np.flip(np.unique(np.round(xstar[-extraPoints:], 4)))]
  rnewnew += [r]
  rnew += [r] * extraPoints
ax.plot(rnew, xstar, 'bo', ms = 2)
ax.set_xlabel("$r$")
ax.set_ylabel("$x^*$")
fig.savefig("figs/furtherEquilibriumPointsBifurcations.pdf")

def finish_path(r_list, xstar_list, r_plot, xstar_plot, ax, index):
    if len(r_list) == 0:
        ax.plot(r_plot, xstar_plot, 'b-')
    else:
        finish_path(r_list[1:], xstar_list[1:], np.append(r_plot, r_list[0]), np.append(xstar_plot, xstar_list[1][index]), ax, index)

def draw_rest_path(r_list, xstar_list, r_plot, xstar_plot, ax, index):
    print(xstar_list[1].size, r_list[0])
    if len(r_list) == 0:
        ax.plot(r_plot, xstar_plot, 'b-')
    elif xstar_list[1].size > xstar_list[0].size:
        if xstar_list[1].size - xstar_list[0].size == 1:
            draw_rest_path(r_list[1:], xstar_list[1:], np.append(r_plot, r_list[0]), np.append(xstar_plot, xstar_list[1][index]), ax, index) 
            draw_rest_path(r_list[1:], xstar_list[1:], np.append(r_plot[-1], r_list[0]), np.append(xstar_plot[-1], xstar_list[1][index + 1]), ax, index + 1) 
        elif xstar_list[1].size - xstar_list[0].size == 2:
            finish_path(r_list[1:], xstar_list[1:], np.append(r_plot, r_list[0]), np.append(xstar_plot, xstar_list[1][2 * index]), ax, 2 * index) 
#            finish_path(r_list[1:], xstar_list[1:], np.append(r_plot[-1], r_list[0]), np.append(xstar_plot[-1], xstar_list[1][2 * index + 1]), ax, 2 * index + 1) 
    else:
        draw_rest_path(r_list[1:], xstar_list[1:], np.append(r_plot, r_list[0]), np.append(xstar_plot, xstar_list[1][index]), ax, index)
        
def draw_path(r_list, xstar_list, ax):
    draw_rest_path(r_list[1:], xstar_list, r_list[0], xstar_list[0][0], ax, 0)

#fig, ax = plt.subplots(1)
#draw_path(rnewnew, xstarnew, ax)
#ax.set_xlabel("$r$")
#ax.set_ylabel("$x^*$")
#fig.savefig("figs/furtherEquilibriumPointsBifurcationsNew.pdf")
