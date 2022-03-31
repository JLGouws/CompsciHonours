import numpy as np
import matplotlib.pyplot as mp

def taylors_method(f, a, b, N, alpha):
  Y = []
  h = (b -a)/N
  t = a;
  w = alpha
  Y += [(t, w)]
  for i in range (1, N + 1):
    w = w + h * f(t, w)
    t = a + i * h
    Y += [(t, w)]
  return Y

logistic = lambda t, u : 1 * u * (1 - u)

points = np.array(taylors_method(logistic, 0, 20, 2000, 0.05))
time = points[:, 0]
solution = points[:, 1]

fig, ax = mp.subplots(figsize=(10,10))
ax.plot(time, solution)
fig.savefig("1a.pdf")
