import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC #sklearns support vector machine
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing as prep
np.random.seed(4242)

# plot the decision boundaries for SVM model
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# -------------------------------------------------------------------------------------------------------------------
# Exercise 1
n_samples = 500
n_features = 2
X1 = np.random.rand(n_samples, n_features)
y1 = np.ones((n_samples, 1))
idx_neg = (X1[:, 0] - 0.5) ** 2 + (X1[:, 1] - 0.5) ** 2 < 0.03
y1[idx_neg] = 0

plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X1[:, 0],-1), np.reshape(X1[:, 1],-1), c=np.reshape(y1,-1),s=100)
# Code solution 1 here:
plt.title("Exercise 1")
clf = make_pipeline(prep.StandardScaler(), SVC(kernel='rbf', C=10000, gamma='scale'))
clf.fit(X1, np.reshape(y1,-1)) 
plot_svc_decision_function(clf, plot_support=False)

plt.savefig("exercise1.pdf")

# -------------------------------------------------------------------------------------------------------------------
# Exercise 2
X2 = np.random.rand(n_samples, n_features)
y2 = np.ones((n_samples, 1))
idx_neg = (X2[:, 0] < 0.5) * (X2[:, 1] < 0.5) + (X2[:, 0] > 0.5) * (X2[:, 1] > 0.5)
y2[idx_neg] = 0

plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X2[:, 0],-1), np.reshape(X2[:, 1],-1), c=np.reshape(y2,-1),s=100)

from sklearn.gaussian_process.kernels import RBF
print(RBF(1.0).__call__(X2,X2))
# Code solution 2 here:
plt.title("Exercise 2")

def myKernal(X, Y): 
    indices = (X[:, 0] < 0.5) * (X[:, 1] < 0.5) + (X[:, 0] > 0.5) * (X[:, 1] > 0.5)
    outMatrix = np.ones_like(X)
    indices = np.resize(indices, outMatrix.shape)
    outMatrix[indices] = 0
    return outMatrix
print(myKernal(X1, y1))
clf = make_pipeline(prep.MinMaxScaler(), SVC(kernel="linear", C=10000, gamma='scale'))
clf.fit(X1, np.reshape(y1,-1)) 
plot_svc_decision_function(clf, plot_support=False)

plt.savefig("exercise2.pdf")

# -------------------------------------------------------------------------------------------------------------------
# Exercise 3
rho_pos = np.random.rand(n_samples // 2, 1) / 2.0 + 0.5
rho_neg = np.random.rand(n_samples // 2, 1) / 4.0
rho = np.vstack((rho_pos, rho_neg))
phi_pos = np.pi * 0.75 + np.random.rand(n_samples // 2, 1) * np.pi * 0.5
phi_neg = np.random.rand(n_samples // 2, 1) * 2 * np.pi
phi = np.vstack((phi_pos, phi_neg))
X3 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
y3 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))

plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X3[:, 0],-1), np.reshape(X3[:, 1],-1), c=np.reshape(y3,-1),s=100)

# Code solution 3 here:
plt.title("Exercise 3")

plt.savefig("exercise3.pdf")

# -------------------------------------------------------------------------------------------------------------------
# Exercise 4
rho_pos = np.linspace(0, 2, n_samples // 2)
rho_neg = np.linspace(0, 2, n_samples // 2) + 0.5
rho = np.vstack((rho_pos, rho_neg))
phi_pos = 2 * np.pi * rho_pos
phi = np.vstack((phi_pos, phi_pos))
X4 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
y4 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))

plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X4[:, 0],-1), np.reshape(X4[:, 1],-1), c=np.reshape(y4,-1),s=100)

# Code solution 4 here:
plt.title("Exercise 4")

plt.savefig("exercise4.pdf")

# -------------------------------------------------------------------------------------------------------------------
# Exercise 5
X5, y5 = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=42)
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X5[:, 0],-1), np.reshape(X5[:, 1],-1), c=np.reshape(y5,-1),s=100)

# Code solution 5 here:
plt.title("Exercise 5")

plt.savefig("exercise5.pdf")


