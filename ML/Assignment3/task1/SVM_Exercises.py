import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC #sklearns support vector machine
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing as prep
from sklearn.gaussian_process.kernels import RBF

from scipy.spatial.distance import pdist, cdist, squareform

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
clf = SVC(kernel='rbf', C=10000, gamma='scale').fit(X1, np.reshape(y1,-1)) 
plot_svc_decision_function(clf, plot_support=False)

plt.savefig("exercise1.pdf")
print("Ex1: ", clf.score(X1, y1))

# -------------------------------------------------------------------------------------------------------------------
# Exercise 2
X2 = np.random.rand(n_samples, n_features)
y2 = np.ones((n_samples, 1))
idx_neg = (X2[:, 0] < 0.5) * (X2[:, 1] < 0.5) + (X2[:, 0] > 0.5) * (X2[:, 1] > 0.5)
y2[idx_neg] = 0

plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X2[:, 0],-1), np.reshape(X2[:, 1],-1), c=np.reshape(y2,-1),s=100)

# Code solution 2 here:

plt.title("Exercise 2")

clf = make_pipeline(prep.MinMaxScaler(), SVC(kernel='rbf', C=10000, gamma='scale'))
clf.fit(X2, np.reshape(y2,-1)) 
plot_svc_decision_function(clf, plot_support=False)

plt.savefig("exercise2.pdf")

print("Ex2: ", clf.score(X2, y2))

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

X3 = np.reshape(X3, X3.shape[:2])
y3 = np.reshape(y3,-1)

clf = SVC(kernel='rbf', C=10000, gamma='scale').fit(X3, y3)
plot_svc_decision_function(clf, plot_support=False)

plt.savefig("exercise3.pdf")
print("Ex3: ", clf.score(X3, y3))

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

#quit()
# Code solution 4 here:

def my_kernel(X, Y = None):
    X = np.atleast_2d(X)
    if Y is None:
        dists = pdist(X, metric="sqeuclidean")
        K = np.exp(-0.5 * dists)
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)
    else:
        dists = cdist(X, Y, metric="sqeuclidean")
        K = np.exp(-0.5 *dists)
    return K
plt.title("Exercise 4")

X4 = np.c_[np.reshape(X4[:, 0],-1), np.reshape(X4[:, 1],-1)]
y4 = np.reshape(y4,-1)

clf = make_pipeline(prep.StandardScaler(), SVC(kernel=my_kernel, C=999999, gamma='scale'))
clf.fit(X4, y4) 
plot_svc_decision_function(clf, plot_support=False)

plt.savefig("exercise4.pdf")
print("Ex4: ", clf.score(X4, y4))

# -------------------------------------------------------------------------------------------------------------------
# Exercise 5
X5, y5 = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=42)
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X5[:, 0],-1), np.reshape(X5[:, 1],-1), c=np.reshape(y5,-1),s=100)

# Code solution 5 here:

plt.title("Exercise 5")

clf = make_pipeline(prep.StandardScaler(), SVC(kernel='rbf', C=9999999, gamma='scale'))
clf.fit(X5, y5) 
plot_svc_decision_function(clf, plot_support=False)

plt.savefig("exercise5.pdf")
print("Ex5: ", clf.score(X5, y5))
