"""---------------------------------------------------------------------------------------------------------------
Motivating Support Vector Machines
Consider the simple case of a classification task, in which two classes of points are well separated:
---------------------------------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns;
from sklearn.svm import SVC

#from MyCommon.Visualize_Data import random_state

sns.set()

from sklearn.datasets import make_blobs

# from sklearn.datasets.samples_generator import make_blobs #samples_generator is depracated
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.show()

'''There is more than one possible dividing line/plane that can perfectly discriminate between the two classes,
but some can cause the model to fit data points on the wrong side:'''

# xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
#
# for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
#     plt.plot(xfit, m * xfit + b, '-k')
#
# plt.xlim(-1, 3.5)
# plt.show()


'''Support Vector Machines: Maximizing the Margin
Add a margin on either side of the line/plane to see width'''

# xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#
# for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
#     yfit = m * xfit + b
#     plt.plot(xfit, yfit, '-k')
#     plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
#                      color='#AAAAAA', alpha=0.4)
#
# plt.xlim(-1, 3.5)
# plt.show()

'''Fitting a support vector machine: Use a large C parameter to avoid regularisation'''


# from sklearn.svm import SVC # "Support vector classifier"
# model = SVC(kernel='linear', C=1E10)
# model.fit(X, y)
#
# '''Create a Visualization function for the SVM decision boundaries and support vectors:'''
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


#
# '''plot the function. Notice that a few of the training points just touch the margin. These are support vectors:'''
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(model)
# # plt.show()
#
# # print(model.support_vectors_)
#
#
# '''Fit is only affected by support vectors. Plotting the first 60 points and 120 points result in the same model:'''
# def plot_svm(N=10, ax=None):
#     X, y = make_blobs(n_samples=200, centers=2,
#                       random_state=0, cluster_std=0.60)
#     X = X[:N]
#     y = y[:N]
#     model = SVC(kernel='linear', C=1E10)
#     model.fit(X, y)
#
#     ax = ax or plt.gca()
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#     ax.set_xlim(-1, 4)
#     ax.set_ylim(-1, 6)
#     plot_svc_decision_function(model, ax)
#
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# for axi, N in zip(ax, [60, 120]):
#     plot_svm(N, axi)
#     axi.set_title('N = {0}'.format(N))
# plt.show()


'''---------------------------------------------------------------------------------------------------------------'''
'''Transcending that which is linear: Kernel SVM'''

'''Here are some data points that are not linearly separable:'''
# from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets import make_moons

X, y = make_moons(100, noise=.1, random_state=42)

clf = SVC(kernel='rbf', C=1, gamma='scale').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)
plt.show()

'''Compute a radial basis function (Gaussian) centred on the middle clump:'''
r = np.exp(-(X ** 2).sum(1))
print(r)
quit()

'''Use mplot3d (or plotly express) to plot the data in 3D:'''
from mpl_toolkits import mplot3d


def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')


plot_3D()
plt.show()

'''Kernelized SVM using the kernel model hyperparameter:'''
# clf = SVC(kernel='rbf', C=1E6)
# clf.fit(X, y)

'''Using kernelized SVM, we learn a suitable nonlinear decision boundary without explicitly computing 3D points'''
'''Plot the result of the RBF SVM:'''
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf)
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=300, lw=1, facecolors='none')
# plt.show()

# ---------------------------------------------------------------------------------------------------------------
# Tuning the SVM: Softening Margins
# What if the data has overlap:
# X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.show()


# The plot shown below gives a visual picture of how changing C parameter affects the final fit:
# X, y = make_blobs(n_samples=100, centers=2,
#                   random_state=0, cluster_std=0.8)
#
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
#
# for axi, C in zip(ax, [10.0, 0.1]):
#     model = SVC(kernel='linear', C=C).fit(X, y)
#     axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#     plot_svc_decision_function(model, axi)
#     axi.scatter(model.support_vectors_[:, 0],
#                 model.support_vectors_[:, 1],
#                 s=300, lw=1, facecolors='none');
#     axi.set_title('C = {0:.1f}'.format(C), size=14)
# plt.show()

quit()
'''---------------------------------------------------------------------------------------------------------------'''
'''PART 2: The optimal value of the C parameter can be tuned using cross-validation on Face Recognition dataset'''
'''---------------------------------------------------------------------------------------------------------------'''

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=100)
print(faces.target_names)
print(faces.images.shape)
print(np.unique(faces.target))

# Plot some of these faces to have an idea of what we dealing with:
fig, ax = plt.subplots(3,3)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
plt.show()

'''Can simply use each of the ~3000 pixel values as a feature, but extracting the dominant features often yield better results. PCA can do this:'''
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
#
'''whiten scales the resulting components to unit variance.'''
pca = PCA(n_components=50, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', gamma='scale')
model = make_pipeline(pca, svc)
#
'''split the data into a training and testing set:'''
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(faces.data, faces.target,random_state=42)

'''Use a grid search cross-validation to explore combinations of parameters that give best metric performance.'''
'''Adjust C (margin hardness) and gamma (size of the RBF kernel) to determine the best model:'''
from sklearn.model_selection import GridSearchCV
tune_param = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, tune_param, cv=3)

grid.fit(X_train, y_train)
print(grid.best_params_)

'''Often more than one optimal set of parameters. Now predict using the 'optimal' model:'''
model = grid.best_estimator_
y_pred = model.predict(X_test)

'''Check the predicted values of the test set of people shown earlier:'''
fig, ax = plt.subplots(3,3)
for i, axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1],
                   color='black' if y_pred[i] == y_test[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.show()

''' Classification report is a convenient way to list the results label by label for three metrics:'''
from sklearn import model_selection,metrics

print(metrics.classification_report(y_test, y_pred,
                            target_names=faces.target_names))
print("Accuracy",metrics.accuracy_score(y_test, y_pred))

'''Confusion matrix to show how classes were misclassified (confused):'''
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
