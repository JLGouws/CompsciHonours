# -------------------------------------------------------------------------------------------------------------------
# Reducing the Dimensionality of the Data
# Defeating the curse of dimensionality: for a given sample size, there exists an optimal number of features for best
# classification performance

# -------------------------------------------------------------------------------------------------------------------
# Implementing Principal Component Analysis (PCA) in OpenCV

# PCA rotates all data points they align with the two axes that explains the spread of the data. PCA aims to transform
# the data to a new coordinate system by means of an orthogonal linear transformation.
# Project the data onto the new coordinate system such that the first coordinate has the greatest variance.
# This is called the first principal component

# Here is some random data drawn from a multivariate Gaussian:
import numpy as np
from sklearn import decomposition

mean = [20, 20]
cov = [[5, 0], [25, 25]]
np.random.seed(42)
data, target = np.random.multivariate_normal(mean, cov, 1000).T

# Plot this data using Matplotlib. Look at the spread:
import matplotlib.pyplot as plt
plt.style.use('ggplot')

'''2D plot'''
plt.figure(figsize=(10, 6))
plt.plot(data, target, 'o', zorder=1)
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()

# Format the data by stacking the `x` and `y` coordinates as a feature matrix `X`:
X = np.vstack((data, target)).T

# Create an empty array `np.array([])` as a mask, telling OpenCV to use all data points in the feature matrix.
# Compute PCA on the feature matrix `X`
import cv2

mu, eig = cv2.PCACompute(X, np.array([]))
# print(eig)

# '''Optional: Easy variable-factor-map (pip install -i https://test.pypi.org/simple/ variable-factor-map-Huy-Bui)'''
# from variable_factor_map import pca_map
# import pandas as pd
# X = pd.DataFrame(data=X, columns=['PC1', 'PC2'])
# pca_map(X, figsize=(10, 10), sup="Toy Gaussian Data", print_values=False)

'''PCA rotates the data so that the two axes (`x` and `y`) are aligned with the first two principal components.'''
# In OpenCV the PCAProject function both initiates and fits the transformed matrix to X2:
X2 = cv2.PCAProject(X, mu, eig)

# Scikit: first instantiate using the `decomposition` module:
pca = decomposition.PCA(n_components=2)

# Now use the `fit_transform` method:
X2 = pca.fit_transform(X)

# The blob of data is rotated so that the most spread is along the `x` axis:
# Note the more even spread
plt.figure(figsize=(10, 6))
plt.plot(X2[:, 0], X2[:, 1], 'o')
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.axis([-20, 20, -10, 10])
plt.show()

'''Implementing Independent Component Analysis (ICA) performs the same mathematical
steps as PCA, but it chooses the components of the decomposition to be as independent as possible from each other.'''
# Again, first instantiate, then use the `fit_transform` method:
ica = decomposition.FastICA()
X2 = ica.fit_transform(X)

# Plot the projected data on the first two independent components:
plt.figure(figsize=(10, 6))
plt.plot(X2[:, 0], X2[:, 1], 'o')
plt.xlabel('first independent component')
plt.ylabel('second independent component')
plt.axis([-0.2, 0.2, -0.2, 0.2])
plt.show()

'''Implementing Linear Discriminant Analysis (LDA) performs the same mathematical
steps as PCA'''

# Google Kaggle notebooks on non-linear methods like UMAP, LLE and IsoMap