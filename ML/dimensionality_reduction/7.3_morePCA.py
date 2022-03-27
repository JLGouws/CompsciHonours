#!/usr/bin/env python

# ----------------------------------------------------------------------------------------------------------------------
# More on Principal Component Analysis

# While PCA is not a machine learning algorithm, it is unsupervised
# It is useful for dim reduction, visualization, for noise filtering, for feature extraction and engineering, etc.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('ggplot')

# Consider the following 200 points in 2D (features):
# rng = np.random.RandomState(42)
# X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

# Plot them:
# plt.scatter(X[:, 0], X[:, 1])
# plt.axis('equal')
# plt.show()

# Rather than attempting to *predict* the y values from the x values, PCA finds the *relationship* between the
# x and y values.

from sklearn.decomposition import PCA as RandomizedPCA
# pca = RandomizedPCA(n_components=2) # Only 2 principal components (2D) are used to represent all features x samples
# pca.fit(X)

# The fit extracts data, as "components" and "explained variance":
#
# print(pca.components_)
# print(pca.explained_variance_)

# The "components" define the direction of the vector, and the "explained variance" defines the squared-length
# of the vector:

# ----------------------------------------------------------------------------------------------------------------------
# PCA as dimensionality reduction
# ----------------------------------------------------------------------------------------------------------------------

# Using PCA for dimensionality reduction involves zeroing out one or more of the smallest principal components
# resulting in a lower-dimensional projection of the data that preserves the maximal data variance.

# pca = RandomizedPCA(n_components=1)
# pca.fit(X)
# X_pca = pca.transform(X)
# print("original shape:   ", X.shape)
# print("transformed shape:", X_pca.shape)

# The transformed data has been reduced to a single component.
# Visualize this perform the inverse transform of this reduced data and
# plot the 1D axis along with the original data:

# X_new = pca.inverse_transform(X_pca)
# plt.scatter(X[:, 0], X[:, 1], alpha=0.4)
# plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.6)
# plt.axis('equal')
# plt.show()


# The light points are the original data, while the dark points are the PCA projected version.
# This makes clear what a PCA dimensionality reduction means: the information along the least important principal axis or axes is removed

# ----------------------------------------------------------------------------------------------------------------------
# PCA for visualization: Hand-written digits
# Here is a 64-dimension dataset (compared to the 2D one)
# 8×8 pixel images, meaning that they are 64-dimensional

from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.data.shape)
#
# # Transform using PCA to 2D
#
pca = RandomizedPCA(2)  # project from 64 to 2 dimensions (Using the 2 components with the highest variance)
projected = pca.fit_transform(digits.data)
print(projected.shape)

# Plot the projected 2D data. Clear shows different digits as clusters of points (with some overlap) :

# plt.scatter(projected[:, 0], projected[:, 1],
#             c=digits.target, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('Spectral', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.show()


# How many of these components do we need? Visualize this showing the cumulative explained variance ratio

# Using only the first eight pixels means throwing away 90% of the pixels!

# The upper row of panels shows the individual pixels,
# and the lower row shows the cumulative contribution of these pixels to the construction of the image.
#
# PCA allows us to recover the salient features of the input image with just a mean plus eight components!

# A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data.
# Plot the cumulative explained variance ratio as a function of the number of components:

# pca = RandomizedPCA().fit(digits.data)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()

# Do we need more than 45 components?

# ----------------------------------------------------------------------------------------------------------------------
# PCA as Noise Filtering

# Components with variance much larger than the effect of the noise should be relatively unaffected by the noise.

# def plot_digits(data):
#     fig, axes = plt.subplots(4, 10, figsize=(10, 4),
#                              subplot_kw={'xticks':[], 'yticks':[]},
#                              gridspec_kw=dict(hspace=0.1, wspace=0.1))
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(data[i].reshape(8, 8),
#                   cmap='binary', interp='nearest',
#                   clim=(0, 16))
# plot_digits(digits.data)
# plt.show()

# Now lets add some random noise to create a noisy dataset, and re-plot it:

# np.random.seed(42)
# noisy = np.random.normal(digits.data, 4)
# plot_digits(noisy)
# plt.show()

# Use PCA to get rid of 50% of the variance:

# pca = RandomizedPCA(0.50).fit(noisy)

# Here 50% of the variance amounts to 12 principal components.
# print(pca.n_components_)

# Use the inverse of the transform to reconstruct the filtered digits:
# components = pca.transform(noisy)
# filtered = pca.inverse_transform(components)
# plot_digits(filtered)
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Eigenfaces

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)

# 62 x 47 pixels:
# print(faces.target_names)
# print(faces.images.shape)

# Here, a dimensionality of nearly 3,000.
# We will take a look at the first 150 components:

from sklearn.decomposition import PCA
pca = RandomizedPCA(150)
# pca.fit(faces.data)
#
# fig, axes = plt.subplots(3, 8, figsize=(9, 4),
#                          subplot_kw={'xticks':[], 'yticks':[]},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
# plt.show()
#
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()

# See that these 300 components account for about 98% of the variance.

# Can also select components and calculate PCA inline:
pca = RandomizedPCA(50).fit(faces.data)

# Compute the components and projected faces
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

# Plot the results
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')

ax[0, 0].set_ylabel('full-dim.\ninput')
ax[1, 0].set_ylabel('Red.-dim.\nreconstruct');
plt.show()

# The top row here shows the input images
# the bottom row shows the reconstruction of the images from just 300 of the ~3,000 initial features.
