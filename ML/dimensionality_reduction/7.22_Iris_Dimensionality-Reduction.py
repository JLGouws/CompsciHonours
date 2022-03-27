# -------------------------------------------------------------------------------------------------------------------
# Reducing the Dimensionality of the Data
# Defeating the curse of dimensionality: for a given sample size, there exists an optimal number of features for best
# classification performance

"""------------------------------------------------------------------------------------------------------------------
Implementing Principal Component Analysis (PCA) in OpenCV
# -------------------------------------------------------------------------------------------------------------------"""

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
x, y = np.random.multivariate_normal(mean, cov, 1000).T

# Plot this data using Matplotlib. Look at the spread:
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

'''2D plot'''
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'o', zorder=1)
# plt.axis([0, 40, 0, 40])
# plt.xlabel('feature 1')
# plt.ylabel('feature 2')
# plt.show()

# Format the data by stacking the `x` and `y` coordinates as a feature matrix `X`:
X = np.vstack((x, y)).T

# Create an empty array `np.array([])` as a mask, telling OpenCV to use all data points in the feature matrix.
# Compute PCA on the feature matrix `X`
import cv2
mu, eig = cv2.PCACompute(X, np.array([]))
print(eig)

# Note the following looks complicated but is simply for demonstration purposes (showing directional arrows of PC.
# Plot the eigenvectors of the decomposition on top of the data:

# pip install -i https://test.pypi.org/simple/ variable-factor-map-Huy-Bui
# from variable_factor_map import pca_map
# from sklearn import datasets
# import pandas as pd
# iris = datasets.load_iris()
# X=pd.DataFrame(data=iris.data,columns=iris.feature_names)
# pca_map(X, figsize=(10,10), sup="Iris", print_values= False)

from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

n_components = 4

# Do the PCA.
pca = PCA(n_components=n_components)
reduced = pca.fit_transform(df[['sepal_length', 'sepal_width',
                                'petal_length', 'petal_width']])

# Append the principle components for each entry to the dataframe
for i in range(0, n_components):
    df['PC' + str(i + 1)] = reduced[:, i]

# display(df.head())

'''scree plot: shows how much each component contributes (diminishing)'''
ind = np.arange(0, n_components)
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.pointplot(x=ind, y=pca.explained_variance_ratio_)
ax.set_title('Scree plot')
ax.set_xticks(ind)
ax.set_xticklabels(ind)
ax.set_xlabel('Component Number')
ax.set_ylabel('Explained Variance')
plt.show()

# Show the points in terms of the first two PCs
g = sns.lmplot('PC1',
               'PC2',
               hue='species', data=df,
               fit_reg=True,
               scatter=True,
               size=7)

plt.show()
quit()
'''Plot 2D variable factor map.'''
(fig, ax) = plt.subplots(figsize=(8, 8))
for i in range(0, pca.components_.shape[1]):
    ax.arrow(0,
             0,  # Start the arrow at the origin
             pca.components_[0, i],  # 0 for PC1
             pca.components_[1, i],  # 1 for PC2
             head_width=0.1,
             head_length=0.1)

    plt.text(pca.components_[0, i] + 0.05,
             pca.components_[1, i] + 0.05,
             df.columns.values[i])

an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
plt.axis('equal')
ax.set_title('Variable factor map')
plt.show()

'''Easy variable-factor-map (pip install -i https://test.pypi.org/simple/ variable-factor-map-Huy-Bui)'''
from variable_factor_map import pca_map
import pandas as pd
X = pd.DataFrame(data=df, columns=['sepal_length', 'sepal_width','petal_length', 'petal_width'])
pca_map(X, figsize=(10, 10), sup="Toy Gaussian Data", print_values=False)
