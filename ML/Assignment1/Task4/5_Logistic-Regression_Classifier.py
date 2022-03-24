# -------------------------------------------------------------------------------------------------------------------
# Logistic Regression Classifier to predict Iris Flower Species (from image data)

# Build a machine learning model that can learn the measurements of the species of iris flowers toward predicting the
# species of an unseen iris flower.
# This iris dataset provides a total of four features.
# To find out how logistic regression works in these cases, please refer to the book.
# -------------------------------------------------------------------------------------------------------------------

# Scikit-learn includes some sample datasets. The Iris dataset is imported:

import numpy as np
import cv2

from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Loading the iris dataset
iris = datasets.load_iris()


# The structure of the boston object:
# - `DESCR`: Get a description of the data
# - `data`: The actual data, <`num_samples` x `num_features`>
# - `feature_names`: The names of the features
# - `target`: The class labels, <`num_samples` x 1>
# - `target_names`: The names of the class labels

# print(dir(iris))


# There are 150 data points; each have four feature values:
# print(iris.data.shape)

# Four features: sepal and petal in two-dimensions as shown in slides:
# print(iris.feature_names)

# Inspecting the class labels reveals a total of three classes:
# print(np.unique(iris.target))
# print(iris.target)
# print(iris.target_names)
# Let's see what the feature values are:

# print(iris.data)

# -------------------------------------------------------------------------------------------------------------------
# Using Logistic Regression as a binary classifier
# Make it Binary: Discard all data points belonging to for e.g. class label 2 (virginica), by selecting all the rows
# that do not
# belong to class 2:
idx = iris.target !=2
data = iris.data[idx].astype(np.float32)
target = iris.target[idx].astype(np.float32)

# print(target.size)# Notice the first 100 rows are class 0 and 1
# This is now a binary problem

# Visualizing the data using a scatter plot of the first two features
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired, s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, target, test_size=0.8, random_state=42
)

# End up with 90 training data points and 10 test data points
# print("training",X_train.shape, y_train)
# print("test",X_test.shape, y_test)

# -------------------------------------------------------------------------------------------------------------------
# Training the LR model
lr = cv2.ml.LogisticRegression_create()

# Choice of a training method:`cv2.ml.LogisticRegression_BATCH` or `cv2.ml.LogisticRegression_MINI_BATCH`
# Choose to update the model after every data point using cv2.ml.LogisticRegression_MINI_BATCH:
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(1)

# Specify the number of iterations for the batch
lr.setIterations(100)
# OpenCV's `train` method returns True upon success:
lr.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Get the learned weights:
lr.get_learnt_thetas()
print("Learnded parameters: ", lr.get_learnt_thetas())

# -------------------------------------------------------------------------------------------------------------------
# ### Testing the classifier
# Using the learnt weights calculate the accuracy score on the training set (seen data).
# This test will show how well the model was able to memorize the training dataset
# y_pred_train named to make it clear that it is an output (y) but not unseen, aka it is testing on the same data it trained on
ret, y_pred_train = lr.predict(X_train)
print("Trained data ", metrics.accuracy_score(y_train, y_pred_train))

# But, how well can it classify unseen data points:
ret, y_pred_test = lr.predict(X_test)
print("predicted data ", metrics.accuracy_score(y_test, y_pred_test))
# print(metrics.classification_report(y_test, y_pred,
#                             target_names=iris.target_names[1:3]))
