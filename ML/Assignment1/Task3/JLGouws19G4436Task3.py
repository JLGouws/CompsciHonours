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
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Loading the iris dataset
iris = datasets.load_iris()

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


#The data is visually separable, machine should have no problem discerning data
plt.figure(figsize=(10, 10), tight_layout = True)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired, s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.show()

plt.figure(figsize=(10, 10), tight_layout = True)
plt.scatter(data[:, 1], data[:, 2], c=target, cmap=plt.cm.Paired, s=100)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, target, test_size=0.8, random_state=42
)

#Initialize the Logistic regression model
lr = LogisticRegression()

#Train the model
lr.fit(X_train, y_train)

# Get the learned weights:
print("Learnt parameters: ", lr.coef_)
print("Learnt parameters: ", lr.intercept_)

# -------------------------------------------------------------------------------------------------------------------
# ### Testing the classifier
# Using the learnt weights calculate the accuracy score on the training set (seen data).
# This test will show how well the model was able to memorize the training dataset
# y_pred_train named to make it clear that it is an output (y) but not unseen, aka it is testing on the same data it trained on
#ret, y_pred_train = lr.predict(X_train)
#print(metrics.accuracy_score(y_train, y_pred_train))

# But, how well can it classify unseen data points:
y_pred_test = lr.predict(X_test) 
print("Accuracy of model", metrics.accuracy_score(y_test, y_pred_test))
