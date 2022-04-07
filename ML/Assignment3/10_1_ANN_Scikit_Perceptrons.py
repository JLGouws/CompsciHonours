#!/usr/bin/env python

# ---------------------------------------------------------------------------------------------------------------
# When we initialize a new perceptron object, we want to pass a learning rate (tol)
# and the number of iterations after which the algorithm should terminate (max_iter):

from sklearn import datasets
from sklearn.preprocessing import *
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np

# Load the digits dataset
digits = datasets.load_digits()

# There are 150 data points; each have four feature values:
# print(digits.data.shape)

# Inspecting the class labels reveals a total of three classes:
print(digits.data)
# print(digits.target)
# Create our X and y data
X = digits.data
y = digits.target

np.random.seed(42)

# Split to 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Standarize all the features to have mean=0 and unit variance
# sc = StandardScaler()
# print(digits.data)
sc = MinMaxScaler(feature_range=(-20, 20))
sc.fit(X_train)
#
# # Apply the scaler to the X training data
X_train_std = sc.transform(X_train)
# print(X_train_std) # oh no, we are lowering floating point precision!
# # Apply the SAME scaler to the X test data
X_test_std = sc.transform(X_test)
#
# # Training ends when the loss or score is not improving by at least tol for two consecutive epochs
# # Perceptron object with parameters: max of 40 epochs over the data, and learning rate of 0.1 tolerance
ppn = Perceptron(max_iter=42, tol=0.1)
#
# # Train the perceptron
# ppn.fit(X_train_std, y_train)
#
# # Apply the trained perceptron on the X data to make predicts for the y test data
# y_pred = ppn.predict(X_test_std)
#
# # Randomised initial weights that converge on different potential solutions each run, hence accuracy changes.
# # Accuracy of the model:
# print('Accuracy of perceptron on digits: %.2f' % accuracy_score(y_test, y_pred))
# ---------------------------------------------------------------------------------------------------------------
# OR
X_training=[[0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
           ]
y_training=[0,
            1,
            1,
            1
           ]

# # XOR
X_training=[[0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
           ]
y_training=[0,
            1,
            1,
            0
           ]


X_testing=X_training
y_true=y_training

# single perceptron can only do linear separation
ptn = Perceptron(max_iter=30, tol=0.1)
ptn.fit(X_training, y_training)
y_pred=ptn.predict(X_testing)
print("Perceptron \n",y_pred)
accuracy=metrics.accuracy_score(y_true, y_pred)
print('accuracy = ',accuracy)
#
print(ptn.intercept_, ptn.coef_,"\n") # show the synapsis weights w0, w1, w2, ...
# ---------------------------------------------------------------------------------------------------------------
# XOR MLP
# neurons = 4 with 3 layers. Play around with number of layers depending on complexity of problem.
# warm_start makes is reproducible by reusing the initialised weights in previous run
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2), activation='relu',warm_start=False)
mlp.fit(X_training, y_training)
y_pred=mlp.predict(X_testing)
print("MLP \n",y_pred)
accuracy=metrics.accuracy_score(y_true, y_pred)
print('accuracy=',accuracy)

# print([coef.shape for coef in mlp.coefs_])  # size of synapsis weights
# print(mlp.coefs_)                               # synapsis weights

quit()
