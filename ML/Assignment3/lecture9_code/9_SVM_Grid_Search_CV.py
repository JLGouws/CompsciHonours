# ---------------------------------------------------------------------------------------------------------------
# Parameter estimation using grid search with cross-validation on two metrics (loop)
# The performance of the selected hyper-parameters and trained model is measured on the digits dataset


from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np

# Loading the Digits dataset
digits = datasets.load_digits()

# print(dir(digits))


# There are 150 data points; each have four feature values:
# print(digits.data.shape)

# Four features: sepal and petal in two-dimensions as shown in slides:
# print(digits.feature_names)

# Inspecting the class labels reveals a total of three classes:
# print(np.unique(digits.target))
# print(digits.target)
print(digits.target_names)

# To apply an classifier on this data, we need to flatten images, to
# turn the data in a (samples, feature) matrix:

n_samples = len(digits.data)
X = digits.data.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tune_param =[
                    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'C': [1, 10, 100, 1000]}
            ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    grid = GridSearchCV(SVC(gamma='auto'), tune_param, cv=3,
                       scoring='%s_macro' % score)
    grid.fit(X_train, y_train) #note how we do CV on the training set


    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    print()

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()

    print("Detailed classification report:")
    print()
    y_pred = grid.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    print("Accuracy",metrics.accuracy_score(y_test, y_pred))

# Note the dataset is too simple but it shows the concept of parameter tuning
