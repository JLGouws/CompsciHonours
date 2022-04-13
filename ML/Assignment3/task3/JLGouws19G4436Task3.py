#sorry this code is quite roughly done an lacks good commenting, good luck marking :)
# ---------------------------------------------------------------------------------------------------------------
# Task 2 and 3 skeleton

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import sklearn.preprocessing as prep
from skimage.color import rgb2gray
from skimage.feature import hog

import skimage
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import resize

from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_image_files(container_path, dimension=(30, 30)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Your own dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


image_dataset = load_image_files("../images/")

'''A quick peek at the 100th image'''
#cv2.imshow("for those of you that cannot wait for img proc.", rgb2gray(image_dataset.images[100]))
#cv2.waitKey()

'''Split data, but randomly allocate to training/test sets'''
X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.5, random_state=42)

tune_param =[
                    {'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3],'C': [0.01, 1, 10]},
            ]

grid = GridSearchCV(SVC(), tune_param, cv=3,
                   scoring='precision')

grid.fit(X_train, y_train) #note how we do CV on the training set

gridSearchResults = pd.DataFrame(grid.cv_results_)
gridSearchResults['Kernel'] = grid.cv_results_['param_kernel']
#print(gridSearchResults.apply(
#        lambda x: "{mean_test_score:#0.3f} (+/-{std_test_score:#0.03f}) for {params}".format(**x), 1
#        ).to_frame().to_string(index=False, max_colwidth=-1, header=False))
#print()
#ax = sns.barplot(x="Kernel", y="mean_test_score", data = gridSearchResults)
#plt.savefig("KernelVsPrecision.png")

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

random_state = 42

preprocessSVC = make_pipeline(prep.MinMaxScaler(), PCA(n_components=20, random_state=random_state, whiten=True))

preprocessSVC.fit(X_train, y_train)
X_train_transform_SVC = preprocessSVC.transform(X_train)

gridPrepSVC = GridSearchCV(SVC(), tune_param, cv=3,
                   scoring='precision')

gridPrepSVC.fit(X_train_transform_SVC, y_train) #note how we do CV on the training set

gridSearchResultsPrepSVC = pd.DataFrame(gridPrepSVC.cv_results_)
gridSearchResultsPrepSVC['Kernel'] = gridPrepSVC.cv_results_['param_kernel']

print(gridSearchResultsPrepSVC.apply(
        lambda x: "{mean_test_score:#0.3f} (+/-{std_test_score:#0.03f}) for {params}".format(**x), 1
        ).to_frame().to_string(index=False, max_colwidth=-1, header=False))

#preprocessing the images
grayImages = rgb2gray(image_dataset.images) #gray out images, this can be done in HOG
#we gray out images as luminance is probably most important feature and individual pixels will in general not be helpful
data = []
for image in grayImages:
    data += [hog(image, orientations = 10, pixels_per_cell=(9,9))] #Getting the HOG of the images
data = np.array(data)
X_train_gray, X_test_gray, y_train_gray, y_test_gray = train_test_split(data, image_dataset.target, test_size=0.5, random_state=42)

#use pca and min max scaler
preprocessMLP = make_pipeline(PCA(n_components=59, random_state=random_state), prep.MinMaxScaler())

#fit the preprocessing to the data
preprocessMLP.fit(X_train_gray,y_train_gray)

#transform test data
X_train_transform_MLP = preprocessMLP.transform(X_train_gray)

mlp = MLPClassifier(solver='lbfgs', activation='relu',warm_start=True, max_iter = 5000, random_state=random_state)

#for tuning the mlp
tune_param_MLP =[
    {'hidden_layer_sizes': [(), (20, 5), (20, 3), (20), (10), (5), (4), (3)], 'alpha': [1e-1, 1, 1.5, 3]},
]

gridPrepMLP = GridSearchCV(mlp, tune_param_MLP, cv=3,
                   scoring='precision')

gridPrepMLP.fit(X_train_transform_MLP, y_train) #note how we do CV on the training set

gridSearchResultsPrepMLP= pd.DataFrame(gridPrepMLP.cv_results_)

#print out the results of the grid search
print(gridSearchResultsPrepMLP.apply(
        lambda x: "{mean_test_score:#0.3f} (+/-{std_test_score:#0.03f}) for {params}".format(**x), 1
        ).to_frame().to_string(index=False, max_colwidth=-1, header=False))

mlp = gridPrepMLP.best_estimator_ #test the best estimator on test data
y_pred_mlp = mlp.predict(X_train_transform_MLP)

precision=metrics.precision_score(y_train_gray, y_pred_mlp)
#print precision on training data
print('Train precision=',precision)

X_test_transform_MLP = preprocessMLP.transform(X_test_gray)
y_pred = mlp.predict(X_test_transform_MLP)

precision=metrics.precision_score(y_test_gray, y_pred)
#print precision on testing data
print('Test precision=',precision)
