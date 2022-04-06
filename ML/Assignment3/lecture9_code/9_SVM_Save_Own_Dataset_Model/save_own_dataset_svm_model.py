# ---------------------------------------------------------------------------------------------------------------
#Load your own image files with categories as subfolder names
# This example assumes that the images are preprocessed, and classifies using tuned SVM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

import skimage
from skimage.io import imread
from skimage.transform import resize
# For model saving and loading:
import os
import joblib

# load images as 60 x 60:
def load_image_files(container_path, dimension=(60, 60)):

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

image_dataset = load_image_files("images/")


# Split data, but randomly allocate to training/test sets
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.5,random_state=42)

# Train data with parameter optimization for linear and Gaussian
tune_param =[
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ]

# Output a pickle file for the model if it does not exist
if(not os.path.isfile('saved_model.pkl')):
    print("no existo...saving model")
    svc = svm.SVC()
    grid = GridSearchCV(svc, tune_param, cv=3)
    grid.fit(X_train, y_train)
    joblib.dump(grid, 'saved_model.pkl')
    modelLoad = grid
else:
    print("already exists...loading model")
    modelLoad = joblib.load('saved_model.pkl')

# Predict
y_pred = modelLoad.predict(X_test)

# Evaluate
print("Classification report for - \n{}:\n{}\n".format(
    modelLoad, metrics.classification_report(y_test, y_pred)))

print("Accuracy",metrics.accuracy_score(y_test, y_pred))
