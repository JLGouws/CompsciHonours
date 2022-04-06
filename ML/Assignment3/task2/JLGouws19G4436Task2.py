# ---------------------------------------------------------------------------------------------------------------
# Task 2 and 3 skeleton

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

import skimage
import cv2
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
#cv2.imshow("for those of you that cannot wait for img proc.", image_dataset.images[100])
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
print(gridSearchResults.apply(
        lambda x: "{mean_test_score:#0.3f} (+/-{std_test_score:#0.03f}) for {params}".format(**x), 1
        ).to_frame().to_string(index=False, max_colwidth=-1, header=False))
print()
ax = sns.barplot(x="Kernel", y="mean_test_score", data = gridSearchResults)
plt.savefig("KernelVsPrecision.png")

