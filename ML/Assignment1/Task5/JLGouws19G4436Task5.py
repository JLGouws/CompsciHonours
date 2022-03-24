#sorry about the lack of comments here, I mv'ed a file and deleted my answer to
#this question on Monday midnight
import pandas as pd
import numpy as np
import cv2

from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
#plotting preferences
sns.set_style('darkgrid')

plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('legend', title_fontsize=14)
plt.rc('font', size=13)

colours = sns.color_palette('deep')
###############################################################################
plt.style.use('ggplot')

# Loading the iris dataset
bcdata = datasets.load_breast_cancer()

data = pd.DataFrame(bcdata.data, columns = bcdata.feature_names)
data['malignancy'] = bcdata.target_names[bcdata.target]

plt.figure(figsize = (10, 10), tight_layout = True)
ax = sns.scatterplot(data=data, x='mean perimeter', y='mean area', 
        palette='deep', hue = 'malignancy')
plt.savefig("perimeterVArea.pdf")

plt.figure(figsize = (10, 10), tight_layout = True)
ax = sns.scatterplot(data=data, x='mean smoothness', y='mean area', 
        palette='deep', hue = 'malignancy')
plt.savefig("smoothVArea.pdf")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data[["mean perimeter", "mean area"]].to_numpy(), 
    data['malignancy'].to_numpy(), test_size=0.8, random_state=42
)

# Training the LR model
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred_test = lr.predict(X_test)
f = open("accuracy.tex", 'w')
f.write(f'%.4f'%metrics.accuracy_score(y_test, y_pred_test))
f.close()

f = open("precision.tex", 'w')
f.write(f'%.4f'%metrics.precision_score(y_test, y_pred_test, pos_label='malignant'))
f.close()

f = open("recall.tex", 'w')
f.write(f'%.4f'%metrics.recall_score(y_test, y_pred_test, pos_label='malignant'))
f.close()

f = open("f1.tex", 'w')
f.write(f'%.4f'%metrics.f1_score(y_test, y_pred_test, pos_label='malignant'))
f.close()

confM = metrics.confusion_matrix(y_test, y_pred_test, labels = ['malignant', 'benign'])

plt.figure(figsize = (10, 10), tight_layout = True)
ax = sns.heatmap(data=confM, annot = True, fmt = 'g', square = True, cbar = True,
        annot_kws={'size' : 10}, cmap = sns.cubehelix_palette(start=.5, rot=-.75, 
        as_cmap=True), xticklabels = ['malignant', 'benign'], yticklabels = ['malignant', 'benign'])

ax.set(xlabel = "Predicted Labels", ylabel = "Actual Labels")
plt.savefig("confM.pdf")
