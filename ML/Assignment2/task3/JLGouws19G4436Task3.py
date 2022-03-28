#sorry about the lack of comments here, I mv'ed a file and deleted my answer to
#this question on Monday midnight
import pandas as pd
import numpy as np
import cv2

from sklearn import datasets
from sklearn import model_selection
import sklearn.preprocessing as prep
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier

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


# Split the data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data.drop('malignancy', axis = 1).to_numpy(), 
    data['malignancy'].to_numpy(), test_size=0.2, random_state=42
)

trainDf = pd.DataFrame(X_train, columns = bcdata.feature_names)                
trainDf['malignancy'] = y_train

#plt.figure(figsize = (10, 10), tight_layout = True)
#ax = sns.scatterplot(data=trainDf, x='mean perimeter', y='mean area', 
#        palette='deep', hue = 'malignancy')
#plt.savefig("perimeterVArea.pdf")

#plt.figure(figsize = (10, 10), tight_layout = True)
#ax = sns.scatterplot(data=trainDf, x='mean smoothness', y='mean area', 
#        palette='deep', hue = 'malignancy')
#plt.savefig("smoothVArea.pdf")

plt.figure(figsize = (30, 30), tight_layout = True)
sns.heatmap(trainDf.corr(), square = True, cbar = True, annot = True, annot_kws={'size' : 10})
plt.savefig("correlationMap.pdf")

scaler = prep.MinMaxScaler()
transformer = scaler.fit(X_train)
scaledData = transformer.transform(X_train) #scale data for further comparison
                                            #I used min max scaler to avoid distribution changes

scaledTrainDf = pd.DataFrame(scaledData, columns = bcdata.feature_names)                
scaledTrainDf['malignancy'] = y_train
scaledTrainDfMelted = pd.melt(scaledTrainDf, id_vars="malignancy", 
                                        var_name = "features", value_name = 'value')

plt.figure(figsize=(40,14))
sns.violinplot(x="features", y="value", hue="malignancy", data=scaledTrainDfMelted,split=True, inner="quart")
plt.xticks(rotation=30)
plt.savefig("ViolinPlot.pdf")

for i, features in enumerate(np.array_split(bcdata.feature_names,3)):
    plt.figure(figsize=(30,12))
#sns.swarmplot(x="features", y="value", hue="malignancy", data=scaledTrainDfMelted, alpha = 0.5) #
    sns.violinplot(x="features", y="value", hue="malignancy",
            data=scaledTrainDfMelted[scaledTrainDfMelted["features"].isin(features)]) #

    plt.xticks(rotation=30)

    plt.savefig(f"ViolinPlot{i}.pdf")

for i, features in enumerate(np.array_split(bcdata.feature_names,3)):
    plt.figure(figsize=(30,12))
    #sns.swarmplot(x="features", y="value", hue="malignancy", data=scaledTrainDfMelted, alpha = 0.5) #
    ############################################################
    #strip plot is like a swarm plot but it does not bunch points on the edges
    # this makes the graph easier to read
    sns.stripplot(x="features", y="value", hue="malignancy", jitter = 0.2,
            data=scaledTrainDfMelted[scaledTrainDfMelted["features"].isin(features)], alpha = 0.8) #

    plt.xticks(rotation=30)

    plt.savefig(f"swarmPlot{i}.pdf")


# got this from https://www.kaggle.com/code/alibaris/feature-selection-and-random-forest-breast-cancer
corr = scaledTrainDf.drop("malignancy", axis = 1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(20, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True,fmt='.2f',mask=mask, cmap=cmap, ax=ax);
plt.savefig("correlationMap.pdf")

bcdFeatureDropList = ['mean perimeter','mean radius','mean compactness',
                        'mean concave points','radius error','perimeter error',
                        'worst radius','worst perimeter','worst compactness',
                        'worst concave points','compactness error',
                        'concave points error','worst texture','worst area']
selectedTrain = scaledTrainDf.drop(bcdFeatureDropList, axis = 1 )       


corr = selectedTrain.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(12, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True,fmt='.2f',mask=mask, cmap=cmap, ax=ax);
plt.savefig("correlationMapSelected.pdf")

X_train_selected, y_train_selected = selectedTrain.drop('malignancy', axis = 1).to_numpy(), \
                    selectedTrain['malignancy'].to_numpy()

scaledTestData = transformer.transform(X_test)#use the same scaler as before to scale test data 

scaledTestDf = pd.DataFrame(scaledTestData, columns = bcdata.feature_names)                
scaledTestDf['malignancy'] = y_test

selectedTest = scaledTestDf.drop(bcdFeatureDropList, axis = 1 )       

X_test_selected, y_test_selected = selectedTest.drop('malignancy', axis = 1).to_numpy(), \
                    selectedTest['malignancy'].to_numpy()

#n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=42)      
clr_rf = clf_rf.fit(X_train_selected, y_train_selected)

ac_score = accuracy_score(y_test_selected,clf_rf.predict(X_test_selected))
print('Accuracy is: ',ac_score)

cnf_m = confusion_matrix(y_test_selected,clf_rf.predict(X_test_selected))

plt.figure(figsize=(3,3))
sns.heatmap(cnf_m, annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Base Model', color='navy', fontsize=15)
plt.show()
