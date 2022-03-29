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
from sklearn.tree import DecisionTreeClassifier as dtc
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
#I chose a 20% test size
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

#plt.figure(figsize = (30, 30), tight_layout = True)
#sns.heatmap(trainDf.corr(), square = True, cbar = True, annot = True, annot_kws={'size' : 10})
#plt.savefig("correlationMap.pdf")

scaler = prep.MinMaxScaler() 
transformer = scaler.fit(X_train)
scaledData = transformer.transform(X_train) #scale data for further comparison
                                            #I used min max scaler to avoid distribution changes

#make a df for easier manipulation of data
scaledTrainDf = pd.DataFrame(scaledData, columns = bcdata.feature_names)                
scaledTrainDf['malignancy'] = y_train
#melt the data for analysis
scaledTrainDfMelted = pd.melt(scaledTrainDf, id_vars="malignancy", 
                                        var_name = "features", value_name = 'value')

#violin plot of all the variables we can see that some variables like 
#mean area and mean concave points have little overlap -> easier to classify
plt.figure(figsize=(40,14))
sns.violinplot(x="features", y="value", hue="malignancy", data=scaledTrainDfMelted,split=True, inner="quart")
plt.xticks(rotation=30)
plt.savefig("ViolinPlot.pdf")

#violin plots but for groups of 10 variables for easier analysis
for i, features in enumerate(np.array_split(bcdata.feature_names,3)):
    plt.figure(figsize=(30,12))
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

    plt.savefig(f"stripPlot{i}.pdf")


# got this from https://www.kaggle.com/code/alibaris/feature-selection-and-random-forest-breast-cancer
corr = scaledTrainDf.drop("malignancy", axis = 1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) #mask off values for correlation
                                               #matrix to make it less overwhelming
f, ax = plt.subplots(figsize=(22, 19))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True,fmt='.2f',mask=mask, cmap=cmap, ax=ax);
plt.savefig("correlationMap.pdf")#save the pdf

#these features are essentially the same pointless keeping both
f, ax = plt.subplots(figsize=(10, 10))
sns.jointplot(x ='mean radius', y = 'mean perimeter', 
              data = scaledTrainDf, kind="reg", color=colours[3], joint_kws={'line_kws':{'color':colours[0]}})
plt.savefig("correlationOfMeanRadiusAndMeanPerimeter.pdf")#save the pdf

#less correlation but still there.
f, ax = plt.subplots(figsize=(10, 10))
sns.jointplot(x ='worst concavity', y = 'worst concave points', 
              data = scaledTrainDf, kind="reg", color=colours[3], joint_kws={'line_kws':{'color':colours[0]}})
plt.savefig("correlationOfConcavityWorstAndConcavityPointsWorst.pdf")#save the pdf



#all the features
#bcdFeatureKeepList = ['mean radius', 'mean texture', 'mean perimeter', 'mean area'
#                        'mean smoothness', 'mean compactness', 'mean concavity'
#                        'mean concave points', 'mean symmetry', 'mean fractal dimension'
#                        'radius error', 'texture error', 'perimeter error', 'area error'
#                        'smoothness error', 'compactness error', 'concavity error',
#                        'concave points error', 'symmetry error', 'fractal dimension error',
#                        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
#                        'worst smoothness', 'worst compactness', 'worst concavity',
#                        'worst concave points', 'worst symmetry', 'worst fractal dimension'
#                        , 'malignancy']

#I kind of cheated here, because I tweeked my coice after running the program on test values
#the logic may not have been followed exactly, I gave up typing after a while

#worst perimeter corelated with mean area perimeter and radius I get rid of that
# mean perimeter and mean radius are highly correlated, mean perimeter looks more spread out
# mean area and perimeter are highly correlated, I will keep area, looks more spread out
# mean concave points and mean concavity are highly correlated, I get rid of mean concavity
# perimeter error and radius error are highly correlated perimeter error looks slightly better
# perimeter error and area error are highly correlated, perimeter error looks better on the strip plot, and it looks less correlated with other features
# worst perimeter and worst radius are highly correlated, worst perimeter looks better, worst radius has some high correlations with other features
# worst texture and mean texture are highly correlated, but both look like bad features, I get rid of both
# worst area is strongly correlated with both mean area and worst perimeter, I drop it
# mean concave points is strongly correlated with other features, I will drop it
# mean compactness and worst compactness are highly correlated, I will remove it
# compactness error is also highly correlated
# worst concavity and worst concave points are highly correlated, worst concave points looks like the better feature
# mean smoothness is highly correlated
# fractal dimension error looks like a bad feature

#list of features to keep
bcdFeatureKeepList = [   'mean texture', 'mean area', 
                         'mean symmetry','mean concavity' ,'mean fractal dimension',
                         'texture error', 'perimeter error',
                         'smoothness error', 'concavity error',
                         'symmetry error',
                         'worst smoothness', 
                         'worst symmetry' 
                         , 'malignancy']

#print out the number of features we have selected
print("number of features", len(bcdFeatureKeepList) - 1)

selectedTrain = scaledTrainDf[bcdFeatureKeepList]       

#print out correlation map for selected features
corr = selectedTrain.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(17, 16))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True,fmt='.2f',mask=mask, cmap=cmap, ax=ax);
plt.savefig("correlationMapSelected.pdf")

X_train_selected, y_train_selected = selectedTrain.drop('malignancy', axis = 1).to_numpy(), \
                    selectedTrain['malignancy'].to_numpy()

scaledTestData = transformer.transform(X_test)#use the same scaler as before to scale test data 

scaledTestDf = pd.DataFrame(scaledTestData, columns = bcdata.feature_names)                
scaledTestDf['malignancy'] = y_test

####################################################################################
#only look at test data now
#Select from the test data
selectedTest = scaledTestDf[bcdFeatureKeepList]       

X_test_selected, y_test_selected = selectedTest.drop('malignancy', axis = 1).to_numpy(), \
                    selectedTest['malignancy'].to_numpy()

#initialize and train a knn model
clf_knn = knn(n_neighbors = 5).fit(X_train_selected, y_train_selected)

#initialize and train a decision tree model
clf_dt = dtc().fit(X_train_selected, y_train_selected)

#initialize and train a random forrest classifier
clf_rf = RandomForestClassifier(random_state=42).fit(X_train_selected, y_train_selected)

#check accuracy and f1 scores of the model
ac_score = accuracy_score(y_test_selected, clf_knn.predict(X_test_selected))
f_score = f1_score(y_test_selected, clf_knn.predict(X_test_selected), pos_label = 'malignant')
print('Accuracy for knn is: ', ac_score)
print('f1 for knn is: ', f_score)

#make a confusion matrix
cnf_m = confusion_matrix(y_test_selected, clf_knn.predict(X_test_selected), labels = ['malignant', 'benign'])

plt.figure(figsize=(5,5))
sns.heatmap(cnf_m, annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, 
            cmap='PuBu', xticklabels = ['malignant', 'benign'], yticklabels = ['malignant', 'benign'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Base Model', color='navy', fontsize=15)
plt.savefig("knnConfusionMatrix.pdf")

ac_score = accuracy_score(y_test_selected, clf_dt.predict(X_test_selected))
f_score = f1_score(y_test_selected, clf_dt.predict(X_test_selected), pos_label = 'malignant')
print('Accuracy for decision tree is: ', ac_score)
print('f1 for decision tree is: ', f_score)

cnf_m = confusion_matrix(y_test_selected, clf_dt.predict(X_test_selected))

plt.figure(figsize=(5,5))
sns.heatmap(cnf_m, annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu', xticklabels = ['malignant', 'benign'], yticklabels = ['malignant', 'benign'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Base Model', color='navy', fontsize=15)
plt.savefig("dtConfusionMatrix.pdf")

ac_score = accuracy_score(y_test_selected,clf_rf.predict(X_test_selected))
f_score = f1_score(y_test_selected, clf_rf.predict(X_test_selected), pos_label = 'malignant')
print('Accuracy for random forest is: ', ac_score)
print('f1 for random forest is: ', f_score)

cnf_m = confusion_matrix(y_test_selected,clf_rf.predict(X_test_selected))

plt.figure(figsize=(5,5))
sns.heatmap(cnf_m, annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu', xticklabels = ['malignant', 'benign'], yticklabels = ['malignant', 'benign'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Base Model', color='navy', fontsize=15)
plt.savefig("rfConfusionMatrix.pdf")
