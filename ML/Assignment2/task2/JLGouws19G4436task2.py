import matplotlib.pyplot as plt
import numpy as np
from numpy import nan
import pandas as pd
import seaborn as sns
import sklearn.model_selection as mselect
from sklearn.neighbors import KNeighborsClassifier as knn
import sklearn.preprocessing as prep
import sklearn.metrics as metrics

plt.style.use('seaborn-darkgrid')

# Note that Best Practices, in terms of order of steps were violated in this program, please fix that.

# # Data Acquisition (I broke the CSV on purpose)
wine_df = pd.read_csv('winequality-red.csv')
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):

print("Number of missing entries per feature before cleaning:")
print(wine_df.isnull().sum())
wine_df.fillna(wine_df.median(), inplace = True)
print("Number of missing entries after cleaning: ", end ="")
print(wine_df.isnull().sum().sum())

X = wine_df.drop('quality', axis=1).values
y = np.ravel(wine_df[['quality']])

#split the data here do not look at test values
X_train, X_test, y_train, y_test = mselect.train_test_split(X, y, test_size=0.3, random_state=42)
wine_train_df = pd.DataFrame(np.c_[X_train, y_train], columns=wine_df.columns)
print("Training Dataframe")
print(wine_train_df.head(3))

# # Cleanup

# # Exploratory Data Analysis
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(wine_df.describe())
# sns.pairplot(wine_df, hue = 'quality', height = 3, palette="husl")
# sns.violinplot(data=wine_df, x='quality', y='alcohol')
sns.FacetGrid(wine_train_df, hue='quality', height=6).map(plt.scatter, 'alcohol', 'fixed acidity').add_legend()
#plt.show()


# ### Distribution of wine quality (target variable)
plt.hist(wine_train_df['quality'], bins=6, edgecolor='black')
plt.xlabel('quality', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.show()

scalerList = [("FunctionTransformer", "No Scaling"), ("MinMaxScaler", "Min Max Scaler"), 
              ("RobustScaler", "Min Max Scaler"), ("StandardScaler", "Standard Scaler"), 
              ("Normalizer", "Normalizer")]

for scalerTup in scalerList:
    scaler = getattr(prep, scalerTup[0])()
    transformer = scaler.fit(X_train)    # fit data for scaler
    scaledData = transformer.transform(X_train)
    knnEstimator = knn(n_neighbors = 1)
    cvresult = mselect.cross_validate(estimator = knnEstimator, X = X_train, 
            y = y_train, cv = 5, n_jobs = -1, 
            scoring = ('accuracy', 'f1_weighted')) #cross validate with required metrics
    print(scalerTup[1] ,cvresult['test_accuracy'],cvresult['test_f1_weighted'])

    scaledTest = transformer.transform(X_test)
    y_pred = knn(n_neighbors = 1).fit(scaledData, y_train).predict(scaledTest)
    print(scalerTup[1])
    print(metrics.classification_report(y_test, y_pred, zero_division = 0))
    reportDict = metrics.classification_report(y_test, y_pred, zero_division = 0, output_dict = True)
    reportDict['accuracy'] = {'precision': ' ', 'recall': ' ', 'f1-score': reportDict['accuracy'], 'support': reportDict['weighted avg']['support']}
    reportDf = pd.DataFrame(data = reportDict).transpose()
    print(reportDf)

y_train_string = y_train.astype(str)
y_train_string[np.any([y_train == 3,  y_train == 4], axis = 0)] = "bad"
y_train_string[np.any([y_train == 5,  y_train == 6], axis = 0)] = "average"
y_train_string[np.any([y_train == 7,  y_train == 8], axis = 0)] = "good"

y_test_string = y_test.astype(str)
y_test_string[np.any([y_test == 3,  y_test == 4], axis = 0)] = "bad"
y_test_string[np.any([y_test == 5,  y_test == 6], axis = 0)] = "average"
y_test_string[np.any([y_test == 7,  y_test == 8], axis = 0)] = "good"

for scalerTup in scalerList:
    scaler = getattr(prep, scalerTup[0])()
    transformer = scaler.fit(X_train)    # fit data for scaler
    scaledData = transformer.transform(X_train)
    knnEstimator = knn(n_neighbors = 1)
    cvresult = mselect.cross_validate(estimator = knnEstimator, X = X_train, 
            y = y_train_string, cv = 5, n_jobs = -1, 
            scoring = ('accuracy', 'f1_weighted')) #cross validate with required metrics
    print(scalerTup[1] ,cvresult['test_accuracy'],cvresult['test_f1_weighted'])

    scaledTest = transformer.transform(X_test)
    y_pred_string = knn(n_neighbors = 1).fit(scaledData, y_train_string).predict(scaledTest)
    print(scalerTup[1])
    print(metrics.classification_report(y_test_string, y_pred_string, zero_division = 0))
    reportDict = metrics.classification_report(y_test_string, y_pred_string, zero_division = 0, output_dict = True)
    reportDict['accuracy'] = {'precision': ' ', 'recall': ' ', 'f1-score': reportDict['accuracy'], 'support': reportDict['weighted avg']['support']}
    reportDf = pd.DataFrame(data = reportDict).transpose()
    print(reportDf)
