import matplotlib.pyplot as plt
import numpy as np
from numpy import nan
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
wine_train_df = pd.DataFrame(np.c_[X_train, y_train], columns=wine_df.columns)
print(wine_train_df.head(3))

# # Cleanup

# # Exploratory Data Analysis
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(wine_df.describe())
# sns.pairplot(wine_df, hue = 'quality', height = 3, palette="husl")
# sns.violinplot(data=wine_df, x='quality', y='alcohol')
sns.FacetGrid(wine_train_df, hue='quality', height=6).map(plt.scatter, 'alcohol', 'fixed acidity').add_legend()
plt.show()


# ### Distribution of wine quality (target variable)
plt.hist(wine_train_df['quality'], bins=6, edgecolor='black')
plt.xlabel('quality', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

