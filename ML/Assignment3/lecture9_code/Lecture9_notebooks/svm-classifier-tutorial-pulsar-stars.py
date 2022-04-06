#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # **Support Vector Machines Classifier Tutorial with Python** 
# 
# Hello friends,
# 
# Support Vector Machines (SVMs in short) are supervised machine learning algorithms that are used for classification and regression purposes. In this kernel, I build a Support Vector Machines classifier to classify a Pulsar star. I have used the **Predicting a Pulsar Star** dataset for this project. 
# 
# So, let's get started.

# **As always, I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**.

# <a class="anchor" id="0.1"></a>
# # **Table of Contents**
# 
# 
# 1.	[Introduction to Support Vector Machines](#1)
# 2.	[Support Vector Machines intuition](#2)
# 3.	[Kernel trick](#3)
# 4.	[SVM Scikit-Learn libraries](#4)
# 5.	[Dataset description](#5)
# 6.	[Import libraries](#6)
# 7.	[Import dataset](#7)
# 8.	[Exploratory data analysis](#8)
# 9.	[Declare feature vector and target variable](#9)
# 10.	[Split data into separate training and test set](#10)
# 11.	[Feature scaling](#11)
# 12.	[Run SVM with default hyperparameters](#12)
# 13.	[Run SVM with linear kernel](#13)
# 14.	[Run SVM with polynomial kernel](#14)
# 15.	[Run SVM with sigmoid kernel](#15)
# 16.	[Confusion matrix](#16)
# 17.	[Classification metrices](#17)
# 18.	[ROC - AUC](#18)
# 19.	[Stratified k-fold Cross Validation with shuffle split](#19)
# 20.	[Hyperparameter optimization using GridSearch CV](#20)
# 21.	[Results and conclusion](#21)
# 22. [References](#22)
# 

# # **1. Introduction to Support Vector Machines** <a class="anchor" id="1"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# **Support Vector Machines** (SVMs in short) are machine learning algorithms that are used for classification and regression purposes. SVMs are one of the powerful machine learning algorithms for classification, regression and outlier detection purposes. An SVM classifier builds a model that assigns new data points to one of the given categories. Thus, it can be viewed as a non-probabilistic binary linear classifier.
# 
# The original SVM algorithm was developed by Vladimir N Vapnik and Alexey Ya. Chervonenkis in 1963. At that time, the algorithm was in early stages. The only possibility is to draw hyperplanes for linear classifier. In 1992, Bernhard E. Boser, Isabelle M Guyon and Vladimir N Vapnik suggested a way to create non-linear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.
# 
# SVMs can be used for linear classification purposes. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using the **kernel trick**. It enable us to implicitly map the inputs into high dimensional feature spaces.
# 
# 
# 

# # **2. Support Vector Machines intuition** <a class="anchor" id="2"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# Now, we should be familiar with some SVM terminology. 
# 
# 
# ### Hyperplane
# 
# A hyperplane is a decision boundary which separates between given set of data points having different class labels. The SVM classifier separates data points using a hyperplane with the maximum amount of margin. This hyperplane is known as the `maximum margin hyperplane` and the linear classifier it defines is known as the `maximum margin classifier`.
# 
# 
# ### Support Vectors
# 
# Support vectors are the sample data points, which are closest to the hyperplane.  These data points will define the separating line or hyperplane better by calculating margins.
# 
# 
# ### Margin
# 
# A margin is a separation gap between the two lines on the closest data points. It is calculated as the perpendicular distance from the line to support vectors or closest data points. In SVMs, we try to maximize this separation gap so that we get maximum margin.
# 
# The following diagram illustrates these concepts visually.
# 
# 
# ### Margin in SVM
# 
# ![Margin in SVM](https://static.wixstatic.com/media/8f929f_7ecacdcf69d2450087cb4a898ef90837~mv2.png)
# 
# 
# ### SVM Under the hood
# 
# In SVMs, our main objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset. SVM searches for the maximum margin hyperplane in the following 2 step process –
# 
# 
# 1.	Generate hyperplanes which segregates the classes in the best possible way. There are many hyperplanes that might classify the data. We should look for the best hyperplane that represents the largest separation, or margin, between the two classes.
# 
# 2.	So, we choose the hyperplane so that distance from it to the support vectors on each side is maximized. If such a hyperplane exists, it is known as the **maximum margin hyperplane** and the linear classifier it defines is known as a **maximum margin classifier**. 
# 
# 
# The following diagram illustrates the concept of **maximum margin** and **maximum margin hyperplane** in a clear manner.
# 
# 
# ### Maximum margin hyperplane
# 
# ![Maximum margin hyperplane](https://static.packt-cdn.com/products/9781783555130/graphics/3547_03_07.jpg)
# 
# 
# 
# ### Problem with dispersed datasets
# 
# 
# Sometimes, the sample data points are so dispersed that it is not possible to separate them using a linear hyperplane. 
# In such a situation, SVMs uses a `kernel trick` to transform the input space to a higher dimensional space as shown in the diagram below. It uses a mapping function to transform the 2-D input space into the 3-D input space. Now, we can easily segregate the data points using linear separation.
# 
# 
# ### Kernel trick - transformation of input space to higher dimensional space
# 
# ![Kernel trick](http://www.aionlinecourse.com/uploads/tutorials/2019/07/11_21_kernel_svm_3.png)
# 
# 

# # **3. Kernel trick** <a class="anchor" id="3"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# In practice, SVM algorithm is implemented using a `kernel`. It uses a technique called the `kernel trick`. In simple words, a `kernel` is just a function that maps the data to a higher dimension where data is separable. A kernel transforms a low-dimensional input data space into a higher dimensional space. So, it converts non-linear separable problems to linear separable problems by adding more dimensions to it. Thus, the kernel trick helps us to build a more accurate classifier. Hence, it is useful in non-linear separation problems.

# We can define a kernel function as follows-
# 
# 
# ### Kernel function
# 
# ![Kernel function](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTodZptqcRor0LGo8Qn7_kJB9n9BACMt6jgIPZ4C3g_rgh_uSRZLQ&s)

# In the context of SVMs, there are 4 popular kernels – `Linear kernel`,`Polynomial kernel`,`Radial Basis Function (RBF) kernel` (also called Gaussian kernel) and `Sigmoid kernel`. These are described below -

# ## **3.1 Linear kernel**
# 
# In linear kernel, the kernel function takes the form of a linear function as follows-
# 
# **linear kernel : K(xi , xj ) = xiT xj**
# 
# Linear kernel is used when the data is linearly separable. It means that data can be separated using a single line. It is one of the most common kernels to be used. It is mostly used when there are large number of features in a dataset. Linear kernel is often used for text classification purposes.
# 
# Training with a linear kernel is usually faster, because we only need to optimize the C regularization parameter. When training with other kernels, we also need to optimize the γ parameter. So, performing a grid search will usually take more time.
# 
# Linear kernel can be visualized with the following figure.
# 
# ### Linear Kernel
# 
# ![Linear Kernel](https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_kernels_thumb.png)

# ## **3.2 Polynomial Kernel**
# 
# Polynomial kernel represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables. The polynomial kernel looks not only at the given features of input samples to determine their similarity, but also combinations of the input samples.
# 
# For degree-d polynomials, the polynomial kernel is defined as follows –
# 
# **Polynomial kernel : K(xi , xj ) = (γxiT xj + r)d , γ > 0**
# 
# Polynomial kernel is very popular in Natural Language Processing. The most common degree is d = 2 (quadratic), since larger degrees tend to overfit on NLP problems. It can be visualized with the following diagram.
# 
# ### Polynomial Kernel
# 
# ![Polynomial Kernel](https://www.researchgate.net/profile/Cheng_Soon_Ong/publication/23442384/figure/fig12/AS:341444054274063@1458418014823/The-effect-of-the-degree-of-a-polynomial-kernel-The-polynomial-kernel-of-degree-1-leads.png)

# ## **3.3 Radial Basis Function Kernel**
# 
# Radial basis function kernel is a general purpose kernel. It is used when we have no prior knowledge about the data. The RBF kernel on two samples x and y is defined by the following equation –
# 
# 
# ### Radial Basis Function kernel
# 
# ![RBK Kernel](data:image/png;base64,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)

# The following diagram demonstrates the SVM classification with rbf kernel.
# 
# ### SVM Classification with rbf kernel
# 
# ![SVM Classification with rbf kernel](https://www.researchgate.net/profile/Periklis_Gogas/publication/286180566/figure/fig5/AS:304327777374210@1449568804246/An-example-of-an-SVM-classification-using-the-RBF-kernel-The-two-classes-are-separated.png)

# ## **3.4 Sigmoid kernel**
# 
# Sigmoid kernel has its origin in neural networks. We can use it as the proxy for neural networks. Sigmoid kernel is given by the following equation –
# 
# **sigmoid kernel : k (x, y) = tanh(αxTy + c)**

# Sigmoid kernel can be visualized with the following diagram-
# 
# ### Sigmoid kernel
# 
# ![Sigmoid kernel](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTKeXbOIlniBXYwMYlEYLKPwZZg8vFU1wVm3RWMACjVcT4iBVDy&s)

# # **4. SVM Scikit-Learn libraries** <a class="anchor" id="4"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# Scikit-Learn provides useful libraries to implement Support Vector Machine algorithm on a dataset. There are many libraries that can help us to implement SVM smoothly. We just need to call the library with parameters that suit to our needs. In this project, I am dealing with a classification task. So, I will mention the Scikit-Learn libraries for SVM classification purposes.
# 
# First, there is a **LinearSVC()** classifier. As the name suggests, this classifier uses only linear kernel. In LinearSVC() classifier, we don’t pass the value of kernel since it is used only for linear classification purposes.
# 
# Scikit-Learn provides two other classifiers - **SVC()** and **NuSVC()** which are used for classification purposes. These classifiers are mostly similar with some difference in parameters. **NuSVC()** is similar to **SVC()** but uses a parameter to control the number of support vectors. We pass the values of kernel, gamma and C along with other parameters. By default kernel parameter uses rbf as its value but we can pass values like poly, linear, sigmoid or callable function.

# # **5. Dataset description** <a class="anchor" id="5"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# I have used the **Predicting a Pulsar Star** dataset for this project.
# 
# Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter. Classification algorithms in particular are being adopted, which treat the data sets as binary classification problems. Here the legitimate pulsar examples form  minority positive class and spurious examples form the majority negative class.
# 
# The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive).
# 
# 
# ### Attribute Information:
# 
# 
# Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile. The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:
# 
# 1. Mean of the integrated profile.
# 
# 2. Standard deviation of the integrated profile.
# 
# 3. Excess kurtosis of the integrated profile.
# 
# 4. Skewness of the integrated profile.
# 
# 5. Mean of the DM-SNR curve.
# 
# 6. Standard deviation of the DM-SNR curve.
# 
# 7. Excess kurtosis of the DM-SNR curve.
# 
# 8. Skewness of the DM-SNR curve.
# 
# 9. Class

# # **6. Import libraries** <a class="anchor" id="6"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# I will start off by importing the required Python libraries.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# # **7.Import dataset** <a class="anchor" id="7"></a>
# 
# [Table of Contents](#0.1)
# 
# ![image.png](attachment:0d95f485-1242-4c46-a888-e3536634bc02.png)![image.png](attachment:b7988564-756f-4807-a596-3e51fb924404.png)
# 
# 

# In[3]:


# data = '/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv'
data = '/kaggle/input/pulsar-star/pulsar_stars.csv'


df = pd.read_csv(data)


# # **8. Exploratory data analysis** <a class="anchor" id="8"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# Now, I will explore the data to gain insights about the data. 

# In[4]:


# view dimensions of dataset

df.shape


# We can see that there are 17898 instances and 9 variables in the data set.

# In[5]:


# let's preview the dataset

df.head()


# We can see that there are 9 variables in the dataset. 8 are continuous variables and 1 is discrete variable. The discrete variable is `target_class` variable. It is also the target variable.
# 
# 
# Now, I will view the column names to check for leading and trailing spaces.

# In[6]:


# view the column names of the dataframe

col_names = df.columns

col_names


# We can see that there are leading spaces (spaces at the start of the string name) in the dataframe. So, I will remove these leading spaces.

# In[7]:


# remove leading spaces from column names

df.columns = df.columns.str.strip()


# I have removed the leading spaces from the column names. Let's again view the column names to confirm the same.

# In[8]:


# view column names again

df.columns


# We can see that the leading spaces are removed from the column name. But the column names are very long. So, I will make them short by renaming them.

# In[9]:


# rename column names

df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']


# In[10]:


# view the renamed column names

df.columns


# We can see that the column names are shortened. IP stands for `integrated profile` and DM-SNR stands for `delta modulation and signal to noise ratio`. Now, it is much more easy to work with the columns.

# Our target variable is the `target_class` column. So, I will check its distribution.

# In[11]:


# check distribution of target_class column

df['target_class'].value_counts()


# In[12]:


# view the percentage distribution of target_class column

df['target_class'].value_counts()/np.float(len(df))


# We can see that percentage of observations of the class label `0` and `1` is 90.84% and 9.16%. So, this is a class imbalanced problem. I will deal with that in later section.

# In[13]:


# view summary of dataset

df.info()


# We can see that there are no missing values in the dataset and all the variables are numerical variables.

# ### Explore missing values in variables

# In[14]:


# check for missing values in variables

df.isnull().sum()


# We can see that there are no missing values in the dataset.

# ### Summary of numerical variables
# 
# 
# - There are 9 numerical variables in the dataset.
# 
# 
# - 8 are continuous variables and 1 is discrete variable. 
# 
# 
# - The discrete variable is `target_class` variable. It is also the target variable.
# 
# 
# - There are no missing values in the dataset.

# ### Outliers in numerical variables

# In[15]:


# view summary statistics in numerical variables

round(df.describe(),2)


# On closer inspection, we can suspect that all the continuous variables may contain outliers.
# 
# 
# I will draw boxplots to visualise outliers in the above variables. 

# In[16]:


# draw boxplots to visualize outliers

plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df.boxplot(column='IP Mean')
fig.set_title('')
fig.set_ylabel('IP Mean')


plt.subplot(4, 2, 2)
fig = df.boxplot(column='IP Sd')
fig.set_title('')
fig.set_ylabel('IP Sd')


plt.subplot(4, 2, 3)
fig = df.boxplot(column='IP Kurtosis')
fig.set_title('')
fig.set_ylabel('IP Kurtosis')


plt.subplot(4, 2, 4)
fig = df.boxplot(column='IP Skewness')
fig.set_title('')
fig.set_ylabel('IP Skewness')


plt.subplot(4, 2, 5)
fig = df.boxplot(column='DM-SNR Mean')
fig.set_title('')
fig.set_ylabel('DM-SNR Mean')


plt.subplot(4, 2, 6)
fig = df.boxplot(column='DM-SNR Sd')
fig.set_title('')
fig.set_ylabel('DM-SNR Sd')


plt.subplot(4, 2, 7)
fig = df.boxplot(column='DM-SNR Kurtosis')
fig.set_title('')
fig.set_ylabel('DM-SNR Kurtosis')


plt.subplot(4, 2, 8)
fig = df.boxplot(column='DM-SNR Skewness')
fig.set_title('')
fig.set_ylabel('DM-SNR Skewness')


# The above boxplots confirm that there are lot of outliers in these variables.

# ### Handle outliers with SVMs
# 
# 
# There are 2 variants of SVMs. They are `hard-margin variant of SVM` and `soft-margin variant of SVM`.
# 
# 
# The `hard-margin variant of SVM` does not deal with outliers. In this case, we want to find the hyperplane with maximum margin such that every training point is correctly classified with margin at least 1. This technique does not handle outliers well.
# 
# 
# Another version of SVM is called `soft-margin variant of SVM`. In this case, we can have a few points incorrectly classified or 
# classified with a margin less than 1. But for every such point, we have to pay a penalty in the form of `C` parameter, which controls the outliers. `Low C` implies we are allowing more outliers and `high C` implies less outliers.
# 
# 
# The message is that since the dataset contains outliers, so the value of C should be high while training the model.

# ### Check the distribution of variables
# 
# 
# Now, I will plot the histograms to check distributions to find out if they are normal or skewed. 

# In[17]:


# plot histogram to check distribution


plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df['IP Mean'].hist(bins=20)
fig.set_xlabel('IP Mean')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 2)
fig = df['IP Sd'].hist(bins=20)
fig.set_xlabel('IP Sd')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 3)
fig = df['IP Kurtosis'].hist(bins=20)
fig.set_xlabel('IP Kurtosis')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 4)
fig = df['IP Skewness'].hist(bins=20)
fig.set_xlabel('IP Skewness')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 5)
fig = df['DM-SNR Mean'].hist(bins=20)
fig.set_xlabel('DM-SNR Mean')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 6)
fig = df['DM-SNR Sd'].hist(bins=20)
fig.set_xlabel('DM-SNR Sd')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 7)
fig = df['DM-SNR Kurtosis'].hist(bins=20)
fig.set_xlabel('DM-SNR Kurtosis')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 8)
fig = df['DM-SNR Skewness'].hist(bins=20)
fig.set_xlabel('DM-SNR Skewness')
fig.set_ylabel('Number of pulsar stars')


# We can see that all the 8 continuous variables are skewed. 

# # **9. Declare feature vector and target variable** <a class="anchor" id="9"></a>
# 
# [Table of Contents](#0.1)

# In[18]:


X = df.drop(['target_class'], axis=1)

y = df['target_class']


# # **10. Split data into separate training and test set** <a class="anchor" id="10"></a>
# 
# [Table of Contents](#0.1)

# In[19]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[20]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# # **11. Feature Scaling** <a class="anchor" id="11"></a>
# 
# [Table of Contents](#0.1)

# In[21]:


cols = X_train.columns


# In[22]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[23]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[24]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[25]:


X_train.describe()


# We now have `X_train` dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.

# # **12. Run SVM with default hyperparameters** <a class="anchor" id="12"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# Default hyperparameter means C=1.0,  kernel=`rbf` and gamma=`auto` among other parameters.

# In[26]:


# import SVC classifier
from sklearn.svm import SVC


# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


# instantiate classifier with default hyperparameters
svc=SVC() 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ### Run SVM with rbf kernel and C=100.0
# 
# 
# We have seen that there are outliers in our dataset. So, we should increase the value of C as higher C means fewer outliers. 
# So, I will run SVM with kernel=`rbf` and C=100.0.

# In[27]:


# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# We can see that we obtain a higher accuracy with C=100.0 as higher C means less outliers.
# 
# Now, I will further increase the value of C=1000.0 and check accuracy.

# ### Run SVM with rbf kernel and C=1000.0
# 

# In[28]:


# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In this case, we can see that the accuracy had decreased with C=1000.0

# # **13. Run SVM with linear kernel** <a class="anchor" id="13"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# ### Run SVM with linear kernel and C=1.0

# In[29]:


# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0) 


# fit classifier to training set
linear_svc.fit(X_train,y_train)


# make predictions on test set
y_pred_test=linear_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# ### Run SVM with linear kernel and C=100.0

# In[30]:


# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0) 


# fit classifier to training set
linear_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ### Run SVM with linear kernel and C=1000.0

# In[31]:


# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0) 


# fit classifier to training set
linear_svc1000.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc1000.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# We can see that we can obtain higher accuracy with C=100.0 and C=1000.0 as compared to C=1.0.

# Here, **y_test** are the true class labels and **y_pred** are the predicted class labels in the test-set.

# ### Compare the train-set and test-set accuracy
# 
# 
# Now, I will compare the train-set and test-set accuracy to check for overfitting.

# In[32]:


y_pred_train = linear_svc.predict(X_train)

y_pred_train


# In[33]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# We can see that the training set and test-set accuracy are very much comparable.

# ### Check for overfitting and underfitting

# In[34]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(linear_svc.score(X_test, y_test)))


# The training-set accuracy score is 0.9783 while the test-set accuracy to be 0.9830. These two values are quite comparable. So, there is no question of overfitting. 
# 

# ### Compare model accuracy with null accuracy
# 
# 
# So, the model accuracy is 0.9832. But, we cannot say that our model is very good based on the above accuracy. We must compare it with the **null accuracy**. Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.
# 
# So, we should first check the class distribution in the test set. 

# In[35]:


# check class distribution in test set

y_test.value_counts()


# We can see that the occurences of most frequent class `0` is 3306. So, we can calculate null accuracy by dividing 3306 by total number of occurences.

# In[36]:


# check null accuracy score

null_accuracy = (3306/(3306+274))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# We can see that our model accuracy score is 0.9830 but null accuracy score is 0.9235. So, we can conclude that our SVM classifier is doing a very good job in predicting the class labels.

# # **14. Run SVM with polynomial kernel** <a class="anchor" id="14"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# ### Run SVM with polynomial kernel and C=1.0

# In[37]:


# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 


# fit classifier to training set
poly_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=poly_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


#  ### Run SVM with polynomial kernel and C=100.0

# In[38]:


# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0) 


# fit classifier to training set
poly_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=poly_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# Polynomial kernel gives poor performance. It may be overfitting the training set.

# # **15. Run SVM with sigmoid kernel** <a class="anchor" id="15"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# ### Run SVM with sigmoid kernel and C=1.0

# In[39]:


# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 


# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ### Run SVM with sigmoid kernel and C=100.0

# In[40]:


# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0) 


# fit classifier to training set
sigmoid_svc100.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# We can see that sigmoid kernel is also performing poorly just like with polynomial kernel.

# ### Comments
# 
# 
# We get maximum accuracy with `rbf` and `linear` kernel with C=100.0. and the accuracy is 0.9832. Based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
# 
# 
# But, this is not true. Here, we have an imbalanced dataset. The problem is that accuracy is an inadequate measure for quantifying predictive performance in the imbalanced dataset problem.
# 
# 
# So, we must explore alternative metrices that provide better guidance in selecting models. In particular, we would like to know the underlying distribution of values and the type of errors our classifer is making. 
# 
# 
# One such metric to analyze the model performance in imbalanced classes problem is `Confusion matrix`.

# # **16. Confusion matrix** <a class="anchor" id="16"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
# 
# 
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
# 
# 
# **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
# 
# 
# **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
# 
# 
# **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**
# 
# 
# 
# **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**
# 
# 
# 
# These four outcomes are summarized in a confusion matrix given below.
# 

# In[41]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# The confusion matrix shows `3289 + 230 = 3519 correct predictions` and `17 + 44 = 61 incorrect predictions`.
# 
# 
# In this case, we have
# 
# 
# - `True Positives` (Actual Positive:1 and Predict Positive:1) - 3289
# 
# 
# - `True Negatives` (Actual Negative:0 and Predict Negative:0) - 230
# 
# 
# - `False Positives` (Actual Negative:0 but Predict Positive:1) - 17 `(Type I error)`
# 
# 
# - `False Negatives` (Actual Positive:1 but Predict Negative:0) - 44 `(Type II error)`

# In[42]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # **17. Classification metrices** <a class="anchor" id="17"></a>
# 
# [Table of Contents](#0.1)

# ### Classification Report
# 
# 
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
# 
# We can print a classification report as follows:-

# In[43]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))


# ### Classification accuracy

# In[44]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[45]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# ### Classification error

# In[46]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# ### Precision
# 
# 
# **Precision** can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP). 
# 
# 
# So, **Precision** identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
# 
# 
# 
# Mathematically, precision can be defined as the ratio of `TP to (TP + FP)`.
# 
# 
# 

# In[47]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# ### Recall
# 
# 
# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes.
# It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). **Recall** is also called **Sensitivity**.
# 
# 
# **Recall** identifies the proportion of correctly predicted actual positives.
# 
# 
# Mathematically, **recall** can be defined as the ratio of `TP to (TP + FN)`.
# 
# 

# In[48]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# ### True Positive Rate
# 
# 
# **True Positive Rate** is synonymous with **Recall**.
# 

# In[49]:


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# ### False Positive Rate

# In[50]:


false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# ### Specificity

# In[51]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# ### f1-score
# 
# 
# **f1-score** is the weighted harmonic mean of precision and recall. The best possible **f1-score** would be 1.0 and the worst 
# would be 0.0.  **f1-score** is the harmonic mean of precision and recall. So, **f1-score** is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of `f1-score` should be used to 
# compare classifier models, not global accuracy.
# 

# ### Support
# 
# 
# **Support** is the actual number of occurrences of the class in our dataset.

# # **18. ROC - AUC** <a class="anchor" id="18"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# 
# ### ROC Curve
# 
# 
# Another tool to measure the classification model performance visually is **ROC Curve**. ROC Curve stands for **Receiver Operating Characteristic Curve**. An **ROC Curve** is a plot which shows the performance of a classification model at various 
# classification threshold levels. 
# 
# 
# 
# The **ROC Curve** plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels.
# 
# 
# 
# **True Positive Rate (TPR)** is also called **Recall**. It is defined as the ratio of `TP to (TP + FN)`.
# 
# 
# 
# **False Positive Rate (FPR)** is defined as the ratio of `FP to (FP + TN)`.
# 
# 
# 
# In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).
# 
# 

# In[52]:


# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a Pulsar Star classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.

# ### ROC  AUC
# 
# 
# **ROC AUC** stands for **Receiver Operating Characteristic - Area Under Curve**. It is a technique to compare classifier performance. In this technique, we measure the `area under the curve (AUC)`. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5. 
# 
# 
# So, **ROC AUC** is the percentage of the ROC plot that is underneath the curve.

# In[53]:


# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred_test)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# ### Comments
# 
# 
# - ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.
# 
# - ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in classifying the pulsar star.

# In[54]:


# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(linear_svc, X_train, y_train, cv=10, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# # **19. Stratified k-fold Cross Validation with shuffle split** <a class="anchor" id="19"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# k-fold cross-validation is a very useful technique to evaluate model performance. But, it fails here because we have a imbalnced dataset. So, in the case of imbalanced dataset, I will use another technique to evaluate model performance. It is called `stratified k-fold cross-validation`.
# 
# 
# In `stratified k-fold cross-validation`, we split the data such that the proportions between classes are the same in each fold as they are in the whole dataset.
# 
# 
# Moreover, I will shuffle the data before splitting because shuffling yields much better result.

# ### Stratified k-Fold Cross Validation with shuffle split with  linear kernel

# In[55]:


from sklearn.model_selection import KFold


kfold=KFold(n_splits=5, shuffle=True, random_state=0)


linear_svc=SVC(kernel='linear')


linear_scores = cross_val_score(linear_svc, X, y, cv=kfold)


# In[56]:


# print cross-validation scores with linear kernel

print('Stratified cross-validation scores with linear kernel:\n\n{}'.format(linear_scores))


# In[57]:


# print average cross-validation score with linear kernel

print('Average stratified cross-validation score with linear kernel:{:.4f}'.format(linear_scores.mean()))


# ### Stratified k-Fold Cross Validation with shuffle split with rbf kernel

# In[58]:


rbf_svc=SVC(kernel='rbf')


rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)


# In[59]:


# print cross-validation scores with rbf kernel

print('Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))


# In[60]:


# print average cross-validation score with rbf kernel

print('Average stratified cross-validation score with rbf kernel:{:.4f}'.format(rbf_scores.mean()))


# ### Comments
# 
# 
# I obtain higher average stratified k-fold cross-validation score of 0.9789 with linear kernel but the model accuracy is 0.9832.
# So, stratified cross-validation technique does not help to improve the model performance.

# # **20. Hyperparameter Optimization using GridSearch CV** <a class="anchor" id="20"></a>
# 
# [Table of Contents](#0.1)

# In[61]:


# import GridSearchCV
from sklearn.model_selection import GridSearchCV


# import SVC classifier
from sklearn.svm import SVC


# instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
svc=SVC() 



# declare parameters for hyperparameter tuning
parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]} 
              ]




grid_search = GridSearchCV(estimator = svc,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)


# In[62]:


# examine the best model


# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))


# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))


# In[63]:


# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))


# ### Comments
# 
# 
# - Our original model test accuracy is 0.9832 while GridSearch CV score on test-set is 0.9835.
# 
# 
# - So, GridSearch CV helps to identify the parameters that will improve the performance for this particular model.
# 
# 
# - Here, we should not confuse `best_score_` attribute of `grid_search` with the `score` method on the test-set. 
# 
# 
# - The `score` method on the test-set gives the generalization performance of the model. Using the `score` method, we employ a model trained on the whole training set.
# 
# 
# - The `best_score_` attribute gives the mean cross-validation accuracy, with cross-validation performed on the training set.

# # **21. Results and conclusion** <a class="anchor" id="21"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# 
# 1. There are outliers in our dataset. So, as I increase the value of C to limit fewer outliers, the accuracy increased. This is true with different kinds of kernels.
# 
# 2.	We get maximum accuracy with `rbf` and `linear` kernel with C=100.0 and the accuracy is 0.9832. So, we can conclude that our model is doing a very good job in terms of predicting the class labels. But, this is not true. Here, we have an imbalanced dataset. Accuracy is an inadequate measure for quantifying predictive performance in the imbalanced dataset problem. So, we must explore `confusion matrix` that provide better guidance in selecting models. 
# 
# 3.	ROC AUC of our model is very close to 1. So, we can conclude that our classifier does a good job in classifying the pulsar star.
# 
# 4.	I obtain higher average stratified k-fold cross-validation score of 0.9789 with linear kernel but the model accuracy is 0.9832. So, stratified cross-validation technique does not help to improve the model performance.
# 
# 5.	Our original model test accuracy is 0.9832 while GridSearch CV score on test-set is 0.9835. So, GridSearch CV helps to identify the parameters that will improve the performance for this particular model.
# 

# # **22. References** <a class="anchor" id="22"></a>
# 
# [Table of Contents](#0.1)
# 
# The work done in this project is inspired from following books and websites:-
# 
#   1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
# 
#   2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
# 
#   3. Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves
# 
#   4. Udemy course – Feature Engineering for Machine Learning by Soledad Galli
# 
#   5. https://en.wikipedia.org/wiki/Support-vector_machine
# 
#   6. https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
# 
#   7. http://dataaspirant.com/2017/01/13/support-vector-machine-algorithm/
# 
#   8. https://www.ritchieng.com/machine-learning-evaluate-classification-model/
# 
#   9. https://en.wikipedia.org/wiki/Kernel_method
# 
#   10. https://en.wikipedia.org/wiki/Polynomial_kernel
# 
#   11. https://en.wikipedia.org/wiki/Radial_basis_function_kernel
# 
#   12. https://data-flair.training/blogs/svm-kernel-functions/

# So, now we will come to the end of this kernel.
# 
# I hope you find this kernel useful and enjoyable.
# 
# Your comments and feedback are most welcome.
# 
# Thank you
# 

# [Go to Top](#0)
