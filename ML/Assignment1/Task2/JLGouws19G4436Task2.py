# -------------------------------------------------------------------------------------------------------------------
# Regression is about predicting continuous outcomes rather than discrete class labels.
# Use linear regression to predict Boston housing prices using information such as crime rate,
# property tax rate, distance to employment centers, and highway accessibility.
# -------------------------------------------------------------------------------------------------------------------

# Scikit-learn includes some sample datasets. The Boston dataset is imported:

import time
import pandas as pd
import numpy as np
import cv2

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model

import matplotlib.pyplot as plt
import seaborn as sns

colours = sns.color_palette('deep') 
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})

boston = datasets.load_boston()

models = [('Linear', 'LinearRegression'), ('Ridge', 'Ridge'), 
          ('Lasso', 'Lasso')]
modelDict = {'Model' : [], 'MSE' : [], '$R^2$': [], 'Train Time $\\times 10^{-6}$' : [], 
             'Test Time $\\times 10^{-6}$': []}

modelParams = {'Model' : [], '$\\theta_0$' : [], '$\\theta_{1}$' : [], 
                    '$\\theta_{2}$' : [], '$\\theta_{3}$' : [], '$\\theta_{4}$' : [], 
                    '$\\theta_{5}$' : [], '$\\theta_{6}$' : [], '$\\theta_{7}$' : [], 
                    '$\\theta_{8}$' : [], '$\\theta_{9}$' : [], '$\\theta_{10}$' : [], 
                    '$\\theta_{11}$' : [], '$\\theta_{12}$' : [], 
                    '$\\theta_{13}$' : []
                    }

data = pd.DataFrame(boston.data, columns = boston.feature_names)
data['MEDV'] = boston.target

# Split the data into training (90%) and test sets (10%).
X_train, X_test, y_train, y_test \
    = model_selection.train_test_split(boston.data, 
            boston.target, test_size=0.1, random_state=42)

linreg = getattr(linear_model, 'LinearRegression')()# train model to avoid timing errors
linreg.fit(X_train, y_train)

start = time.time() #instantiate time, so that it doesn't have a systematic
                    #error for first timing

for model in models:
# Initialize the model
    modelDict['Model'] += [model[0]]

    linreg = getattr(linear_model, model[1])()

    # Train the model
    start = time.time()
    linreg.fit(X_train, y_train)
    modelDict['Train Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    y_pred = linreg.predict(X_train)

    modelDict['MSE'] += [metrics.mean_squared_error(y_train, y_pred)]

    start = time.time()
    y_pred = linreg.predict(X_test)

    modelParams['Model'] += [model[0]]
    modelParams['$\\theta_0$'] += [f"%.2f"%linreg.intercept_]
    for i in range(1, 14):
        modelParams[f'$\\theta_{{{i}}}$'] += [f"%.2f"%linreg.coef_[i - 1]]

    modelDict['Test Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    modelDict['$R^2$'] += [metrics.r2_score(y_test,y_pred)]

modelDf = pd.DataFrame(modelDict)
#model
s = modelDf.style
s.format_index(escape="latex", axis=0).hide(axis='index'
        ).to_latex(buf= "tables/linRegModels.tex",  
        column_format = "ccccc", label = "tab:models", 
        caption = "Different Linear Regression Models",
        position_float = "centering", hrules = True, position = "H")

modelPDf = pd.DataFrame(modelParams)
#model
sp = modelPDf.style
sp.format_index(escape="latex", axis=0).hide(axis='index'
        ).to_latex(buf= "tables/linRegModelsParams.tex",  
        column_format = "ccccccccccccccc", label = "tab:modelsParams", 
        caption = "Parameters of Different Linear Regression Models",
        position_float = "centering", hrules = True, position = "H")

#bivariate graphs of different models
###############################################################################

plt.figure(figsize = (13, 10), tight_layout=True)
ax = sns.scatterplot(data=data, x='RM', y='MEDV',    
          color = colours[-1], label = 'Data points', s=50)

rmDom = np.linspace(np.min(data["RM"]), np.max(data["RM"]))

for model in models:
# Initialize the model
    linreg = getattr(linear_model, model[1])()

    # Train the model
    linreg.fit(np.array(data['RM']).reshape(-1, 1), boston.target)

    plt.plot(rmDom, linreg.predict(rmDom.reshape(-1, 1)), 
            c = colours[models.index(model)], label = model[0], linewidth = 5)

ax.set(xlabel = 'No. of Rooms per Dwelling', ylabel = 'Median Value/($1000)')
ax.legend(frameon = True, facecolor='w')    
plt.savefig("RMvsMEVBivariate.pdf")  

################################################################################
#part i

modelParams = {'Model' : [], '$\\theta_0$' : [], '$\\theta_{1}$' : [], 
                    '$\\theta_{2}$' : [], '$\\theta_{3}$' : [], '$\\theta_{4}$' : [], 
                    '$\\theta_{5}$' : [], '$\\theta_{6}$' : [], '$\\theta_{7}$' : [], 
                    '$\\theta_{8}$' : [], '$\\theta_{9}$' : [], '$\\theta_{10}$' : [], 
                    '$\\theta_{11}$' : [], '$\\theta_{12}$' : [], 
                    '$\\theta_{13}$' : []
                    }

dataBExc = data.drop(['B', 'MEDV'], axis = 1)

# Split the data into training (90%) and test sets (10%).
X_train, X_test, y_train, y_test \
    = model_selection.train_test_split(dataBExc.to_numpy(), 
            boston.target, test_size=0.1, random_state=42)

start = time.time() #instantiate time, so that it doesn't have a systematic
                    #error for first timing

#plt.figure(figsize = (13, 10), tight_layout=True)
#ax = sns.scatterplot(data=dataBExc, x='RM', y='MEDV', color = colours[-1], label = 'Data points')  

rmDom = np.linspace(np.min(data["RM"]), np.max(data["RM"]))

modelDict = {'Model' : [], 'MSE' : [], '$R^2$': [], 'Train Time $\\times 10^{-6}$' : [], 
             'Test Time $\\times 10^{-6}$': []}

for model in models:
# Initialize the model
    modelDict['Model'] += [model[0]]

    linreg = getattr(linear_model, model[1])()

    # Train the model
    start = time.time()
    linreg.fit(X_train, y_train)
    modelDict['Train Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    y_pred = linreg.predict(X_train)

    modelDict['MSE'] += [metrics.mean_squared_error(y_train, y_pred)]

    start = time.time()
    y_pred = linreg.predict(X_test)
    modelDict['Test Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    modelDict['$R^2$'] += [metrics.r2_score(y_test,y_pred)]

    modelParams['Model'] += [model[0]]
    modelParams['$\\theta_0$'] += [f"%.2f"%linreg.intercept_]
    for i in range(1, 13):
        #print(f'$\\theta_{{{i if i - 1 < np.where(boston.feature_names == "B")[0][0] else i + 1}}}$')
        modelParams[f'$\\theta_{{{i if i - 1 < np.where(boston.feature_names == "B")[0][0] else i + 1}}}$'] += [f"%.2f"%linreg.coef_[i - 1]]
#    print(model[0],"coeff" , linreg.coef_)
#    print(model[0], "intercept" ,linreg.intercept_)

#    plt.plot(rmDom, rmDom * linreg.coef_[5] + linreg.intercept_, c = colours[models.index(model)], label = model[0])

#ax.set(xlabel = 'No. of Rooms per Dwelling', ylabel = 'Median Value/($1000)')
#ax.legend(frameon = True, facecolor='w')    
#plt.savefig("RMvsMEV.pdf")  


dataBExc = data.drop(['B', 'MEDV'], axis = 1)
    
modelDf = pd.DataFrame(modelDict)
model
s = modelDf.style
s.format_index(escape="latex", axis=0).hide(axis='index'
        ).to_latex(buf= "tables/linRegModelsBExc.tex",  
        column_format = "ccccc", label = "tab:models", 
        caption = "Different Linear Regression Models when B is Excluded",
        position_float = "centering", hrules = True, position = "H")

index = np.where(boston.feature_names == "B")[0][0] + 1
modelParams.pop(f'$\\theta_{{{index}}}$')
modelPDf = pd.DataFrame(modelParams)
#model
sp = modelPDf.style
sp.format_index(escape="latex", axis=0).hide(axis='index'
        ).to_latex(buf= "tables/linRegModelsParamsExcB.tex",  
        column_format = "ccccccccccccccc", label = "tab:modelsParamsExcB", 
        caption = "Parameters of Different Linear Regression Models With B Excluded",
        position_float = "centering", hrules = True, position = "H")

################################################################################################

models = [('Linear', 'LinearRegression'), ('Ridge', 'Ridge'), 
          ('Lasso', 'Lasso')]
modelDict = {'Model' : [], 'MSE' : [], '$R^2$': [], 'Train Time $\\times 10^{-6}$' : [], 
             'Test Time $\\times 10^{-6}$': []}

modelParams = {'Model' : [], '$\\theta_0$' : [], '$\\theta_{1}$' : [], 
                    '$\\theta_{2}$' : [], '$\\theta_{3}$' : [], '$\\theta_{4}$' : [], 
                    '$\\theta_{5}$' : [], '$\\theta_{6}$' : [], '$\\theta_{7}$' : [], 
                    '$\\theta_{8}$' : [], '$\\theta_{9}$' : [], '$\\theta_{10}$' : [], 
                    '$\\theta_{11}$' : [], '$\\theta_{12}$' : [], 
                    '$\\theta_{13}$' : []
                    }

data = pd.DataFrame(boston.data, columns = boston.feature_names)
data['MEDV'] = boston.target

# Split the data into training (90%) and test sets (10%).
X_train, X_test, y_train, y_test \
    = model_selection.train_test_split(boston.data, 
            boston.target, test_size=0.1, random_state=42)

start = time.time() #instantiate time, so that it doesn't have a systematic
                    #error for first timing

models = [('Linear', 'LinearRegression'), ('Ridge', 'Ridge'), 
          ('Lasso', 'Lasso')]
modelDict = {'Model' : [], 'MSE' : [], '$R^2$': [], 'Train Time $\\times 10^{-6}$' : [], 
             'Test Time $\\times 10^{-6}$': []}

modelParams = {'Model' : [], '$\\theta_0$' : [], '$\\theta_{1}$' : [], 
                    '$\\theta_{2}$' : [], '$\\theta_{3}$' : [], '$\\theta_{4}$' : [], 
                    '$\\theta_{5}$' : [], '$\\theta_{6}$' : [], '$\\theta_{7}$' : [], 
                    '$\\theta_{8}$' : [], '$\\theta_{9}$' : [], '$\\theta_{10}$' : [], 
                    '$\\theta_{11}$' : [], '$\\theta_{12}$' : [], 
                    '$\\theta_{13}$' : []
                    }

for model in models:
# Initialize the model
    modelDict['Model'] += [model[0]]

    if(model[0] == "Linear"):
        linreg = getattr(linear_model, model[1])()
    else:
        linreg = getattr(linear_model, model[1])(alpha = 2)

    # Train the model
    start = time.time()
    linreg.fit(X_train, y_train)
    modelDict['Train Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    y_pred = linreg.predict(X_train)

    modelDict['MSE'] += [metrics.mean_squared_error(y_train, y_pred)]

    start = time.time()
    y_pred = linreg.predict(X_test)

    modelParams['Model'] += [model[0]]
    modelParams['$\\theta_0$'] += [f"%.2f"%linreg.intercept_]
    for i in range(1, 14):
        modelParams[f'$\\theta_{{{i}}}$'] += [f"%.2f"%linreg.coef_[i - 1]]

    modelDict['Test Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    modelDict['$R^2$'] += [metrics.r2_score(y_test,y_pred)]

modelDf = pd.DataFrame(modelDict)
#model
s = modelDf.style
s.format_index(escape="latex", axis=0).hide(axis='index'
        ).to_latex(buf= "tables/linRegModelsA2.tex",  
        column_format = "ccccc", label = "tab:modelsA2", 
        caption = "Different Linear Regression Models with $\\alpha = 2$",
        position_float = "centering", hrules = True, position = "H")

################################################################################################
data = pd.DataFrame(boston.data, columns = boston.feature_names)
data['MEDV'] = boston.target

# Split the data into training (90%) and test sets (10%).
X_train, X_test, y_train, y_test \
    = model_selection.train_test_split(boston.data, 
            boston.target, test_size=0.1, random_state=42)

start = time.time() #instantiate time, so that it doesn't have a systematic
                    #error for first timing

models = [('Linear', 'LinearRegression'), ('Ridge', 'Ridge'), 
          ('Lasso', 'Lasso')]
modelDict = {'Model' : [], 'MSE' : [], '$R^2$': [], 'Train Time $\\times 10^{-6}$' : [], 
             'Test Time $\\times 10^{-6}$': []}

modelParams = {'Model' : [], '$\\theta_0$' : [], '$\\theta_{1}$' : [], 
                    '$\\theta_{2}$' : [], '$\\theta_{3}$' : [], '$\\theta_{4}$' : [], 
                    '$\\theta_{5}$' : [], '$\\theta_{6}$' : [], '$\\theta_{7}$' : [], 
                    '$\\theta_{8}$' : [], '$\\theta_{9}$' : [], '$\\theta_{10}$' : [], 
                    '$\\theta_{11}$' : [], '$\\theta_{12}$' : [], 
                    '$\\theta_{13}$' : []
                    }

for model in models:
# Initialize the model
    modelDict['Model'] += [model[0]]

    if(model[0] == "Linear"):
        linreg = getattr(linear_model, model[1])()
    else:
        linreg = getattr(linear_model, model[1])(alpha = 3)

    # Train the model
    start = time.time()
    linreg.fit(X_train, y_train)
    modelDict['Train Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    y_pred = linreg.predict(X_train)

    modelDict['MSE'] += [metrics.mean_squared_error(y_train, y_pred)]

    start = time.time()
    y_pred = linreg.predict(X_test)

    modelParams['Model'] += [model[0]]
    modelParams['$\\theta_0$'] += [f"%.2f"%linreg.intercept_]
    for i in range(1, 14):
        modelParams[f'$\\theta_{{{i}}}$'] += [f"%.2f"%linreg.coef_[i - 1]]

    modelDict['Test Time $\\times 10^{-6}$'] += [(time.time() - start) * 1e6]

    modelDict['$R^2$'] += [metrics.r2_score(y_test,y_pred)]

modelDf = pd.DataFrame(modelDict)
#model
s = modelDf.style
s.format_index(escape="latex", axis=0).hide(axis='index'
        ).to_latex(buf= "tables/linRegModelsA3.tex",  
        column_format = "ccccc", label = "tab:modelsA3", 
        caption = "Different Linear Regression Models with $\\alpha = 3$",
        position_float = "centering", hrules = True, position = "H")
