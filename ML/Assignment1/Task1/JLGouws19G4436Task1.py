#Imports for used packages.
import pandas as pd
from sklearn import linear_model as linReg
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import numpy as np
import subprocess

###############################################################################
#plotting preferences
sns.set_style('darkgrid')                                                       
                                                                                  
plt.rc('axes', titlesize=18)                                                    
plt.rc('axes', labelsize=14)                                                    
plt.rc('xtick', labelsize=13)                                                   
plt.rc('ytick', labelsize=13)                                                   
plt.rc('legend', fontsize=13)                                                   
plt.rc('font', size=13) 

colours = sns.color_palette('deep')
###############################################################################

columnNames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
               'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housingdata.csv", names = columnNames)

X = np.array(data['RM']).reshape(-1, 1)
y = np.array(data['MEDV']).reshape(-1, 1)

lreg = linReg.LinearRegression() # initialize model

model = lreg.fit(X, y)#train the model

yPred = lreg.predict(X)#get fitted line

extra = rect((0, 0), 1, 1, fc="w", 
        fill=False, edgecolor='none', linewidth=0)

plt.figure(figsize = (13, 10), tight_layout=True)
ax = sns.scatterplot(data=data, x='RM', y='MEDV', 
        color = colours[0], label = 'Data points', s = 50)
plt.plot(X, yPred, c = colours[1], label = 'Fitted Line', linewidth = 5)
plt.plot([], [], ' ',label = f'$\\theta_0 = %.03f$'%(model.intercept_[0]))
plt.plot([], [], ' ', label = f'$\\theta_1 = %.03f$'%(model.coef_[0,0]))
plt.plot([], [], ' ', label = f'$MSE = %.03f$'%(mse(y, yPred)/2))
ax.set(xlabel = 'No. of Rooms per Dwelling', ylabel = 'Median Value/($1000)')
ax.legend(frameon = True, facecolor='w')
plt.savefig("RMvsMEV.pdf")

#compile latex file
#subprocess.check_call(['lualatex', 'JLGouws19G4436MLAssignment1.tex'])
