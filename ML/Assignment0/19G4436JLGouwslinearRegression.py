import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import chart_studio.plotly as py
import matplotlib.pyplot as plt
from matplotlib import style

sns.set_style('darkgrid')

plt.rc('axes', titlesize=18) 
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13) 
plt.rc('legend', fontsize=13) 
plt.rc('font', size=13)

df = pd.read_csv("housingdata.csv", header = None)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

def plotFeatures(col_list, title):
    plt.figure(figsize=(12, 25))
    i = 0
    for col in col_list:
        i += 1
        plt.subplot(7, 2, i)
        plt.plot(df[col], df["MEDV"], marker='.', linestyle='none') #plot this data vs median value
        plt.title(title % (col)) #format the titles
        plt.tight_layout()

colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
plotFeatures(colnames, "Relationship between %s and MEDV")
plt.savefig("twoVariableRelationships.pdf")

plt.subplots(figsize = (10, 10))
sns.set(font_scale = 1.5)
sns.heatmap(df.corr(), square = True, cbar = True, annot = True, annot_kws={'size': 10}) #get correlation coefficients of data in 'confusion table'
plt.savefig("heatmap.pdf")
#plt.show()

colours = sns.color_palette('deep')

def predictPrice(x, theta):
    return np.dot(x, theta)

def calculateCost(x, theta, Y):
    prediction = predictPrice(x, theta)
    return ((prediction - Y)**2).mean()/2

def abline(x, theta, Y, i):
    """Plot al ine from slope and intercept"""
    y_vals = predictPrice(x, theta)
    plt.xlim(0, 20)
    plt.ylim(-10, 60)
    plt.xlabel('No. of Rooms in the house')
    plt.ylabel('Price of house')
#   plt.gca().set_aspect(0.1, adjustable='datalim')
    plt.plot(x[:,1], y_vals, '-', label = f'Iteration: {i + 1}', c = colours[i // 1000])


def gradientDescentLinearRegression(alpha = 0.047, iter = 5000):
    """Calculates the linear regression line with the gradient Descent method"""
    plt.subplots(figsize = (13, 10))
    theta0 = []
    theta1 = []
    costs = []
    predictor = df["RM"] # the predictor number of rooms per dwelling
    x = np.column_stack((np.ones(len(predictor)), predictor))
    Y = df["MEDV"] # the median value of house
    plt.scatter(predictor, Y, label = "Data points", color = colours[-1], s = 5)
    theta = np.zeros(2) #our trial guesses for theta
    for i in range(iter):
        pred = predictPrice(x, theta)
        t0 = theta[0] - alpha * (pred - Y).mean()
        t1 = theta[1] - alpha * ((pred - Y) * x[:,1]).mean()

        theta = np.array([t0, t1])
        J = calculateCost(x, theta, Y) #find cost of this fit
        theta0.append(t0)
        theta1.append(t1)
        costs.append(J)
        if i % 1000 == 0:
            #print(f"Iteration: {i + 1}, Cost = {J}, theta = {theta}")
            abline(x, theta, Y, i) #plot this fit
    print(f'theta0 = {theta0[-1]}\ntheta = {theta[-1]}\nCosts = {costs[-1]}')
    plt.legend()

gradientDescentLinearRegression()

plt.savefig("regression.pdf")
#plt.show()
