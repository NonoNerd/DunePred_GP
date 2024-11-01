# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:11:14 2024

@author: Arnaud Doré / University of Auckland /DHI 
"""

# Load libraries
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

from MDAH.mda import MaxDiss_Simplified_NoThreshold
from MDAH.plots.mda import Plot_MDA_Data

#Select data directory 
import os
os.chdir(r".\Data")

names=['Width','Depth','w_h','d50','H','L','U','D_d50','ws','KD','Lsat_D','Lsat_d50','Lsat_b_D','Th','Thc','DelTh','Ret','ust','P']
namesIn = ['Depth','Width','d50','ust','ws','Lsat_b','Lsat_s'] #  
namesOut = ['H']
namesAll=['Depth','Width','d50','ust','ws','Lsat_b','Lsat_s','H'] 


#Import input prameters
Xp = pd.read_csv('DATA_ML_FILTERED5.csv', header=0,usecols=namesIn)
print(Xp.describe())
#X = np.array(pd.read_csv('DATA_ML_FILTERED5.csv',header=0,usecols=namesIn))

#Import targets
Yp = pd.read_csv('DATA_ML_FILTERED5.csv',header=0,usecols=namesOut)
print(Yp.describe())
#Y = np.array(pd.read_csv('DATA_ML_FILTERED5.csv',header=0,usecols=namesOut))

#Renske et al. data 
Xr = pd.read_csv('RenskeCol.csv',header=0,usecols=namesIn)
Yr = pd.read_csv('RenskeCol.csv',header=0,usecols=namesOut)

# Full dataset
dataset = pd.read_csv('DATA_ML_FILTERED5.csv',header=0,usecols=names)
print(dataset.describe())

#In and out variables
Ds=pd.concat([Xp,Yp],axis=1)
print(Ds.shape)
print(Ds.head(20))
print(Ds.describe(include='all'))
Ds.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey = False)
pyplot.show()

# histograms
Ds.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(Ds)
pyplot.show()

# subset size and scalar index
n_subset = 190  # TRAINING DATA
ix_scalar = [0,1,2,3,4,5,6,7]  

# MDA 
train, test  = MaxDiss_Simplified_NoThreshold(Ds[namesAll].values[:], n_subset, ix_scalar)
train = pd.DataFrame(data=train, columns=namesAll)
test = pd.DataFrame(data=test, columns=namesAll)
Xa=pd.concat((pd.DataFrame(data=train, columns=namesIn), pd.DataFrame(data=test, columns=namesIn)), axis=0)
Ya=pd.concat((pd.DataFrame(data=train, columns=namesOut), pd.DataFrame(data=test, columns=namesOut)), axis=0)

# plot MDA classification
Plot_MDA_Data(test, train)
plt.show()
#fig.savefig('MDA_L_ns190', dpi=600)
...

#Libraries for normalization and standarization
# from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import Normalizer

#Standarize data /fit on training set
scale= StandardScaler()
X_train = scale.fit_transform(pd.DataFrame(data=train, columns=namesIn))
Y_train=(pd.DataFrame(data=train, columns=namesOut))*100
X_test = scale.transform(pd.DataFrame(data=test, columns=namesIn))
Y_test=(pd.DataFrame(data=test, columns=namesOut))*100
Xtn = scale.transform(Xa)
Xr=Xr.reindex(columns=namesIn)
Xrn=scale.transform(Xr)

def _protected_exponent(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)
    
def _power(x,y):
    with np.errstate(over='ignore'):
        # return np.where((np.abs(x) > 0.0001,x**y, 0.) | (np.abs(y) > 0.001, np.divide(x, y), 1.))    
        return np.where(np.abs(x**y) < 100000 ,x**y, 0.)          
        # return x**y

exponential = make_function(function=_protected_exponent, name='exp', arity=1)
power = make_function(function=_power, name='pow', arity=2)

# function_set = ['add', 'sub', 'mul', 'div', 'sqrt','log',
#                 'abs', 'neg', 'inv', exponential]

function_set = ['add', 'sub', 'mul', 'div', 'sqrt',
                'abs', 'neg', 'inv']

#Define Algorithm
est_gp = SymbolicRegressor(population_size=10000,
                           generations=500, init_depth=(4, 20), stopping_criteria=0.0,
                           function_set=function_set, metric='rmse',
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=1, verbose=1,parsimony_coefficient=0.002, random_state=41,n_jobs=1)

# Train algorithm
est_gp.fit(X_train, Y_train)

#Print model
print(est_gp)
print(est_gp._program)
print(est_gp._program.raw_fitness_)
print('R2 train:', est_gp.score(X_train,Y_train))
print('R2 test:', est_gp.score(X_test,Y_test))

# print('Equation:', est_gp._program)
...
#%%
from sympy import *

# see https://www.appsloveworld.com/machine-learning/6/how-to-export-the-output-of-gplearn-as-a-sympy-expression-or-some-other-readable
# or https://stackoverflow.com/questions/48404263/how-to-export-the-output-of-gplearn-as-a-sympy-expression-or-some-other-readable 
converter = {
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'mul': lambda x, y : x*y,
    'div': lambda x, y : x/y,
    'sqrt': lambda x : x**0.5,
    'neg': lambda x : -x,
    'pow2': lambda x : x**2,
    'pow3': lambda x : x**3,
    'inv' : lambda x : 1/x
}

equation = sympify(str(est_gp._program), locals=converter)
print ('Equation:', equation, '\n')


#%%

os.chdir(r"..\Results")

pred_r=est_gp.predict(Xrn)

y_train_predicted = est_gp.predict(X_train)
fig=plt.figure(1)
plt.plot(Y_train/100, y_train_predicted/100, '*k')
plt.text(0.0, 0.23, 'RMSE :'+'{:.3f}'.format(mean_squared_error(y_train_predicted,Y_train,squared=False)/100), style='italic', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6})
plt.text(0.0, 0.20, 'R2 :'+'{:.2f}'.format(est_gp.score(X_train,Y_train)), style='italic', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6})
# plt.xlim(0, 1.6), plt.ylim(0, 1.6)
plt.xlabel ('H Lab (Train Set)'); plt.ylabel ('H model (Train Set)')
# best fit line (1:1)
plt.plot([0, 0.25], [0, 0.25], '--r')
plt.rc('grid', linestyle="dotted", color='black')
plt.grid()
plt.show()

y_test_predicted = est_gp.predict(X_test)
fig=plt.figure(2)
plt.plot(Y_test/100, y_test_predicted/100, '*k')
plt.scatter(np.array(Yr)[0:5,0],pred_r/100,color='g',marker=(5, 1),s=20, alpha=0.8)
plt.text(0.0, 0.23, 'RMSE :'+'{:.3f}'.format(mean_squared_error(Y_test,y_test_predicted,squared=False)/100), style='italic', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6})
plt.text(0.0, 0.20, 'R2 :'+'{:.2f}'.format(est_gp.score(X_test,Y_test)), style='italic', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6})
plt.xlabel ('H Lab (Test Set)'); plt.ylabel ('H model (Test Set)')
plt.rc('grid', linestyle="dotted", color='black')
plt.grid()
# best fit line (1:1)
plt.plot([0, 0.25], [0, 0.25], '--r')
plt.show()

y_predicted = est_gp.predict(Xtn)

fig=plt.figure(3)
plt.plot(Ya, y_predicted/100, '*k')
plt.text(0.0, 0.23, 'RMSE :'+'{:.3f}'.format(mean_squared_error(Ya,y_predicted/100,squared=False)), style='italic', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6})
plt.text(0.0, 0.20, 'R2 :'+'{:.2f}'.format(est_gp.score(Xtn,Ya*100)), style='italic', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6})
# plt.xlim(0, 1.6), plt.ylim(0, 1.6)
plt.xlabel ('H Lab (All data)'); plt.ylabel ('H model (All data)')
plt.rc('grid', linestyle="dotted", color='black')
plt.grid()
# best fit line (1:1)
plt.plot([0, 0.25], [0, 0.25], '--r')
plt.show()
