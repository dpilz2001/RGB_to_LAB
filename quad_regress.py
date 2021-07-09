# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:13:00 2021

@author: PilzD
"""
#%% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
#%% Get data
def subset_data(rgb_file, LAB_file):
    
    #Load data
    colnames = ["Patch", ":", "R", "G", "B"]
    rgb_data = pd.read_csv(rgb_file, sep=' ', names=colnames)
    rgb_data = rgb_data.drop(labels=["Patch", ":"],axis=1)
    rgb_data = rgb_data.to_numpy()
    
    lab_data = pd.read_csv(LAB_file, sep=';', decimal = ',')
    lab_data = lab_data.drop(["Patch"], axis=1)
    lab_data = lab_data.to_numpy()
    
    
    return rgb_data, lab_data

def quad_setup(rgb_data):
    r_col = rgb_data[:,0]
    g_col = rgb_data[:,1]
    b_col = rgb_data[:,2]
    
    #Add 6 new columns, corresponding to RG, RB, GB, R^2, G^2, B^2
    
    z = np.zeros((len(rgb_data),6))
    
    rgb_data = np.concatenate((rgb_data, z), 1)
    
    #RG, RB, GB
    rgb_data[:,3] = r_col * g_col
    rgb_data[:,4] = r_col * b_col
    rgb_data[:,5] = g_col * b_col
    
    #R^2, G^2, B^2
    rgb_data[:,6] = r_col**2
    rgb_data[:,7] = g_col**2
    rgb_data[:,8] = b_col**2
    
    return rgb_data

def calculate_stats(expected, predicted, title):
    errors = []
    for i in range(len(expected)):
        err = abs(expected[i] - predicted[i])
        errors.append(err)
    
    mae = mean_absolute_error(expected, predicted)
    print(title)
    print("-------------------------------------")
    print("Mean abs error: " + str(mae))
    print("Median error: " + str(np.median(errors)))
    print("Min error: " + str(min(errors)) + " at index " + str(errors.index(min(errors))))
    print("Max error: " + str(max(errors))+ " at index " + str(errors.index(max(errors))))
    print("-------------------------------------")
    
    return mae 

def plot_regression(X, y, title):
    plt.axes(aspect='equal')
    plt.scatter(X, y)
    
    
    m, b = np.polyfit(X, y, 1)
    plt.plot(X, m*X + b, color='orange')
    
    plt.xlim = [-127, 128]
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.title(title)

#%% Quadratic regression
rgb = "ColVals_07.04.txt"
lab = "LAB_Vals_140.csv"

rgb_data, lab_data = subset_data(rgb, lab)
rgb_data = quad_setup(rgb_data)

#Split datasets
X_train = rgb_data[0:100,]
X_test = rgb_data[101:139,]

L_train = lab_data[0:100,0]
L_test = lab_data[101:139,0]

a_train = lab_data[0:100,1]
a_test = lab_data[101:139,1]

b_train = lab_data[0:100,2]
b_test = lab_data[101:139,2]

#Define models
reg_L = linear_model.LinearRegression()
reg_a = linear_model.LinearRegression()
reg_b = linear_model.LinearRegression()

#Train models
reg_L.fit(X_train, L_train)
reg_a.fit(X_train, a_train)
reg_b.fit(X_train, b_train)

#Predict test dataset
L_pred = reg_L.predict(X_test)
a_pred = reg_a.predict(X_test)
b_pred = reg_b.predict(X_test)

#Calculate error
err_L_quadratic = calculate_stats(L_test, L_pred, "L value - quadratic model")
err_a_quadratic = calculate_stats(a_test, a_pred, "a value - quadratic model")
err_b_quadratic = calculate_stats(b_test, b_pred, "b value - quadratic model")
#%% plot

#%%
plot_regression(L_test, L_pred, "L value quadratic")
#%%
plot_regression(a_test, a_pred, "a value quadratic")
#%%
plot_regression(b_test, b_pred, "b value quadratic")
