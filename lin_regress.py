# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:16:36 2021

@author: PilzD
"""
#%% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

#%% 
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


#%% Linear regression workflow

rgb = "ColVals_07.04.txt"
lab = "LAB_Vals_140.csv"

X, y = subset_data(rgb, lab)

rgb_train = X[0:100,]
rgb_test = X[101:139]

L_train = y[0:100,0]
a_train = y[0:100,1]
b_train = y[0:100:,2]

L_test = y[101:139, 0]
a_test = y[101:139, 1]
b_test = y[101:139, 2]

reg_L = linear_model.LinearRegression()
reg_a = linear_model.LinearRegression()
reg_b = linear_model.LinearRegression()

reg_L.fit(rgb_train, L_train)
reg_a.fit(rgb_train, a_train)
reg_b.fit(rgb_train, b_train)

L_pred = reg_L.predict(rgb_test)
a_pred = reg_a.predict(rgb_test)
b_pred = reg_b.predict(rgb_test)

err_L_linear = calculate_stats(L_test, L_pred, "L value - linear model")
err_a_linear = calculate_stats(a_test, a_pred, "a value - linear model")
err_b_linear = calculate_stats(b_test, b_pred, "b value - linear model")
#%% Plot L
plot_regression(L_test, L_pred, "L value linear")
#%% Plot a
plot_regression(a_test, a_pred, "a value linear")
#%% Plot b
plot_regression(b_test, b_pred, "b value linear")

