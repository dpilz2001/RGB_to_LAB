# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:44:19 2021

@author: Pilzd
"""
#%% Setup
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

#%% Subset datasets

#Read data from files and return rgb and Lab data as Numpy arrays
def subset_data(rgb_file, LAB_file, col):
    #Load data
    colnames = ["Patch", ":", "R", "G", "B"]
    rgb_data = pd.read_csv(rgb_file, sep=' ', names=colnames)
    rgb_data = rgb_data.drop(labels=["Patch", ":"],axis=1)
    rgb_data = rgb_data.to_numpy()
    rgb_data = rgb_data/255.0
    
    lab_data = pd.read_csv(LAB_file, sep=';', decimal = ',')
    lab_data = lab_data.drop(["Patch"], axis=1)
    lab_data = lab_data.to_numpy()
    
    #Rescale based on L*a*b* value of interest
    #(L: 0 to 100 --> 0 to 1, a/b: -127 to 128 --> -1 to 1)
    if col == 0:
        lab_data = lab_data[:,0]/100
    else:
        lab_data = lab_data[:,col]/128
    
    return rgb_data, lab_data

rgb = "ColVals_07.04.txt"
lab = "LAB_Vals_140.csv"

#Split into x and y datasets
rgb_L, L = subset_data(rgb, lab, 0)
rgb_a, a = subset_data(rgb, lab, 1)
rgb_b, b = subset_data(rgb, lab, 2)

#Split further into train and test datasets
x_train_L, x_test_L, y_train_L, y_test_L = train_test_split(rgb_L, L, test_size=0.3)
x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(rgb_a, a, test_size=0.3)
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(rgb_b, b, test_size=0.3)
        
#%%

np.random.seed(42)

def build_model(output_activation):
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(4,kernel_initializer = 'normal', activation='LeakyReLU',input_dim=3),
      tf.keras.layers.Dense(4,kernel_initializer = 'normal', activation='LeakyReLU'),
      tf.keras.layers.Dense(1,activation=output_activation)
    ])
    
    model.compile(
      optimizer='adam',
      loss = 'mse',
      metrics=['accuracy'])
    
    return model

def plot_loss(hist, label):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title(label + " loss fn")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def plot_data(test_dataset, test_labels, model, title):
    test_predictions = model.predict(test_dataset)
    
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.title(title)
    
    if title.startswith('L'):
        lims = [0, 1]
    else:
        lims = [-1, 1]
        
    _ = plt.plot(lims, lims)

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

#%% Training
#Use sigmoid output for L values
model_L = build_model("sigmoid")
history_L = model_L.fit(x_train_L, y_train_L, epochs=500, verbose = 0, validation_split=0.3, batch_size=16)


#Use tanh for a and b values to allow for negative output
model_a = build_model("tanh")
history_a = model_a.fit(x_train_a, y_train_a, epochs=500, verbose = 0, validation_split=0.3, batch_size=16)


model_b = build_model("tanh")
history_b = model_b.fit(x_train_b, y_train_b, epochs=500, verbose = 0, validation_split=0.3, batch_size=16)

print("done training")

#%% Plot loss
plot_loss(history_L, "model_L")
plot_loss(history_a, "model_a")
plot_loss(history_b, "model_b")

#%%
L_pred = model_L.predict(x_test_L)*100
plot_data(x_test_L, y_test_L, model_L, "L Value")
err_L_nn = calculate_stats(L_pred, y_test_L*100, "L value NN")
#%%
a_pred = model_a.predict(x_test_a)*128
plot_data(x_test_a, y_test_a, model_a, "a Value")
err_a_nn = calculate_stats(a_pred, y_test_a*128, "a value NN")

#%%
b_pred = model_b.predict(x_test_b)*128
plot_data(x_test_b, y_test_b, model_b, "b Value")
err_b_nn = calculate_stats(b_pred, y_test_b*128, "b value NN")

#%% Summary table

table = [["Value\Model", "Neural network", "Linear regression", "Quadratic regression"],
        [ "L*", err_L_nn, err_L_linear, err_L_quadratic],
        ["a*", err_a_nn, err_a_linear, err_a_quadratic],
        ["b*", err_b_nn, err_b_linear, err_b_quadratic]]
print(tabulate(table, tablefmt="github"))