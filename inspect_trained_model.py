#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 21:08:59 2025

Script to inspect relation of trained
surrogate model misfit to the model parameter ranges.
Visualized with pairs of model parameters.

@author: Jonathan Bedford, Tectonic Geodesy, Ruhr University Bochum, DE
"""


import tensorflow as tf
from tensorflow.keras import layers, models, Input
from keras import activations
import keras
import numpy as np
import matplotlib.pyplot as plt

from surrogate_utils import *


###############################################################################
#### Reading in data


### Uncomment and use the correct path if you want to continue training 
### an already trained model.
continue_path = './trained_models/v2i_max_entropy_C2DT_2025_09_15_2109.keras'



### Getting the scaled X,Y 
data_path = './synthetic_training_data_generation/random_faults_33247.mat'
X,Y = return_X_scaled_Y_scaled(path=data_path,normalize_Y=True)
X = X[::1]
Y = Y[::1]


### Defining the train test splitting
full_set = np.arange(Y.shape[0])#[0:5_000]
Train_Val_Test = np.array([80,10,10])
pp1 = int(((Train_Val_Test[0]/np.sum(Train_Val_Test))*full_set.size))
pp2 = int(((np.sum(Train_Val_Test[0:2])/np.sum(Train_Val_Test))*full_set.size))


### Continuing the training of a previously trained model or starting from scratch
model_path = './trained_models/v2i_max_entropy_C2DT_2025_09_15_2109.keras'
nn_model = load_nn_model(model_path)
    

## Now looping through and running model in inference mode


chunk_size = 1_000


mse_all = np.array([])
pp = 0

while pp < X[:].shape[0]:
    print(pp,X[:].shape[0])
    Y_pred = nn_model(X[:][pp:pp+chunk_size])
    mse_new = tf.reduce_mean(\
            tf.math.squared_difference(\
            Y_pred, Y[:][pp:pp+chunk_size]), axis=(1,2,3)).numpy()
    mse_all = np.append(mse_all,mse_new)
    pp+=chunk_size
    

## Putting X back in original sale
X_test = RS8_to_ORIG8_on_mat(X[:])


####  Plotting now 

vmaxi = np.nanpercentile(mse_all,99)
my_cmap = 'inferno'
S_control = 0.05
from scipy.interpolate import griddata


classes = ['Centroid\n (x)','Centroid\n (y)','Centroid\n (z)',\
           'strike\n (rad)','dip\n (rad)','rake\n (rad)',\
               'width','length']



fig, axs = plt.subplots(8, 8, figsize=(20, 20))
for i in range(X.shape[1]):
    for j in range(X.shape[1]):
        if j>i:
            ha_any = axs[i, j].scatter(X_test[:, j], X_test[:, i], c=mse_all, \
                              cmap=my_cmap, vmin=0, vmax=vmaxi,s=S_control)
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
        else:
            axs[i,j].remove()
    

class_font_size = 16
LEFT_LABEL_PAD = 10
TOP_LABEL_PAD = 10
fs_ax = 12

for i in range(X.shape[1])[0:-1]:
    axs[i,i+1].yaxis.set_label_position("left")
    axs[i,i+1].set_ylabel(classes[i], \
        fontsize=class_font_size, rotation=0, labelpad=LEFT_LABEL_PAD)
    axs[i,i+1].yaxis.set_label_coords(-0.35, 0.3)
        
        
        
    axs[i,-1].set_yticklabels(axs[i,-1].get_yticks(),fontsize=fs_ax)
    axs[i,-1].yaxis.tick_right()
    
    
for i in range(X.shape[1])[1:]:
    axs[0,i].xaxis.set_label_position("top")
    axs[0,i].set_xlabel(classes[i], \
        fontsize=class_font_size, rotation=0, labelpad=TOP_LABEL_PAD)
    axs[i-1,i].set_xticklabels(axs[i-1,i].get_xticks(),fontsize=fs_ax)


## making a colorbar

CBAR_FONT_SIZE = 16

cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.20])  # [left, bottom, width, height], adjust as needed
    # Suppose im is the mappable object from your imshow/contourf/plot (e.g., im = ax_all[0].imshow(...))
# Use the last im you plotted, or any representative one:
cbar = fig.colorbar(ha_any, cax=cbar_ax, orientation='vertical')
clab = 'Misfit '+'\n'+'(m^2)'
cbar.set_label(clab, fontsize=CBAR_FONT_SIZE,rotation=0,labelpad=15)
cbar.set_ticks([0, vmaxi])
cbar.ax.tick_params(labelsize=fs_ax)
cbar.ax.set_yticklabels([f'{0:.5f}',f'{vmaxi:.5f}'])



### saving
plt.savefig('TestFig3.png',dpi=180)






    