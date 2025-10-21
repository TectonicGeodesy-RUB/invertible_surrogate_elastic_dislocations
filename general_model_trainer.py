#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 22:46:23 2025

Script to train the surrogate model.

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

### Defining the train test splitting
full_set = np.arange(Y.shape[0])#[0:5_000]
Train_Val_Test = np.array([80,10,10])
pp1 = int(((Train_Val_Test[0]/np.sum(Train_Val_Test))*full_set.size))
pp2 = int(((np.sum(Train_Val_Test[0:2])/np.sum(Train_Val_Test))*full_set.size))


### Continuing the training of a previously trained model or starting from scratch
if 'continue_path' in locals() or 'continue_path' in globals():
    model = keras.saving.load_model(continue_path)
    print('Continuing with training of:')
    print(continue_path)
else:
    model = return_v2i_me_conv2dT() ### starting from scratch

###############################################################################
#### Doing the training

BS = 512
LR = 1e-4

lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=LR,
    first_decay_steps=30*int(X[0:pp1].shape[0]/BS), # Measured in batches.
    t_mul=1.0,              # cycle length stays the same
    m_mul=1.0,              # max LR stays the same
    alpha=1e-1,             # minimum LR as a fraction of initial LR
    name="CosineDecayRestarts"
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mse')



cb = keras.callbacks.EarlyStopping(monitor='val_loss',\
    verbose=1,restore_best_weights=True,patience=100)


class MoreDecimals(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        opt = self.model.optimizer
        lr = opt.learning_rate
        if hasattr(lr, '__call__'):
            step = tf.cast(opt.iterations, tf.float32)
            lr = lr(step)
        lr = float(tf.keras.backend.get_value(lr))
        print(
            f"Epoch {str(epoch+1).zfill(5)}: "
            + ", ".join([f"{k}: {v:.7f}" for k, v in logs.items()])
            + f", lr: {lr:.6f}"
        )

    
model.fit(x=X[0:pp1],y=Y[0:pp1],validation_data=\
      (X[pp1:pp2],Y[pp1:pp2]),epochs=10_000,batch_size=BS,\
          callbacks=[cb,MoreDecimals()],verbose=0)
    

##############################################################################
### Saving the model
import datetime as dt
ts = '_'+str(dt.datetime.now().year).zfill(4)+'_'+\
    str(dt.datetime.now().month).zfill(2)+'_'+\
        str(dt.datetime.now().day).zfill(2)+\
            '_'+str(dt.datetime.now().hour).zfill(2)+\
                str(dt.datetime.now().minute).zfill(2)
model_type = 'v2i_max_entropy_C2DT'

save_path = './trained_models/'+model_type+ts+'.keras'
model.save(save_path)
    
    
### Inspecting the results
Y_test_pred, logvars_test, x_test_trig,x_test_sampled = \
    model.predict(X[pp2:][0:1000])

plt.figure();
[plt.plot(np.sort(logvars_test[:,j])) \
     for j in range(logvars_test.shape[1])[0:]]
    
    
[see_Y_set(Y[pp2:][0:Y_test_pred.shape[0]],Y_test_pred,j) for \
  j in np.random.permutation(Y_test_pred.shape[0])[0:10]]































    
