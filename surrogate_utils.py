#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:32:08 2025

A place for classes and functions that are called by various scripts.

@author: Jonathan Bedford, Tectonic Geodesy, Ruhr University Bochum, DE
"""

import tensorflow as tf
from keras.utils import register_keras_serializable
from tensorflow.keras import layers, models, Input
from keras import activations
import keras
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import time
import itertools
import matplotlib.gridspec as gridspec
import datetime as dt


###############################################################################
###################             CLASSES                   #####################
###############################################################################

@register_keras_serializable()
class Layer_8_to_11(keras.layers.Layer):
    def __init__(self, input_shape=None, **kwargs):
        super(Layer_8_to_11, self).__init__(**kwargs)
        self.input_shape_ = input_shape
    def call(self, x):
        x = tf.concat([
            tf.clip_by_value(x[:,0:3],-1,1),
            tf.sin(x[:,3:4]),
            tf.cos(x[:,3:4]),
            tf.sin(x[:,4:5]),
            tf.cos(x[:,4:5]),
            tf.sin(x[:,5:6]),
            tf.cos(x[:,5:6]),
            tf.clip_by_value(x[:,6:8],-1,1),
            ], axis=1)
        return x
    def compute_output_shape(self):
        return (None,11)
    
    def get_config(self):
        config = super(Layer_8_to_11, self).get_config()
        config.update({"input_shape": self.input_shape_})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class ClipLayer(keras.layers.Layer):
    def __init__(self, input_shape=None, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.input_shape_ = input_shape
    def call(self, x):
        x = tf.concat([
            tf.clip_by_value(x[:,0:11],-1,1),
            ], axis=1)
        return x
    def compute_output_shape(self):
        return (None,11)
    def get_config(self):
        config = super(ClipLayer, self).get_config()
        config.update({"input_shape": self.input_shape_})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class Apply_LV_range(keras.layers.Layer):
    def __init__(self, input_shape=None, **kwargs):
        super(Apply_LV_range, self).__init__(**kwargs)
        self.input_shape_ = input_shape
    def call(self, x_in,logvar_range,var_weight):
        x = tf.reduce_mean(logvar_range)+\
            0.5*(tf.reduce_max(logvar_range)-tf.reduce_min(logvar_range))\
                *x_in
        self.add_loss(tf.reduce_mean(-1*x_in)*var_weight)
        return x
    def compute_output_shape(self):
        return (None,11)
    def get_config(self):
        config = super(Apply_LV_range, self).get_config()
        config.update({"input_shape": self.input_shape_})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
@register_keras_serializable()
class Sampled(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sampled, self).__init__(**kwargs)
    def call(self, x, log_variance):
        # Sample from a normal distribution with mean 0 and standard deviation exp(log_variance/2)
        epsilon = tf.random.normal(tf.shape(x))
        sampled = x + tf.exp(0.5*log_variance) * epsilon
        return sampled
    def get_config(self):
        config = super(Sampled, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class ScaleLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.scale = self.add_weight(name='scale', initializer='ones')
    def call(self, inputs):
        return inputs * self.scale
    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class MatMulInputLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MatMulInputLayer, self).__init__(**kwargs)
        self.w = self.add_weight(name='w', shape=(1, 8), initializer='ones', trainable=True)
    def call(self, inputs):
        w_T = tf.transpose(self.w)  # (8, 1)
        out = tf.matmul(inputs, w_T)  # (batch, 8, 1)
        return tf.squeeze(out, axis=-1)  # (batch, 8)
    def get_config(self):
        config = super(MatMulInputLayer, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class MatMulInputLayerF(keras.layers.Layer):
    def __init__(self,F, **kwargs):
        super(MatMulInputLayerF, self).__init__(**kwargs)
        self.w = self.add_weight(name='w', shape=(F,8,1), initializer='ones', trainable=True)
        self.F = F
    def call(self, inputs):
        out = tf.matmul(inputs, self.w)  # (batch, F, 8, 1)
        return tf.squeeze(out, axis=-1)  # (batch, F,8)
    def get_config(self):
        config = super(MatMulInputLayerF, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@register_keras_serializable()
class ScaleLayerF(keras.layers.Layer):
    def __init__(self, F, **kwargs):
        super(ScaleLayerF, self).__init__(**kwargs)
        self.F = F
        self.scale = self.add_weight(
            name='scale',
            shape=(F, 1, 1, 1),  # broadcastable to (F, 32, 32, 3)
            initializer='ones',
            trainable=True
        )
    def call(self, inputs):
        # inputs: (batch, F, 32, 32, 3)
        scaled = inputs * self.scale  # broadcast scale over spatial and channel dims
        summed = tf.reduce_sum(scaled, axis=1)  # sum over F
        return summed  # shape: (batch, 32, 32, 3)
    def get_config(self):
        config = super(ScaleLayerF, self).get_config()
        config.update({'F': self.F})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class Layer_8_to_11F(keras.layers.Layer):
    def __init__(self,F,w_min,l_min,input_shape=None, **kwargs):
        super(Layer_8_to_11F, self).__init__(**kwargs)
        self.input_shape_ = input_shape
        self.F = F
        self.w_min = w_min
        self.l_min = l_min
    def call(self, x):
        x = tf.concat([
            tf.clip_by_value(x[:,:,0:3],-1,1),
            tf.sin(x[:,:,3:4]),
            tf.cos(x[:,:,3:4]),
            tf.sin(x[:,:,4:5]),
            tf.cos(x[:,:,4:5]),
            tf.sin(x[:,:,5:6]),
            tf.cos(x[:,:,5:6]),
            tf.clip_by_value(x[:,:,6:7],self.w_min,1),
            tf.clip_by_value(x[:,:,7:8],self.l_min,1),
            ], axis=2)
        return x
    def compute_output_shape(self):
        return (None,self.F,11)
    def get_config(self):
        config = super(Layer_8_to_11F, self).get_config()
        config.update({"input_shape": self.input_shape_})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
###############################################################################
###################             FUNCTIONS                 #####################
###############################################################################


def scale_xy(x):
    out = tf.clip_by_value(x,-1,1).numpy()
    return out

def scale_z(x):
    out = tf.clip_by_value(2*(x+0.5),-1,1).numpy()
    return out

def scale_lw(x):
    out = tf.clip_by_value(2*(x-0.5),-1,1).numpy()
    return out

def rev_scale_xy(x):
    out = tf.clip_by_value(x,-1,1).numpy()
    return out

def rev_scale_z(x):
    out = 0.5*tf.clip_by_value(x,-1,1).numpy()-0.5
    return out
    
def rev_scale_lw(x):
    out = 0.5*tf.clip_by_value(x,-1,1).numpy()+0.5
    return out
    


def return_X_scaled_Y_scaled(path,normalize_Y):#### Loading in the dataset
    DS = loadmat(path)#
    keys = DS.keys()
        
    X = np.hstack([DS['x_cen_inputs'],\
                   DS['y_cen_inputs'],\
                       DS['z_cen_inputs'],\
                DS['strike_inputs']*((2*np.pi)/360),\
                
                DS['dip_inputs']*((2*np.pi)/360),\
                
                DS['rake_inputs']*((2*np.pi)/360),\
                
                            DS['fault_width_inputs'],\
                                DS['fault_length_inputs']])#.astype('float32');
    X[:,0] = scale_xy(X[:,0])
    X[:,1] = scale_xy(X[:,1])
    X[:,2] = scale_z(X[:,2])
    X[:,6] = scale_lw(X[:,6])
    X[:,7] = scale_lw(X[:,7])
        
    X = X.astype(np.float32)
    Y = DS['targets'].astype(np.float32)
    
    ### Normalizing Y if normalize_Y is true
    if normalize_Y == True:
        max_disps = np.linalg.norm(Y, axis=3).max(axis=1).max(axis=1)
        Y = np.einsum('ijkl,i->ijkl', Y, 1/max_disps)
    
    ### Trimming away instances of where the scaling has failed
    scaling_failed = np.sum(np.isnan(X),axis=1)+np.sum(np.isinf(X),axis=1) > 1
    X = X[scaling_failed==0]
    Y = Y[scaling_failed==0]
    
    return X,Y

def scale_inputs_for_NN(X_orig):
    ## rescaling X_orig so it can be fed into NN as X
    X = X_orig.copy()
    X[:,0] = scale_xy(X_orig[:,0])
    X[:,1] = scale_xy(X_orig[:,1])
    X[:,2] = scale_z(X_orig[:,2])
    X[:,6] = scale_lw(X_orig[:,6])
    X[:,7] = scale_lw(X_orig[:,7])
    X = X.astype(np.float32)
    return X


def mk_subplots_training():
    fig = plt.figure(figsize=(20,15))
    ax_all = [fig.add_subplot(3,3,j+1) for j in range(9)]
    return fig,ax_all


def see_Y_set(Y_true,Y_pred,ii):
        
    vmini = -1*np.max(np.abs(Y_true[ii]))
    vmaxi = np.max(np.abs(Y_true[ii]))
    
    
    fig,ax_all = mk_subplots_training()
    ## E
    ax_all[0].imshow(Y_true[ii][:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[3].imshow(Y_pred[ii][:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[6].plot(Y_true[ii][:,:,0].ravel(),\
                      Y_pred[ii][:,:,0].ravel(),'bx')
    ax_all[6].plot(Y_true[ii][:,:,0].ravel(),\
                          Y_true[ii][:,:,0].ravel(),'k--')
    
    ## N
    ax_all[1].imshow(Y_true[ii][:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[4].imshow(Y_pred[ii][:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[7].plot(Y_true[ii][:,:,1].ravel(),\
                      Y_pred[ii][:,:,1].ravel(),'bx')
    ax_all[7].plot(Y_true[ii][:,:,1].ravel(),\
                          Y_true[ii][:,:,1].ravel(),'k--')
    ## U
    ax_all[2].imshow(Y_true[ii][:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[5].imshow(Y_pred[ii][:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[8].plot(Y_true[ii][:,:,2].ravel(),\
                      Y_pred[ii][:,:,2].ravel(),'bx')
    ax_all[8].plot(Y_true[ii][:,:,2].ravel(),\
                          Y_true[ii][:,:,2].ravel(),'k--')


def plot_faults(faults,ax,my_color_string):
    for fault in faults:
        strike, dip, fault_length, fault_width, x, y, z = fault
        
        dip_rad = dip#
        str_rad = strike#
        
        ### Making corners of fault that strikes at 0 deg and dips with dip
        A = np.array([0,0,0]);
        B = np.array([fault_width*np.cos(dip_rad),0,\
                      -fault_width*np.sin(dip_rad)]);
        C = np.array([fault_width*np.cos(dip_rad),-fault_length,\
              -fault_width*np.sin(dip_rad)]);
        D = np.array([0,-fault_length,0]);
        rect = np.vstack([A[None,:],B[None,:],\
                          C[None,:],D[None,:],A[None,:]]);# each row is a corner of the rectangular fault
        
        ### Removing the mean
        rect = rect-np.mean(rect[0:-1,:],axis=0)
        
        ### Rotating
        R = np.array([[np.cos(-str_rad), -np.sin(-str_rad)],\
                      [np.sin(-str_rad), np.cos(-str_rad)]]);
        rect[:,0:2] = np.matmul(R,rect[:,0:2].T).T;
        
        ### Relocating the faults
        rect[:,0] = rect[:,0] + x;
        rect[:,1] = rect[:,1] + y;
        rect[:,2] = rect[:,2] + z;
        
        # Plot the fault
        ax.plot3D(rect[:, 0], rect[:, 1], rect[:, 2], '-',\
                  color=my_color_string)
    
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Depth')
    
    lim = 1
    
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-2*lim,0)
    ax.view_init(elev=40,azim=-90)
    
    
def plot_faults_return_handle(faults,ax,my_color_string):
    
    for fault in faults:
        strike, dip, fault_length, fault_width, x, y, z = fault
        
        dip_rad = dip#np.radians(dip)
        str_rad = strike#np.radians(strike)
        
        ### Making corners of fault that strikes at 0 deg and dips with dip
        A = np.array([0,0,0]);
        B = np.array([fault_width*np.cos(dip_rad),0,\
                      -fault_width*np.sin(dip_rad)]);
        C = np.array([fault_width*np.cos(dip_rad),-fault_length,\
              -fault_width*np.sin(dip_rad)]);
        D = np.array([0,-fault_length,0]);
        rect = np.vstack([A[None,:],B[None,:],\
                          C[None,:],D[None,:],A[None,:]]);# each row is a corner of the rectangular fault
        
        ### Removing the mean
        rect = rect-np.mean(rect[0:-1,:],axis=0)
        
        ### Rotating
        R = np.array([[np.cos(-str_rad), -np.sin(-str_rad)],\
                      [np.sin(-str_rad), np.cos(-str_rad)]]);
        rect[:,0:2] = np.matmul(R,rect[:,0:2].T).T;
        
        ### Relocating the faults
        rect[:,0] = rect[:,0] + x;
        rect[:,1] = rect[:,1] + y;
        rect[:,2] = rect[:,2] + z;
        
        
        
        # Plot the fault
        ha_fault = ax.plot3D(rect[:, 0], rect[:, 1], rect[:, 2], '-',\
                  color=my_color_string)

    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Depth')
        
    
    lim = 1
    
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-2*lim,0)
    ax.view_init(elev=40,azim=-70)
    
    
    return ha_fault
    
def mk_subplots():
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20,20))
    gs = gridspec.GridSpec(3, 3)
    
    ax_all = [
        fig.add_subplot(gs[0, 0], projection='rectilinear'),
        fig.add_subplot(gs[0, 1], projection='rectilinear'),
        fig.add_subplot(gs[0, 2], projection='rectilinear'),
        fig.add_subplot(gs[1, 0], projection='rectilinear'),
        fig.add_subplot(gs[1, 1], projection='rectilinear'),
        fig.add_subplot(gs[1, 2], projection='rectilinear'),
        fig.add_subplot(gs[2, :], projection='3d')
    ]
            
    return fig,ax_all

def mk_subplots_target_only():
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20,20))
    gs = gridspec.GridSpec(2, 3)
    
    ax_all = [
        fig.add_subplot(gs[0, 0], projection='rectilinear'),
        fig.add_subplot(gs[0, 1], projection='rectilinear'),
        fig.add_subplot(gs[0, 2], projection='rectilinear'),
        fig.add_subplot(gs[1, :], projection='3d')
    ]
            
    return fig,ax_all

def fault_geom_from_source(source):
    out = np.array([source[3],source[4],source[7],source[6],source[0],\
                    source[1],source[2]])
    return out

def plot_inversion_results(F_true,F_best,Y_true,Y_best):
    fig,ax_all = mk_subplots()
    vmini = -1*np.max(np.abs(Y_true))
    vmaxi = np.max(np.abs(Y_true))
    
    ax_all[0].imshow(Y_true[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[1].imshow(Y_true[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[2].imshow(Y_true[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    ax_all[3].imshow(Y_best[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[4].imshow(Y_best[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[5].imshow(Y_best[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    faults_true = [fault_geom_from_source(F_true.ravel())]
    faults_best = [fault_geom_from_source(F_best.ravel())]
    plot_faults(faults_true,ax_all[6],'b')
    plot_faults(faults_best,ax_all[6],'r')

def plot_target_fields(faults_true,Y_true):
    fig,ax_all = mk_subplots_target_only()
    vmini = -1*np.max(np.abs(Y_true))
    vmaxi = np.max(np.abs(Y_true))
    
    XI = np.linspace(-1,1,Y_true.shape[0])
    YI = np.linspace(-1,1,Y_true.shape[1])
    
    ax_all[0].pcolor(XI,YI,Y_true[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[1].pcolor(XI,YI,Y_true[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[2].pcolor(XI,YI,Y_true[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    plot_faults(faults_true,ax_all[3],'b')
    
def plot_target_fields_return_things(faults_true,Y_true):
    fig,ax_all = mk_subplots_target_only()
    vmini = -1*np.max(np.abs(Y_true))
    vmaxi = np.max(np.abs(Y_true))
    
    XI = np.linspace(-1,1,Y_true.shape[0])
    YI = np.linspace(-1,1,Y_true.shape[1])
    
    ax_all[0].pcolor(XI,YI,Y_true[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[1].pcolor(XI,YI,Y_true[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[2].pcolor(XI,YI,Y_true[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    plot_faults(faults_true,ax_all[3],'b')
    
    return fig,ax_all

def plot_residual_and_faults_temp(faults_true,faults_2,Y_true,Y_pred):
    fig,ax_all = mk_subplots()
    vmini = -1*np.max(np.abs(Y_true))
    vmaxi = np.max(np.abs(Y_true))
    
    XI = np.linspace(-1,1,Y_true.shape[0])
    YI = np.linspace(-1,1,Y_true.shape[1])
    
    ax_all[0].pcolor(XI,YI,Y_true[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[1].pcolor(XI,YI,Y_true[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[2].pcolor(XI,YI,Y_true[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    ax_all[3].pcolor(XI,YI,Y_pred[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[4].pcolor(XI,YI,Y_pred[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[5].pcolor(XI,YI,Y_pred[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    plot_faults(faults_true,ax_all[6],'b')
    plot_faults(faults_2,ax_all[6],'r')
    
    return ax_all
    
    
def plot_initial_3_correlated_faults(F_true,F_1,F_2,F_3):
    
    fig = plt.figure()
    ax0 = fig.add_subplot(1,1,1,projection='3d')
    

    faults_true = [fault_geom_from_source(F_true.ravel())]
    faults_1 = [fault_geom_from_source(F_1.ravel())]
    faults_2 = [fault_geom_from_source(F_2.ravel())]
    faults_3 = [fault_geom_from_source(F_3.ravel())]
    
    plot_faults(faults_true,ax0,'k')
    plot_faults(faults_1,ax0,'b')
    plot_faults(faults_2,ax0,'r')
    plot_faults(faults_3,ax0,'g')


def get_Y_true_M_true_M_init(X,Y,ii,rand_flag):
    """
    returning Y_true, M_true, M_init
    """
    
    if rand_flag == True:
        pp = np.random.permutation(X.shape[0])[0]
    else:
        pp = ii
    
    Y_true = Y[pp]
    M_true = X[pp][:,None]
    
    rand_point = np.zeros(X.shape[1])
    M_init = rand_point[:,None]
    
    return Y_true,M_true,M_init
    

def RS8_to_ORIG8(x):
    """
    Takes input (that should be a sample of scaled X)
    Returns output that should be X with the scaling reversed.

    """
    out = x.copy()
    out[0:2] = rev_scale_xy(out[0:2])
    out[2] = rev_scale_z(out[2])
    out[3:6] = out[3:6]%(2*np.pi)
    out[6:] = rev_scale_lw(out[6:])
    return out


def RS8_to_ORIG8_on_mat(x):
    """
    Takes input (that should be a sample of scaled X)
    Returns output that should be X with the scaling reversed.

    """
    out = x.copy()
    out[:,0:2] = rev_scale_xy(out[:,0:2])
    out[:,2] = rev_scale_z(out[:,2])
    out[:,3:6] = out[:,3:6]%(2*np.pi)
    out[:,6:] = rev_scale_lw(out[:,6:])
    return out


def add_fault(x_in,all_faults,all_faults_scaled):
    x_limited,x_rs = ORIG8_to_RS8(x_in)
    all_faults.append(x_limited)
    all_faults_scaled.append(x_rs)
    
    return all_faults,all_faults_scaled

    
def gen_random_fault():
    return np.array([2*(np.random.rand()-0.5),\
              2*(np.random.rand()-0.5),\
              -1*(np.random.rand()),\
              2*np.pi*(np.random.rand()),\
              2*np.pi*(np.random.rand()),\
              2*np.pi*(np.random.rand()),\
              1*(np.random.rand()),\
              1*(np.random.rand())]).astype('float32')
        

def ORIG8_to_RS8(x):
    """
    Takes original 8 element vector and scales 
    so that it can be used as input to the NN
    
    
    """
    eps = 1e-7
    
    out = x.copy()
    
    
    ### first making sure no boundaries are entered
    for i in [0,1]:
        if out[i]<=-1:
            out[i] = -1+eps
        if out[i]>=1:
            out[i] = 1-eps
    for i in [2]:
        if out[i]<=-1:
            out[i] = -1+eps
        if out[i]>=0:
            out[i] = 0-eps
    for i in [6,7]:
        if out[i]<=0:
            out[i] = 0+eps
        if out[i]>=1:
            out[i] = 1-eps
            
    x_limited = out.copy()
    x_rs = x_limited.copy()
    x_rs[0] = scale_xy(x_limited[0])
    x_rs[1] = scale_xy(x_limited[1])
    x_rs[2] = scale_z(x_limited[2])
    x_rs[6] = scale_lw(x_limited[6])
    x_rs[7] = scale_lw(x_limited[7])
    
    
    return x_limited,x_rs


def ORIG8_to_RS8_on_mat(x):
    """
    Takes original 8 element vector and scales 
    so that it can be used as input to the NN
    
    
    """
    eps = 1e-7
    
    out = x.copy()
    
    
    ### first making sure no boundaries are entered
    for i in [0,1]:
        out[out[:,i]<=-1,i] = -1+eps
        out[out[:,i]>=1,i] = 1-eps
        
    for i in [2]:
        out[out[:,i]<=-1,i] = -1+eps
        out[out[:,i]>=0,i] = 0-eps
        
    for i in [6,7]:
        out[out[:,i]<=0,i] = 0+eps
        out[out[:,i]>=1,i] = 1-eps
        
    x_limited = out.copy()
    x_rs = x_limited.copy()
    x_rs[:,0] = scale_xy(x_limited[:,0])
    x_rs[:,1] = scale_xy(x_limited[:,1])
    x_rs[:,2] = scale_z(x_limited[:,2])
    x_rs[:,6] = scale_lw(x_limited[:,6])
    x_rs[:,7] = scale_lw(x_limited[:,7])
    
    
    return x_limited,x_rs


    
def my_sortrows(a,columns,asc_desc_flag):
    """
    # function aims to copy matlab function "sortrows.m"
    # 'a' must be the table (np.array) that you want to be sorted by columns
    # 'columns' must be a list
    # 'asc_desc_flag' (also a list) is the ascending or descending order for 
    #  each column: (1) ascending, (-1) descending
    """

    if len(columns) == 1:
        columns.append(columns[0])
        asc_desc_flag.append(asc_desc_flag[0])

    columns_arr = np.asarray(columns)
    asc_desc_flag_arr = np.asarray(asc_desc_flag)
    b = columns_arr[::-1]
    b_sgn = asc_desc_flag_arr[::-1]
    b = np.absolute(b)
    scols = np.array([]).reshape(a.shape[0],0)
    scols = ()
    for i in range(b.size):
        scols = scols +(b_sgn[i]*a[:,b[i]],)
    inds = np.lexsort(scols)
    out = a[inds,:]
    return out

def do_multi_fault_inv_sne(G,y,nn_models,options):
    A_prior = tf.matmul(\
                  tf.matmul(\
        tf.linalg.inv(tf.matmul(tf.transpose(G),G)+\
        options['tik_mul']*np.eye(G.shape[1])),\
            tf.transpose(G,[1,0])),
                      y[:,None])
    return A_prior

def do_multi_fault_inv_sne_best(G,y,nn_models,options):
    A_prior = tf.matmul(\
                  tf.matmul(\
        tf.linalg.inv(tf.matmul(tf.transpose(G),G)+\
        options['tik_mul']*np.eye(G.shape[1])),\
            tf.transpose(G,[1,0])),
                      y[:,None])
    return A_prior


def gen_BFs(n_xyz,n_sdr,mag_wl,model,Y_true,model_path):
    xy_spac = np.linspace(-1+1e-2,1-1e-2,n_xyz+2)[1:-1]
    z_spac = np.linspace(-1+1e-2,-1e-2,n_xyz+2)[1:-1]
    sdr_spac = np.linspace(0,2*np.pi,n_sdr+1)[0:-1]
      
    
    x,y,z,s,d,r = np.meshgrid(xy_spac,xy_spac,z_spac,\
                                  sdr_spac,sdr_spac,sdr_spac)
    
    X = np.hstack([x.ravel()[:,None],y.ravel()[:,None],z.ravel()[:,None],\
                     s.ravel()[:,None],d.ravel()[:,None],r.ravel()[:,None],\
                         mag_wl*np.ones([x.size,2])])
    
    X[:,0] = scale_xy(X[:,0])
    X[:,1] = scale_xy(X[:,1])
    X[:,2] = scale_z(X[:,2])
    X[:,6] = scale_lw(X[:,6])
    X[:,7] = scale_lw(X[:,7])
    X = X.astype(np.float32)
    
    ### Trimming away instances of where the scaling has failed
    scaling_failed = np.sum(np.isnan(X),axis=1)+np.sum(np.isinf(X),axis=1) > 0
    X = X[scaling_failed==0]
    
    ### Checking if GF dictionary exists (save time if it exists)
    fname = './BFs/'+model_path.split('.')[0]+'_'+str(n_xyz).zfill(2)+\
        '_'+str(n_sdr).zfill(2)+'.npz'
    if os.path.exists(fname):
        Y_bfs = np.load(fname)['Y_bfs']
    else:
        Y_bfs = model.predict(X)
        Y_bfs = np.array([j.ravel() for j in Y_bfs])
        np.savez(file=fname,Y_bfs=Y_bfs)
    
    ## doing correlation coefficient
    corco = np.dot(Y_bfs, Y_true.ravel())/\
        (np.linalg.norm(Y_bfs, axis=1) * np.linalg.norm(Y_true.ravel()))
    
    ## returning the best initial model as measured by correlation
    ii = np.arange(corco.size)[corco==np.max(corco)][0] # could think about doing "abs" values
        
    return X[ii]


def finer_search_loop(n_sdr,n_wl,model,Y_true):
    
    sdr_spac = np.linspace(0,np.pi,n_sdr+1)[0:-1]
    wl_spac = np.linspace(0,1,n_wl+2)[1:-1]
    
    vec_spacer = np.array([-1,1])
    
    
    x3 = vec_spacer
    y3 = vec_spacer
    z3 = vec_spacer
    
    x,y,z,s,d,r,w,l = np.meshgrid(x3,y3,z3,\
                    sdr_spac,sdr_spac,sdr_spac,\
                        wl_spac,wl_spac)
    
    X = np.hstack([x.ravel()[:,None],y.ravel()[:,None],z.ravel()[:,None],\
                     s.ravel()[:,None],d.ravel()[:,None],r.ravel()[:,None],\
                         w.ravel()[:,None],l.ravel()[:,None]])
    X[:,6] = scale_lw(X[:,6])
    X[:,7] = scale_lw(X[:,7])
    
    X_non_loc = X[:,3:].astype('float32') ## will stay fixed
    X_loc_spacer = X[:,0:3].astype('float32') ### will be varied in loop
    
    ### defining starting position
    x_curr,y_curr,z_curr = 0,0,-0.5
    shrinking_mul = 0.25
    best_cc = 0
    improve_by = 1.01
    
    
    iterating = True
    while iterating == True:
        
        ### updating new search positions in real space
        X_loc = shrinking_mul*X_loc_spacer
        X_loc[:,0] += x_curr
        X_loc[:,1] += y_curr
        X_loc[:,2] += z_curr
        
        ## converting to scaled locations
        X_loc[:,0] = scale_xy(X_loc[:,0])
        X_loc[:,1] = scale_xy(X_loc[:,1])
        X_loc[:,2] = scale_z(X_loc[:,2])
        
        ## Getting the basis functions and unravelling
        Y_bfs = model.predict(np.hstack([X_loc,X_non_loc]))
        Y_bfs = np.array([j.ravel() for j in Y_bfs])
        
        ## calculating cross correlation
        corco = np.dot(Y_bfs, Y_true.ravel())/\
            (np.linalg.norm(Y_bfs, axis=1) * np.linalg.norm(Y_true.ravel()))
        corco = np.abs(corco)
        
        ## Getting best cross correlation
        ii = np.arange(corco.size)[corco==np.max(corco)][0] # could think about doing "abs" values
        
        if corco[ii] > improve_by*best_cc:### add some buffer against tiny improvements
            pp = ii.copy()    
            out_full = np.hstack([X_loc,X_non_loc])
            corco_out = corco.copy()
            x_curr = rev_scale_xy(X_loc[ii,0])
            y_curr = rev_scale_xy(X_loc[ii,1])
            z_curr = rev_scale_z(X_loc[ii,2])
            best_cc = np.max(corco)
            shrinking_mul*=0.5
            print('x,y,z,last_spacer,correl.')
            print(x_curr,y_curr,z_curr,2*shrinking_mul,best_cc)
        else:
            iterating=False
    
    s_top = out_full[pp,3]
    d_top = out_full[pp,4]
    print(s_top)
    print(d_top)
    
    X_best = out_full[ii,:]
    
    ### Get best model that doesn't have same strike
    
    X_str_diff = out_full[out_full[:,3]!=s_top,:]
    corco_str_diff = corco_out[out_full[:,3]!=s_top]
    
    X_dip_diff = out_full[out_full[:,4]!=d_top,:]
    corco_dip_diff = corco_out[out_full[:,4]!=d_top]
    
    ii = np.arange(corco_str_diff.size)[corco_str_diff==np.max(corco_str_diff)][0] # could think about doing "abs" values
    X_str_diff = X_str_diff[ii]
    
    ii = np.arange(corco_dip_diff.size)[corco_dip_diff==np.max(corco_dip_diff)][0] # could think about doing "abs" values
    X_dip_diff = X_dip_diff[ii]
    
    
    return X_best, X_str_diff, X_dip_diff


def gen_fault_params_sdr_wl(n_sdr,n_wl):
    """
    simply outputs a table of strike, dip, rake,
    width, length
    """
    sdr_spac = np.linspace(0,np.pi,n_sdr+1)[0:-1]
    wl_spac =  np.linspace(0,1,n_wl+2)[1:-1]
    
    s,d,r,w,l = np.meshgrid(sdr_spac,sdr_spac,sdr_spac,\
                                      wl_spac,wl_spac)
    
    return np.column_stack([s.ravel(),d.ravel(),r.ravel(),\
                            w.ravel(),l.ravel()])

def gen_fault_params_xyz_sdr_wl(n_xyz,mv_xyz,n_sdr,n_wl):
    """
    simply outputs a table of strike, dip, rake,
    width, length
    """
    sdr_spac = np.linspace(0,np.pi,n_sdr+1)[0:-1]
    wl_spac =  np.linspace(0,1,n_wl+2)[1:-1]
    xyz_spac =  np.linspace(-mv_xyz,mv_xyz,n_xyz)
    
    x,y,z,s,d,r,w,l = np.meshgrid(xyz_spac,xyz_spac,xyz_spac,\
                            sdr_spac,sdr_spac,sdr_spac,\
                                      wl_spac,wl_spac)
    
    return np.column_stack([x.ravel(),y.ravel(),z.ravel(),\
                        s.ravel(),d.ravel(),r.ravel(),\
                            w.ravel(),l.ravel()])
    
def gen_BFs_xyz_sdr_wl(n_xyz,n_sdr,n_wl,model,model_path):
    """
    outputs X_orig, which is the NON-SCALED fault params.
    This is a convention that has been chosen.

    """
    xy_spac = np.linspace(-1+1e-2,1-1e-2,n_xyz+2)[1:-1]
    z_spac = np.linspace(-1+1e-2,-1e-2,n_xyz+2)[1:-1]
    sdr_spac = np.linspace(0,np.pi,n_sdr+1)[0:-1]
    wl_spac =  np.linspace(0,1,n_wl+2)[1:-1]
    
    x,y,z,s,d,r,w,l = np.meshgrid(xy_spac,xy_spac,z_spac,\
                                  sdr_spac,sdr_spac,sdr_spac,\
                                      wl_spac,wl_spac)
    
    X_orig = np.hstack([x.ravel()[:,None],y.ravel()[:,None],z.ravel()[:,None],\
                     s.ravel()[:,None],d.ravel()[:,None],r.ravel()[:,None],\
                         w.ravel()[:,None],l.ravel()[:,None]])
    
    ## rescaling X_orig so it can be fed into NN as X
    X = X_orig.copy()
    X[:,0] = scale_xy(X_orig[:,0])
    X[:,1] = scale_xy(X_orig[:,1])
    X[:,2] = scale_z(X_orig[:,2])
    X[:,6] = scale_lw(X_orig[:,6])
    X[:,7] = scale_lw(X_orig[:,7])
    X = X.astype(np.float32)
    
    ### Trimming away instances of where the scaling has failed
    scaling_failed = np.sum(np.isnan(X),axis=1)+np.sum(np.isinf(X),axis=1) > 0
    X = X[scaling_failed==0]
    X_orig = X_orig[scaling_failed==0]
    
    ### Checking if GF dictionary exists (save time if it exists)
    if os.path.exists('./BFs/') == False:
        os.mkdir('BFs')
    
    fname = './BFs/'+model_path.split('.')[0]+'_'+str(n_xyz).zfill(2)+\
        '_' + str(n_sdr).zfill(2)+\
        '_' + str(n_wl).zfill(2)+'.npz'
    if os.path.exists(fname):
        Y_bfs = np.load(fname)['Y_bfs']
    else:
        Y_bfs = model.predict(X)
        Y_bfs = np.array([j.ravel() for j in Y_bfs])
        np.savez(file=fname,Y_bfs=Y_bfs)
    
    return X_orig,Y_bfs

def gen_BFs_xyz_sdr_wl_chunked(n_xyz,n_sdr,n_wl,model,model_path):
    """
    outputs X_orig, which is the NON-SCALED fault params.
    This is a convention that has been chosen.

    """
    xy_spac = np.linspace(-1+1e-2,1-1e-2,n_xyz+2)[1:-1]
    z_spac = np.linspace(-1+1e-2,-1e-2,n_xyz+2)[1:-1]
    sdr_spac = np.linspace(0,np.pi,n_sdr+1)[0:-1]
    wl_spac =  np.linspace(0,1,n_wl+2)[1:-1]
    
    x,y,z,s,d,r,w,l = np.meshgrid(xy_spac,xy_spac,z_spac,\
                                  sdr_spac,sdr_spac,sdr_spac,\
                                      wl_spac,wl_spac)
    
    X_orig = np.hstack([x.ravel()[:,None],y.ravel()[:,None],z.ravel()[:,None],\
                     s.ravel()[:,None],d.ravel()[:,None],r.ravel()[:,None],\
                         w.ravel()[:,None],l.ravel()[:,None]])
    
    ## rescaling X_orig so it can be fed into NN as X
    X = X_orig.copy()
    X[:,0] = scale_xy(X_orig[:,0])
    X[:,1] = scale_xy(X_orig[:,1])
    X[:,2] = scale_z(X_orig[:,2])
    X[:,6] = scale_lw(X_orig[:,6])
    X[:,7] = scale_lw(X_orig[:,7])
    X = X.astype(np.float32)
    
    ### Trimming away instances of where the scaling has failed
    scaling_failed = np.sum(np.isnan(X),axis=1)+np.sum(np.isinf(X),axis=1) > 0
    X = X[scaling_failed==0]
    X_orig = X_orig[scaling_failed==0]
    
    ### Checking if GF dictionary exists (save time if it exists)
    if os.path.exists('./BFs/') == False:
        os.mkdir('BFs')
    
    fname = './BFs/'+model_path.split('.')[0]+'_'+str(n_xyz).zfill(2)+\
        '_' + str(n_sdr).zfill(2)+\
        '_' + str(n_wl).zfill(2)+'.npz'
    if os.path.exists(fname):
        Y_bfs = np.load(fname)['Y_bfs']
    else:
        Y_bfs_all = []
        pp = 0
        chunk_size = int(50_000)
        while pp < X.shape[0]:
            X_sub = X[pp:pp+chunk_size,:]
            Y_bfs_all.append(model.predict(X_sub))
            pp+=chunk_size
            print(pp,X.shape[0])
        
        
        for i in range(len(Y_bfs_all)):
            if i == 0:
                Y_bfs = np.array([j.ravel() for j in Y_bfs_all[i]])
            else:
                Y_bfs_extra = np.array([j.ravel() for j in Y_bfs_all[i]])
                Y_bfs = np.vstack([Y_bfs,Y_bfs_extra])
        np.savez(file=fname,Y_bfs=Y_bfs)
        
    return X_orig,Y_bfs
        

def get_corco(Y_bfs,Y_true):
    ## doing correlation coefficient
    corco = np.dot(Y_bfs, Y_true.ravel())/\
        (np.linalg.norm(Y_bfs, axis=1) * np.linalg.norm(Y_true.ravel()))
    
    return corco


def gen_BFs_smarter(n_xyz,n_sdr,mag_wl,model,Y_true,model_path):
    xy_spac = np.linspace(-1+1e-2,1-1e-2,n_xyz+2)[1:-1]
    z_spac = np.linspace(-1+1e-2,-1e-2,n_xyz+2)[1:-1]
    sdr_spac = np.linspace(0,np.pi,n_sdr+1)[0:-1]
      
    
    x,y,z,s,d,r = np.meshgrid(xy_spac,xy_spac,z_spac,\
                                  sdr_spac,sdr_spac,sdr_spac)
    
    X = np.hstack([x.ravel()[:,None],y.ravel()[:,None],z.ravel()[:,None],\
                     s.ravel()[:,None],d.ravel()[:,None],r.ravel()[:,None],\
                         mag_wl*np.ones([x.size,2])])
    
    X[:,0] = scale_xy(X[:,0])
    X[:,1] = scale_xy(X[:,1])
    X[:,2] = scale_z(X[:,2])
    X[:,6] = scale_lw(X[:,6])
    X[:,7] = scale_lw(X[:,7])
    X = X.astype(np.float32)
    
    ### Trimming away instances of where the atanh has failed
    scaling_failed = np.sum(np.isnan(X),axis=1)+np.sum(np.isinf(X),axis=1) > 1
    X = X[scaling_failed==0]
    
    ### Checking if GF dictionary exists (save time if it exists)
    fname = './BFs/'+model_path.split('.')[0]+'_'+str(n_xyz).zfill(2)+\
        '_'+str(n_sdr).zfill(2)+'.npz'
    if os.path.exists(fname):
        Y_bfs = np.load(fname)['Y_bfs']
    else:
        Y_bfs = model.predict(X)
        Y_bfs = np.array([j.ravel() for j in Y_bfs])
        np.savez(file=fname,Y_bfs=Y_bfs)
    
    ## doing correlation coefficient
    corco = np.dot(Y_bfs, Y_true.ravel())/\
        (np.linalg.norm(Y_bfs, axis=1) * np.linalg.norm(Y_true.ravel()))
    corco = np.abs(corco)
    
    
    #ii_score = np.hstack([np.arange(corco.size)[:,None],corco[:,None]])
    #ii_score = my_sortrows(ii_score,[1],[-1])
    #topX = 5
    
    ## returning the best initial model as measured by correlation
    ii = np.arange(corco.size)[corco==np.max(corco)][0] # could think about doing "abs" values
    s_top = X[ii,3]
    d_top = X[ii,4]
    
    X_best = X[ii]
    
    ### Get best model that doesn't have same strike
    X_str_diff = X[X[:,3]!=s_top,:]
    corco_str_diff = corco[X[:,3]!=s_top]
    
    X_dip_diff = X[X[:,4]!=d_top,:]
    corco_dip_diff = corco[X[:,4]!=d_top]
    
    ii = np.arange(corco_str_diff.size)[corco_str_diff==np.max(corco_str_diff)][0] # could think about doing "abs" values
    X_str_diff = X_str_diff[ii]
    
    ii = np.arange(corco_dip_diff.size)[corco_dip_diff==np.max(corco_dip_diff)][0] # could think about doing "abs" values
    X_dip_diff = X_dip_diff[ii]
    
    return X_best, X_str_diff, X_dip_diff


def load_nn_model(model_path):
    trained_model = keras.saving.load_model(model_path)
    trained_weights = trained_model.get_weights()
    ### loading in a model with no variational part and with a scaling later appended
    nn_model = return_v2i_me_conv2dT_variational_removed_scale_added()
    new_weights = nn_model.get_weights()
    num_trained_weights = len(trained_weights[20:])
    for i in range(num_trained_weights):
        new_weights[i] = trained_weights[20 + i]
    new_weights[-1] = tf.ones_like(new_weights[-1])  # initialize the weights of the ScaleLayer with 1.0
    nn_model.set_weights(new_weights)
    
    return nn_model


def synthesize_faults_and_Y(N,A_min_max,nn_model):
    all_faults = []
    all_faults_scaled = []
    all_A = [min(A_min_max)+\
             (max(A_min_max)-min(A_min_max))*\
                 np.random.rand() for j in range(N)]
    
    
    for i in range(N):
        all_faults,all_faults_scaled = add_fault(x_in=gen_random_fault(),\
                                all_faults=all_faults,
                                all_faults_scaled=all_faults_scaled)
    
    ## Making a list that is used for plotting faults
    faults_true = [fault_geom_from_source(j.ravel()) for j in all_faults]
    
    Y_target = np.zeros([j for j in nn_model.output.shape[1:]])
    for i in range(N):
        Y_target += all_A[i]*nn_model(all_faults_scaled[i][None,:])[0]

    
    return all_faults, all_faults_scaled, all_A, Y_target



def synthesize_faults_and_Y_smarter(N,A_min_max,nn_model,AR_min,AR_max,side_min):
    all_faults = []
    all_faults_scaled = []
    all_A = [min(A_min_max)+\
             (max(A_min_max)-min(A_min_max))*\
                 np.random.rand() for j in range(N)]
    
    ### makign N faults that don't interstc and are with AR_max and side_min restrictions
    all_faults = make_N_faults(N,AR_min,AR_max,side_min)
    all_faults,all_faults_scaled = ORIG8_to_RS8_on_mat(np.array(all_faults))
    all_faults = list(all_faults)
    all_faults_scaled = list(all_faults_scaled)
    
    #print(all_faults_scaled[0].shape)
    
    ## Making a list that is used for plotting faults
    faults_true = [fault_geom_from_source(j.ravel()) for j in all_faults]
    
    Y_target = np.zeros([j for j in nn_model.output.shape[1:]])
    for i in range(N):
        Y_target += all_A[i]*nn_model(all_faults_scaled[i][None,:])[0]

    
    return all_faults, all_faults_scaled, all_A, Y_target


def find_two_faults(Y_target,X,y_bfs):
    """
    outputs the best two candidate faults [non-scaled] because X_in corresponds
    to X_orig from the function that makes/loads the basis function catalogue.
    X_orig is non-scaled fault params.
    """
    
    corco = get_corco(y_bfs,Y_target)
    X_ii_cc = np.hstack([X,np.arange(corco.size)[:,None],np.abs(corco[:,None])])
    X_ii_cc = my_sortrows(X_ii_cc,[9],[-1])
    
    ### Since X_ii_cc is ordered by cc, getting the best cc for each location
    xyz_positions,indices = np.unique(X_ii_cc[:,0:3],axis=0,return_index=True)
    X_ii_cc = X_ii_cc[np.sort(indices),:] ## trimming so that each centroid has one fault
    
    #_bpp signifies "best per position"
    faults_bpp = [fault_geom_from_source(j.ravel()) for j in X_ii_cc[:,0:8]]
    
    
    ### Getting the selection basis functions ordered in the same order as X_ii_cc
    y_bfs_sel = y_bfs[X_ii_cc[:,-2].astype(int),:]
    
    
    ### Flipping the sign of rakes and basis functions for case where the correlation
    #### coefficient is negative
    to_flip = np.sign(corco[X_ii_cc[:,-2].astype(int)]) == -1
    y_bfs_sel[to_flip,:] = -1*y_bfs_sel[to_flip,:]  ## flipping sign of surface pred
    X_ii_cc[to_flip,5] = (X_ii_cc[to_flip,5]+np.pi) % (2*np.pi) ## rotating rake by pi 
    
    
    ###  Getting now all combinations of the Y_bfs_sel and working out the best two
    ### for predicting the whole displacement field.
    
    tik_mul = 1e-9
    
    y = Y_target.ravel()[:,None]
    
    F_combos_all = np.array(\
                list(itertools.combinations(np.arange(X_ii_cc.shape[0]), 2)))
    G = np.zeros([y.size,2,F_combos_all.shape[0]])
    G[:,0,:] = y_bfs_sel[F_combos_all[:,0],:].T
    G[:,1,:] = y_bfs_sel[F_combos_all[:,1],:].T
    
    
    m_all = tf.matmul(\
                  tf.matmul(\
        tf.linalg.inv(tf.matmul(tf.transpose(G,[2,1,0]),tf.transpose(G,[2,0,1]))+\
        tik_mul*np.eye(G.shape[1])),\
            tf.transpose(G,[2,1,0])),
                      y)
    
    y_pred = tf.matmul(tf.transpose(G,[2,0,1]),m_all)
    resids = (y_pred - y)
    resid_norms = tf.reshape(tf.linalg.norm(resids,axis=1),[-1]).numpy()
    best_combo = np.arange(resid_norms.size)[resid_norms==resid_norms.min()][0]
    two_faults = X_ii_cc[F_combos_all[best_combo],0:8]
    Y_best_combo = y_pred[best_combo].numpy().ravel().reshape(32,32,3)
    A_prior = m_all[best_combo].numpy().ravel()
    
    return two_faults,A_prior


def find_one_fault(Y_target,X,y_bfs):
    """
    copied and pasted find_two_faults and edited .
    some variable names could be improved at later point...
    """
    
    corco = get_corco(y_bfs,Y_target)
    X_ii_cc = np.hstack([X,np.arange(corco.size)[:,None],np.abs(corco[:,None])])
    X_ii_cc = my_sortrows(X_ii_cc,[9],[-1])
    
    ### Since X_ii_cc is ordered by cc, getting the best cc for each location
    xyz_positions,indices = np.unique(X_ii_cc[:,0:3],axis=0,return_index=True)
    X_ii_cc = X_ii_cc[np.sort(indices),:] ## trimming so that each centroid has one fault
    
    #_bpp signifies "best per position"
    faults_bpp = [fault_geom_from_source(j.ravel()) for j in X_ii_cc[:,0:8]]
    
    
    ### Getting the selection basis functions ordered in the same order as X_ii_cc
    y_bfs_sel = y_bfs[X_ii_cc[:,-2].astype(int),:]
    
    
    ### Flipping the sign of rakes and basis functions for case where the correlation
    #### coefficient is negative
    to_flip = np.sign(corco[X_ii_cc[:,-2].astype(int)]) == -1
    y_bfs_sel[to_flip,:] = -1*y_bfs_sel[to_flip,:]  ## flipping sign of surface pred
    X_ii_cc[to_flip,5] = (X_ii_cc[to_flip,5]+np.pi) % (2*np.pi) ## rotating rake by pi 
    
    
    ###  Getting now all combinations of the Y_bfs_sel and working out the best two
    ### for predicting the whole displacement field.
    
    tik_mul = 1e-5
    
    y = Y_target.ravel()[:,None]
    
    F_combos_all = np.array(\
                list(itertools.combinations(np.arange(X_ii_cc.shape[0]), 1)))
    G = np.zeros([y.size,1,F_combos_all.shape[0]])
    G[:,0,:] = y_bfs_sel[F_combos_all[:,0],:].T
    
    m_all = tf.matmul(\
                  tf.matmul(\
        tf.linalg.inv(tf.matmul(tf.transpose(G,[2,1,0]),tf.transpose(G,[2,0,1]))+\
        tik_mul*np.eye(G.shape[1])),\
            tf.transpose(G,[2,1,0])),
                      y)
    
    y_pred = tf.matmul(tf.transpose(G,[2,0,1]),m_all)
    resids = (y_pred - y)
    resid_norms = tf.reshape(tf.linalg.norm(resids,axis=1),[-1]).numpy()
    best_combo = np.arange(resid_norms.size)[resid_norms==resid_norms.min()][0]
    one_fault = X_ii_cc[F_combos_all[best_combo],0:8]
    Y_best_combo = y_pred[best_combo].numpy().ravel().reshape(32,32,3)
    A_prior = m_all[best_combo].numpy().ravel()
    
    return one_fault, A_prior

def optimize_fault_combo_2(input_fault_list,input_amplitudes_list,\
                                fault_combo,\
                                Y,nn_model,model_path,\
                                    side_min_solution,\
                                        my_min_delta,\
                                    keras_verbose_flag):
    
    ## Getting X_scaled from the input_fault_list and fault_combo
    X,X_scaled = ORIG8_to_RS8_on_mat(input_fault_list[fault_combo])#X_scaled = scale_inputs_for_NN(input_fault_list[fault_combo])
    
    
    ### Getting the amplitudes from where to begin the optimizations
    A_prior = input_amplitudes_list[fault_combo]
    
    ## Generating the multi_model with cloned sub-networks depending on how many faults in tested global model
    multi_model = one_model_many_F_for_inversion(\
                        len(input_fault_list[fault_combo]),model_path,
                        side_min_solution)
    
    ## initializing the models with the solutions of normal equations (SNE)
    multi_model.layers[-1].weights[0][:,0,0,0].assign(A_prior.ravel()) ## amplitudes
    f_scaled_init = np.zeros([len(input_fault_list[fault_combo]),8,1]) ### fault params
    for j in range(len(X_scaled)):
        f_scaled_init[j,:,:] = X_scaled[j][:,None]
    multi_model.layers[1].weights[0].assign(f_scaled_init)
        

    ### Putting the optimizer and options here and running optimization
    ### Temporarilyy hard coded here, but later we can make this an option
    
    epoch_size = 100#1#
    n_epochs = 10_000#1#
    BS = 1
    LR = 1e-3
    
    decay_epochs = 10
    
    cosine_annealing = True
    if cosine_annealing == True:
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

        LR_CA = 1e-3
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=LR_CA,
            first_decay_steps=int(decay_epochs*epoch_size), # 1225 steps (batches seen) per cycle
            t_mul=1.0,              # cycle length stays the same
            m_mul=1.0,              # max LR stays the same
            alpha=1e-2,             # minimum LR as a fraction of initial LR
            name="CosineDecayRestarts"
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
    else:
        donothing = 1
        #multi_model.optimizer.learning_rate = options['opt'].learning_rate
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    multi_model.compile(optimizer=optimizer, loss='mse')
    
    cb = keras.callbacks.EarlyStopping(monitor='loss',\
        verbose=0,restore_best_weights=True,patience=55,min_delta=my_min_delta)
    
    if cosine_annealing == True:
        cb_all = [cb,MoreDecimals()]
    else:
        cb_all = [cb]
    
        
    Y_train = np.array([Y for j in range(epoch_size)])
    X = np.array([np.tile(np.eye(8),(len(input_fault_list[fault_combo]),1,1)) \
                  for j in range(epoch_size)])
    X_blur = 0#0.001 ### some attempt at regularization to avoid numerical artifacts, but could be unnecessary
    if np.abs(X_blur)>0:
        X+=X_blur*np.random.randn(X.shape[0],X.shape[1],X.shape[2])
    
    multi_model.fit(x=X,y=Y_train,epochs=n_epochs,batch_size=BS,\
              callbacks=cb_all,verbose=keras_verbose_flag)
    
    ### Now extracting M_best and A_best
    M_best = multi_model.layers[1].get_weights()[0].\
        reshape(len(input_fault_list[fault_combo]),8).astype('float32')
    A_best = multi_model.layers[-1].get_weights()[0].ravel().astype('float32')
    M_best_orig8 = np.array([RS8_to_ORIG8(j) for j in M_best])
    
    
    ### Getting the prediction of the current model
    Y_pred_A = multi_model(np.tile(np.eye(8),(len(input_fault_list[fault_combo]),1,1))[None,:])[0].numpy()
    
    
    Y_pred_B = np.zeros(Y.shape)
    for j in range(len(M_best)):
        Y_pred_B += A_best[j]*(nn_model(M_best[j][None,:])[0])
        
        
    ### making case C, when we make fresh multi_model and use best weights
    ### making a multimodel and populating with weights we have
    #new_multi_model = N_fresh_models(len(input_fault_list[fault_combo]))
    new_multi_model = one_model_many_F_for_inversion(\
                        len(input_fault_list[fault_combo]),model_path,\
                            side_min_solution)
    new_multi_model.layers[-1].weights[0][:,0,0,0].assign(A_best.ravel()) ## amplitudes
    f_scaled_init = np.zeros([len(input_fault_list[fault_combo]),8,1]) ### fault params
    for j in range(len(M_best)):
        f_scaled_init[j,:,:] = M_best[j][:,None]
    new_multi_model.layers[1].weights[0].assign(f_scaled_init)
    Y_pred_C = new_multi_model(np.tile(np.eye(8),(len(input_fault_list[fault_combo]),1,1))[None,:])[0].numpy()
    
    
    ### Getting the misfit of the current mode;  A is with th etrained multi model, B is with single fault trained model
    model_misfit_A = np.linalg.norm(Y-Y_pred_A)
    model_misfit_B = np.linalg.norm(Y-Y_pred_B)
    model_misfit_C = np.linalg.norm(Y-Y_pred_C)
    print('mf A: ',model_misfit_A)
    print('mf B: ',model_misfit_B)
    print('mf C: ',model_misfit_C)
    
    type_ABC = 'C'
    if type_ABC == 'A':
        model_misfit = model_misfit_A
        Y_pred = Y_pred_A
    else:
        model_misfit = model_misfit_C
        Y_pred = Y_pred_C
    
    print('Fault combo misfit after converging: ',model_misfit)
    
    return M_best_orig8, A_best, Y_pred, model_misfit


def nudging(xyz_init,std_loc,max_tries,Y,nn_model_in,\
            xyz_sdr_wl_1,xyz_sdr_wl_2):
    
    verbose_on = False
    
    ## some hardcoded things to be later made options
    samples_per_iteration = 10_000
    inference_mode = 2 ## if 1, using eager mode, 2 with jit thing, 3 using model.predict()
    ### seems mode 2 (with jit decorator thing) is the fastest. about 33% speedup
    corco_needed = 1
    max_moves = 10

    
    @tf.function(jit_compile=True)## the jit thing
    def infer(x):
        return nn_model_in(x)

    def return_best_fault_and_corco_rot_size(fault_in,xyz_sdr_wl_in,Y):
        X = xyz_sdr_wl_in.copy()
        ### rotating and shape changing
        X[:,0:3] = fault_in[0:3]
        X[:,3:6] += fault_in[3:6]### adding the angles ot existing angle
        #print('rot and size')
            
        # if mode == 2: ### moving and shape changing
        #     X[:,0:3] += fault_in[0:3]
        #     X[:,3:6] = fault_in[3:6]
        #     print('mode 2')
            
        X,X_scaled = ORIG8_to_RS8_on_mat(X)
        ss = np.arange(X.shape[0])[0::samples_per_iteration]
        ee = np.append(ss[1:],X.shape[0])
        corco_all = np.zeros(X.shape[0])
        for i in range(ss.size):
            #print(i,time.time()-t1)
            if inference_mode == 1:
                Y_bfs = nn_model_in(X_scaled[ss[i]:ee[i],:])
                Y_bfs = np.array([j.ravel() for j in Y_bfs.numpy()])
            if inference_mode == 2:
                Y_bfs = infer(X_scaled[ss[i]:ee[i],:])
                Y_bfs = np.array([j.ravel() for j in Y_bfs.numpy()])
            if inference_mode == 3:
                BS = 512
                Y_bfs = nn_model_in.predict(X_scaled[ss[i]:ee[i],:],batch_size=BS)
                Y_bfs = np.array([j.ravel() for j in Y_bfs])
            corco_all[ss[i]:ee[i]] = get_corco(Y_bfs,Y)
                
        best_ii = np.arange(corco_all.size)[np.abs(corco_all)==\
                np.abs(corco_all).max()][0]
        return X[best_ii],np.abs(corco_all[best_ii])


    def return_best_fault_and_corco_loc_size(fault_in,xyz_sdr_wl_in,corco_in,Y):
        
        corco_current = np.array(corco_in).copy()
        fault_best = fault_in.copy()
        
        ## initializing search space (not yet the locations)
        X = xyz_sdr_wl_in.copy()
        X[:,3:6] = fault_in[3:6]
        
        converged = False
        counter = 0
        while converged == False:
            counter+=1
            #print(counter)
            X[:,0:3] = xyz_sdr_wl_in[:,0:3] + fault_best[0:3] ## moving location
            
            #print('loc and size')
                
            X,X_scaled = ORIG8_to_RS8_on_mat(X)
            ss = np.arange(X.shape[0])[0::samples_per_iteration]
            ee = np.append(ss[1:],X.shape[0])
            corco_all = np.zeros(X.shape[0])
            for i in range(ss.size):
                #print(i,time.time()-t1)
                if inference_mode == 1:
                    Y_bfs = nn_model_in(X_scaled[ss[i]:ee[i],:])
                    Y_bfs = np.array([j.ravel() for j in Y_bfs.numpy()])
                if inference_mode == 2:
                    Y_bfs = infer(X_scaled[ss[i]:ee[i],:])
                    Y_bfs = np.array([j.ravel() for j in Y_bfs.numpy()])
                if inference_mode == 3:
                    BS = 512
                    Y_bfs = nn_model_in.predict(X_scaled[ss[i]:ee[i],:],batch_size=BS)
                    Y_bfs = np.array([j.ravel() for j in Y_bfs])
                corco_all[ss[i]:ee[i]] = get_corco(Y_bfs,Y)
                    
            best_ii = np.arange(corco_all.size)[np.abs(corco_all)==\
                    np.abs(corco_all).max()][0]
            
            if np.abs(corco_all[best_ii]) > corco_current:
                fault_best = X[best_ii]
                corco_current = np.abs(corco_all[best_ii])
            else:
                converged = True
        
        
        return fault_best,corco_current

    ### preparing loop through tries
    good_enough = False
    counter = 0
    fault_history_current = []
    corco_best = 0
    
    while good_enough == False:
        counter+=1
        xyz = xyz_init+std_loc*np.random.randn(3)
        
        my_fault = np.zeros(8)
        my_fault[0:3] = xyz.copy()
        
        fault_history = []
        corco_history = []
        
        corco_out = 0.0
        
        for i in range(max_moves):
            
            if verbose_on == True:
                print('Iteration #',i)
                print('****************** Rotating and resizing fault ******************')
            my_fault,corco_out1 = return_best_fault_and_corco_rot_size(\
                                        my_fault,xyz_sdr_wl_1,Y)
            #fault_history.append(my_fault.copy()) ### to stop finding "rot" fault as best fault
            #corco_history.append(corco_out1.copy())
            if verbose_on == True:
                print('cor.co.:',corco_out1)
                print('\n')
            
            if verbose_on == True:
                print('*#*#*#*#*#*#*#* Moving and resizing fault *#*#*#*#*#*#*#*#*#')
            my_fault,corco_out = return_best_fault_and_corco_loc_size(\
                                        my_fault,xyz_sdr_wl_2,corco_out,Y)
            fault_history.append(my_fault.copy())
            corco_history.append(corco_out.copy())
            
            if verbose_on == True:
                print('cor.co.:',corco_out)
                print('\n')
            
            if i == 0:
                corco_last = corco_out.copy()
            
            if i>0:
                if corco_out == corco_last:
                    break
                else:
                    corco_last = corco_out.copy()
            else:
                corco_last = corco_out.copy() ## redundant with if ii = 0 statement above, but leave it here for now...
            
            if corco_out >= corco_needed:
                break
        
        if corco_out >= corco_needed:
            good_enough = True
        
        if corco_out > corco_best:
            corco_best = corco_out
            fault_history_current = fault_history
            corco_history_current = corco_history
        
        if counter == max_tries:
            fault_history = fault_history_current
            corco_history = corco_history_current
            good_enough = True
    
    ii_best = np.arange(len(corco_history))\
        [np.array(corco_history)==np.array(corco_history).max()][0]
    
    
    X,X_scaled = ORIG8_to_RS8_on_mat(fault_history[ii_best][None,:])
    Y_bfs = infer(X_scaled)
    Y_bfs = np.array([j.ravel() for j in Y_bfs.numpy()])
    
    ### doing inversion
    tik_mul = 1e-9
    
    y = Y.ravel()[:,None]
    G = Y_bfs.T
    m = np.linalg.matmul(np.linalg.matmul(np.linalg.inv(\
            np.linalg.matmul(G.T,G)+\
                tik_mul*np.eye(G.shape[1])),G.T),y)
    
    return fault_history[ii_best], m.ravel()[0]


def run_nudging_2(input_fault_list_in,input_amplitudes_list_in,\
                                Y_in,nn_model_in,model_path,\
                                    side_min_solution,\
                                    keras_verbose_flag,std_loc,\
                                        max_tries,mfi_nudge,\
                                xyz_sdr_wl_1,xyz_sdr_wl_2):
    
    
    flip_order = True
    
                              
    ## defining lists that will be updated within the funciotn and eventually
    ### output after function converges
    fault_list = input_fault_list_in.copy()
    amplitudes_list = input_amplitudes_list_in.copy()
        
    if flip_order == True:
        fault_list = fault_list[::-1,:]
        amplitudes_list = amplitudes_list[::-1]
    
    ############################################################################
    ### later to go in a for or while loop
    
    
    jj = -1
    
    num_checked = 0
    
    while num_checked < len(fault_list):
    
        jj+=1
        if jj>(len(fault_list)-1):
            jj = 0 ## current fault
        
        ### making a multimodel and populating with weights we have
        multi_model = one_model_many_F_for_inversion(len(fault_list),\
                                    model_path,side_min_solution)
        
        ## initializing the models with the solutions of normal equations (SNE)
        X,X_scaled = ORIG8_to_RS8_on_mat(fault_list)#X_scaled = scale_inputs_for_NN(fault_list)
        
        multi_model.layers[-1].weights[0][:,0,0,0].assign(amplitudes_list.ravel()) ## amplitudes
        f_scaled_init = np.zeros([len(fault_list),8,1]) ### fault params
        for j in range(len(X_scaled)):
            f_scaled_init[j,:,:] = X_scaled[j][:,None]
        multi_model.layers[1].weights[0].assign(f_scaled_init)
        
        Y_all_faults = multi_model(np.tile(np.eye(8),(len(fault_list),1,1))[None,:])[0].numpy()
        initial_mf = np.linalg.norm(Y_in-Y_all_faults)
        print('')
        print('Initial mf: ',initial_mf)
        
        # assiging current afault amplitude as 0 
        multi_model.layers[-1].weights[0][jj,0,0,0].assign(0.0)
        #print('amps:',multi_model.layers[-1].weights[0][:,0,0,0])## for debugging
        
        ### makign target by removing the pred of other faults
        Y_missing_one_fault = multi_model(np.tile(np.eye(8),(len(fault_list),1,1))[None,:])[0].numpy()
        Y_rem = Y_in - Y_missing_one_fault
        #print('Y_in.shape',Y_in.shape)
        #print('Y_missing_one_fault.shape',Y_missing_one_fault.shape)
        #print('Y_rem.shape',Y_rem.shape)
        
        
        ### Doing the dual nudge approach on the 
        xyz_init = fault_list[jj][0:3]
        
        
        
        print('Starting nudging fault,',str(jj+1),' of ',str(len(fault_list)),\
              '. num_checked = ',num_checked)
        out_fault, m = nudging(xyz_init,std_loc,max_tries,Y_rem,nn_model_in,\
                    xyz_sdr_wl_1,xyz_sdr_wl_2)
        print(np.column_stack([fault_list[jj],out_fault]))
        print(amplitudes_list[jj],m)
        
        ### Now assembling provisional new lists of faults and amplitudes
        ### if giving significantly better fit, keeping, if not adding a counter
        fault_list_provisional = fault_list.copy()
        amplitudes_list_provisional = amplitudes_list.copy()
        fault_list_provisional[jj,:] = out_fault.copy()
        amplitudes_list_provisional[jj] = m.copy()
        
        
        multi_model = one_model_many_F_for_inversion(\
                    len(fault_list_provisional),model_path,side_min_solution)
        X,X_scaled = ORIG8_to_RS8_on_mat(fault_list_provisional)#X_scaled = scale_inputs_for_NN(fault_list_provisional)
        multi_model.layers[-1].weights[0][:,0,0,0].assign(amplitudes_list_provisional.ravel()) ## amplitudes
        f_scaled_init = np.zeros([len(fault_list_provisional),8,1]) ### fault params
        for j in range(len(X_scaled)):
            f_scaled_init[j,:,:] = X_scaled[j][:,None]
        multi_model.layers[1].weights[0].assign(f_scaled_init)
        
        Y_all_provisional = multi_model(np.tile(np.eye(8),(len(fault_list_provisional),1,1))[None,:])[0].numpy()
        provisional_mf = np.linalg.norm(Y_in-Y_all_provisional)
        print('Initial mf: ',initial_mf)
        print('Provisional mf: ',provisional_mf)
        
        mfi = 1-(provisional_mf/initial_mf)
                
        if mfi>mfi_nudge:
            num_checked = 0
            fault_list = fault_list_provisional.copy()
            amplitudes_list = amplitudes_list_provisional.copy()
        else:
            num_checked += 1
        print('num_checked :',num_checked)
    
    ######### After no more nudging to be done, outputting
    ### making the output things
    if flip_order == True: ## flipping things back
        fault_list = fault_list[::-1,:]
        amplitudes_list = amplitudes_list[::-1]
    
    multi_model = one_model_many_F_for_inversion(\
                        len(fault_list),model_path,side_min_solution)
    X,X_scaled = ORIG8_to_RS8_on_mat(fault_list)#X_scaled = scale_inputs_for_NN(fault_list)
    multi_model.layers[-1].weights[0][:,0,0,0].assign(amplitudes_list.ravel()) ## amplitudes
    f_scaled_init = np.zeros([len(fault_list),8,1]) ### fault params
    for j in range(len(X_scaled)):
        f_scaled_init[j,:,:] = X_scaled[j][:,None]
    multi_model.layers[1].weights[0].assign(f_scaled_init)
    
    Y_pred = multi_model(np.tile(np.eye(8),(len(fault_list),1,1))[None,:])[0].numpy()
    model_misfit = np.linalg.norm(Y_in-Y_all_faults)
    
    return fault_list,amplitudes_list,Y_pred, model_misfit




    
    
def get_fault_combos(testing_fault_list,max_remove):
    """
    getting combinations of faults with certain numbers of faults removed.
    """
    all_combs = []
    
    n_remove = 2+max_remove ## because we always add 2 faults if we remove 3 faults, we need to actually remove 5 because of the two new candidate faults
    
    ## starting from 1 if we only want to add 1 fault (removing 0 would be adding two faults)
    #rem = np.arange(0,n_remove+1)## + 1, because of python slice indexing..
    #rem = np.arange(1,n_remove+1)## + 1, because of python slice indexing..## MAX ADDING 1 fault
    rem = np.arange(1,n_remove)## + 1, because of python slice indexing..## MAX ADDING 1 and AT LEAST 1
    
    for i in range(rem.size):
        
        if testing_fault_list.shape[0] - rem[i] >= 0:
        
            if rem[i] != 2:
                temp_combs = \
                    np.array(\
                    list(itertools.combinations(\
                        np.arange(testing_fault_list.shape[0]),\
                            testing_fault_list.shape[0]-rem[i])))
            else:
                temp_combs = \
                    np.array(\
                    list(itertools.combinations(\
                        np.arange(testing_fault_list.shape[0]),\
                            testing_fault_list.shape[0]-rem[i])))[1:]
        #print(temp_combs.shape)
        
        [all_combs.append(j) for j in temp_combs]
    
    ## removing combs with nothing
    out_combs = []
    for i in range(len(all_combs)):
        if len(all_combs[i])>0:
            out_combs.append(all_combs[i])
            
    return out_combs    
    

def add_noise_to_field(field,std_mul=1):
    """
    Adds noise to the displacement field according to a standard deviation
    of all pixels of all channels.
    """
    return field+std_mul*np.std(field)*\
        np.random.randn(field.shape[0],field.shape[1],field.shape[2])


def see_two_fields_temp(Y1,Y2):
    fig = plt.figure(figsize=(20,20))
    gs = gridspec.GridSpec(3, 3)
    
    ax_all = [
        fig.add_subplot(gs[0, 0], projection='rectilinear'),
        fig.add_subplot(gs[0, 1], projection='rectilinear'),
        fig.add_subplot(gs[0, 2], projection='rectilinear'),
        fig.add_subplot(gs[1, 0], projection='rectilinear'),
        fig.add_subplot(gs[1, 1], projection='rectilinear'),
        fig.add_subplot(gs[1, 2], projection='rectilinear'),
        fig.add_subplot(gs[2, :], projection='3d')
    ]
    
    vmini = -1*np.max(np.abs(Y1))
    vmaxi = np.max(np.abs(Y1))
    
    XI = np.linspace(-1,1,Y1.shape[0])
    YI = np.linspace(-1,1,Y1.shape[1])
    
    ax_all[0].pcolor(XI,YI,Y1[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[1].pcolor(XI,YI,Y1[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[2].pcolor(XI,YI,Y1[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    ax_all[3].pcolor(XI,YI,Y2[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[4].pcolor(XI,YI,Y2[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ax_all[5].pcolor(XI,YI,Y2[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    return ax_all


def save_fault_info(all_faults,all_A):
    ts = str(dt.datetime.now().year).zfill(4)+'_'+\
        str(dt.datetime.now().month).zfill(2)+'_'+\
            str(dt.datetime.now().day).zfill(2)+\
                '_'+str(dt.datetime.now().hour).zfill(2)+\
                    str(dt.datetime.now().minute).zfill(2)
    out_path = './fault_param_text_files/fault_info_'+ts+'.txt'
    f = open(out_path,'w')
    print(np.array(all_faults).T,file=f)
    print('\n',file=f)
    print(all_A,file=f)
    f.close()


def one_model_many_F_for_inversion(F,trained_model_path,side_min_solution):# returning vec2image with max entropy
    
    ## Hyperparameters for log variance bounding and loss weighting
    var_weight = tf.constant(1e-3)
    std_range = tf.cast(tf.constant([0.001,0.05]),'float32') ### For bounding the logvar space
    logvar_range = 2*tf.math.log(std_range) ### do the math, this is correct
        
    # Model construction
    inputs = keras.Input(shape=(F,8, 8))  # Now expects (F,8,8) input
        
    x = MatMulInputLayerF(F=F)(inputs)      # (batch, F, 8)
    
    ## Branching off toward a log_variance vector of same length
    wl_min = scale_lw(side_min_solution)### clips solutions with sides lower than this magnitude
    x_trig = Layer_8_to_11F(F=F,w_min=wl_min,l_min=wl_min)(x)
    #outputs = Layer_8_to_11F(F=F)(x)
    
    ### Now running through a vec2image type model
    
    x = keras.layers.Dense(128)(x_trig)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Dense(256)(x)  # Further expansion
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Dense(8*8*64)(x)  # Prepare for reshaping into a smaller image
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Reshape((F, 8, 8, 64))(x)  # Reshape into a smaller 3D tensor
    
    x = keras.layers.TimeDistributed(\
                keras.layers.Conv2D(64, kernel_size=3, padding='same'))(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Replace UpSampling2D + Conv2D with Conv2DTranspose
    x = keras.layers.TimeDistributed(\
            keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'))(x)  # 8x8 -> 16x16
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.TimeDistributed(\
              keras.layers.Conv2DTranspose(16, kernel_size=7, strides=2, padding='same'))(x)  # 16x16 -> 32x32
    x = keras.layers.LeakyReLU()(x)
    
    x = keras.layers.TimeDistributed(\
            keras.layers.Conv2D(3, kernel_size=9, padding='same', activation='linear'))(x)  # Final convolution with large kernel and linear activation
        
    outputs = ScaleLayerF(F=F)(x)
        
    model = keras.Model(inputs=inputs, outputs=outputs)
        
    print_summary = False
    if print_summary == True:
        model.summary()
    
    # Compile the model with a custom loss function
    model.compile(optimizer='adam', loss='mse')
    
    
    ### Now freezing some model weights and setting other intial weights
    trained_model = keras.saving.load_model(trained_model_path)
    trained_weights = trained_model.get_weights()

    new_weights = model.get_weights()
    num_trained_weights = len(trained_weights[20:])


    for i in range(num_trained_weights):
        new_weights[i+1] = trained_weights[20 + i]



    new_weights[-1] = tf.ones_like(new_weights[-1])  # initialize the weights of the ScaleLayer with 1.0
    model.set_weights(new_weights)
    new_weights[0] = tf.ones_like(new_weights[0])  # initialize the weights of the first layer with 1.0
    model.set_weights(new_weights)

    ## Making all layers not trainable
    for layer in model.layers:  # exclude the last layer
        layer.trainable = False

    ## Making two of the layers trainable
    model.layers[1].trainable=True ## making layer that represents the scaled 8 fault params trainable
    model.layers[-1].trainable=True ## making the output scaling layer trainable

    see_trainable_layers = False ### for debugging (ideally put the whole "base" nn_model into its own specific function for loading in.)
    if see_trainable_layers == True:
        for j in range(len(model.layers)):
            print(model.layers[j],\
                  'Trainable? ',\
                  model.layers[j].trainable)
    
    return model


def N_fresh_models(N):
    """
    Creates N models each taking an identity matrix input 
    and with a scaled output layer
    """
    ###########################################################################
    ###########################################################################
    ### Loading in specific architecture (add a separate function later)
    trained_model_path = 'v2i_max_entropy_C2DT_2025_09_15_2109.keras'
    nn_model = return_v2i_me_conv2dT_variational_removed_scale_added_identity_input_added()
    trained_model = keras.saving.load_model(trained_model_path)
    trained_weights = trained_model.get_weights()

    ### loading in a model with no variational part and with a scaling later appended
    new_weights = nn_model.get_weights()
    num_trained_weights = len(trained_weights[20:])

    for i in range(num_trained_weights):
        new_weights[i+1] = trained_weights[20 + i]

    new_weights[-1] = tf.ones_like(new_weights[-1])  # initialize the weights of the ScaleLayer with 1.0
    nn_model.set_weights(new_weights)
    new_weights[0] = tf.ones_like(new_weights[0])  # initialize the weights of the first layer with 1.0
    nn_model.set_weights(new_weights)

    ## Making all layers not trainable
    for layer in nn_model.layers:  # exclude the last layer
        layer.trainable = False

    ## Making two of the layers trainable
    nn_model.layers[1].trainable=True ## making layer that represents the scaled 8 fault params trainable
    nn_model.layers[-1].trainable=True ## making the output scaling layer trainable
    
    see_trainable_layers = False ### for debugging (ideally put the whole "base" nn_model into its own specific function for loading in.)
    if see_trainable_layers == True:
        for j in range(len(nn_model.layers)):
            print(nn_model.layers[j],\
                  'Trainable? ',\
                  nn_model.layers[j].trainable)
    
    ###########################################################################
    ###########################################################################
    ####### Making many models
    N_sub = N  # Number of sub-networks
    input_shape = nn_model.input_shape[1:]
    inputs = keras.Input(shape=input_shape)

    clones = []
    for i in range(N_sub):
        clone = keras.models.clone_model(nn_model)
        
        ### Now making the weights the same as in the original nn_model
        old_weights = nn_model.get_weights()
        new_weights = clone.get_weights()
        for j in range(len(new_weights)):
            new_weights[j] = old_weights[j].copy()
        clone.set_weights(new_weights)
        
        clone.name = f"subnet_{i}"  # Set unique name for each clone
        clones.append(clone)
    
    
    outputs = [clone(inputs) for clone in clones]
    summed_output = keras.layers.Add()(outputs)
    multi_model = keras.Model(inputs=inputs, outputs=summed_output)
    
    if see_trainable_layers == True:
        for j in range(len(multi_model.layers)):
            print(multi_model.layers[j],\
                  multi_model.layers[j].trainable)
    multi_model.layers[0].trainable = False ### the general input (which flows into all clones)
    for j in range(N_sub):
        multi_model.layers[j+1].layers[0].trainable = False ### the inputs to each submodel flows from general input
    multi_model.layers[-1].trainable = False ## the addition layer at the end (adding all sub models)
    
    return multi_model


### function to make N new faults.  All subsequent rectangles cannot
### intersect previous ones

def make_N_faults(N,AR_min,AR_max,side_min):
    
    ## temp values to build up function
    #N = 3
    #AR_max = 5
    #side_min = 0.2
    
    
    counter = 0
    fault_list = []
    
    while counter < N:
    
        ### making w and l for new fault
        dims_good = False
        while dims_good == False:
            w,l = np.random.rand(),np.random.rand()
            AR = (np.max([w,l]))/(np.min([w,l]))
            if (AR > AR_min)*(AR < AR_max)*(np.min([w,l])>side_min):
                dims_good = True
        
        ### Making random xyzsdr
        xyzsdrwl = np.array([2*(np.random.rand()-0.5),\
                  2*(np.random.rand()-0.5),\
                  -1*(np.random.rand()),\
                  2*np.pi*(np.random.rand()),\
                  2*np.pi*(np.random.rand()),\
                  2*np.pi*(np.random.rand()),\
                      w,l]).astype('float32')
    
        ## testing against all existing faults to check for intersection
        rect1 = rect_from_fault(xyzsdrwl)
        
        intersection = False
        for i in range(len(fault_list)):
            rect2 = rect_from_fault(fault_list[i])
            intersection = intersect(rect1, rect2)
            if intersection == True:
                break
        
        if intersection == False:
            counter+=1
            fault_list.append(xyzsdrwl)
    
    return fault_list
            
                
            
            

########## functions below for checkign intersection of rectanges from fault params

def rect_from_fault(xyzsdrwl):
    strike, dip, fault_length, fault_width, x, y, z = \
        xyzsdrwl[3],xyzsdrwl[4],xyzsdrwl[7],xyzsdrwl[6],\
            xyzsdrwl[0],xyzsdrwl[1],xyzsdrwl[2]
    
    dip_rad = dip#np.radians(dip)
    str_rad = strike#np.radians(strike)
    
    ### Making corners of fault that strikes at 0 deg and dips with dip
    A = np.array([0,0,0]);
    B = np.array([fault_width*np.cos(dip_rad),0,\
                  -fault_width*np.sin(dip_rad)]);
    C = np.array([fault_width*np.cos(dip_rad),-fault_length,\
          -fault_width*np.sin(dip_rad)]);
    D = np.array([0,-fault_length,0]);
    rect = np.vstack([A[None,:],B[None,:],\
                      C[None,:],D[None,:],A[None,:]]);# each row is a corner of the rectangular fault
    
    ### Removing the mean
    rect = rect-np.mean(rect[0:-1,:],axis=0)
    
    ### Rotating
    R = np.array([[np.cos(-str_rad), -np.sin(-str_rad)],\
                  [np.sin(-str_rad), np.cos(-str_rad)]]);
    rect[:,0:2] = np.matmul(R,rect[:,0:2].T).T;
    
    ### Relocating the faults
    rect[:,0] = rect[:,0] + x;
    rect[:,1] = rect[:,1] + y;
    rect[:,2] = rect[:,2] + z;
    
    rect = rect[0:-1,:]
    return rect


def get_normals(rect):
    # Compute normal vectors of each face
    edges = rect[1:] - rect[:-1]
    normals = np.cross(edges, np.roll(edges, 1, axis=0))
    return normals

def get_projection(rect, normal):
    # Project rectangle onto normal vector
    proj = np.dot(rect, normal / np.linalg.norm(normal))
    return np.min(proj), np.max(proj)

def intersect(rect1, rect2):
    # Compute edges of each rectangle
    edges1 = rect1[1:] - rect1[:-1]
    edges1 = np.vstack((edges1, rect1[0] - rect1[-1]))
    edges2 = rect2[1:] - rect2[:-1]
    edges2 = np.vstack((edges2, rect2[0] - rect2[-1]))

    # Compute normal vectors of each rectangle's faces
    normals1 = get_normals(rect1)
    normals2 = get_normals(rect2)

    # Check if rectangles can be separated by any normal vector or cross product of edges
    for normal in np.vstack((normals1, normals2, np.cross(edges1[:, None], edges2[None, :]).reshape(-1, 3))):
        min1, max1 = get_projection(rect1, normal)
        min2, max2 = get_projection(rect2, normal)
        if max1 < min2 or max2 < min1:
            return False

    return True




### running the perturbation
def run_perturb(input_fault_list_in,input_amplitudes_list_in,\
                                Y_in,nn_model,options,min_delta_perturb,\
                                    keras_verbose_flag):
    
    
                              
    ## defining lists that will be updated within the funciotn and eventually
    ### output after function converges
    fault_list = input_fault_list_in.copy()
    amplitudes_list = input_amplitudes_list_in.copy()
        
    
    ############################################################################
    ### later to go in a for or while loop
    n_perturb  = 5
    
    fault_params_all = []
    amplitudes_all = []
    Y_pred_all = []
    misfits_all = []
    
        
    
    for i in range(n_perturb):
    
        # randomly pertubing sdr and setting w,l to 0.5,0.5
        fault_list_perturbed = fault_list.copy()
        fault_list_perturbed[:,-2:] = 0.5
        fault_list_perturbed[:,3:6] = 2*np.pi*np.random.rand(\
                    fault_list_perturbed[:,3:6].shape[0],\
                        fault_list_perturbed[:,3:6].shape[1])
        
        ## setting amplitudes to 0
        amplitudes_list_perturbed = amplitudes_list.copy()
        amplitudes_list_perturbed[:] = 1.0
        
        fault_params,amplitudes,Y_pred, model_misfit = optimize_fault_combo(\
                fault_list_perturbed,amplitudes_list_perturbed,\
                    np.arange(len(fault_list_perturbed)),\
                        Y_in.numpy(),nn_model,options,\
                            min_delta_perturb,keras_verbose_flag)
        
    
        fault_params_all.append(fault_params)
        amplitudes_all.append(amplitudes)
        Y_pred_all.append(Y_pred)
        misfits_all.append(model_misfit)
    
    
    
    return fault_params_all,amplitudes_all,Y_pred_all, misfits_all


def do_many_inversions(G,y):
    tik_mul = 1e-9
    m_all = tf.matmul(\
                  tf.matmul(\
        tf.linalg.inv(tf.matmul(tf.transpose(G,[0,2,1]),G)+\
        tik_mul*np.eye(G.shape[2])),\
            tf.transpose(G,[0,2,1])),
                      y[:,None])
    mse_all = \
        np.mean(((y[:,None] - tf.matmul(G,m_all)).numpy()**2),axis=1).ravel()
    
    return mse_all, m_all.numpy()


def flip_search(best_faults,\
            Y,nn_model):
    
    wl_spac = 1
    G_all_faults = []
    
    #best_faults = global_fault_list_history[-1] ## use judgement here
    
    for i in range(best_faults.shape[0]):
        fault = best_faults[i][None,:]
        wl = np.linspace(0,1,wl_spac+2)[1:-1]
        w,l = np.meshgrid(wl,wl) 
        all_wl_combs = np.column_stack([w.ravel(),l.ravel()])
        
        new_faults_1  = np.tile(fault,(all_wl_combs.shape[0],1))
        new_faults_1[:,-2:] = all_wl_combs
        
        new_faults_2 = new_faults_1.copy()
        new_faults_3 = new_faults_2.copy()
        
        new_faults_2[:,3]+= np.pi/2
        new_faults_2[:,5] += np.pi
        
        new_faults_3[:,4]+= np.pi/2
        new_faults_3[:,5] += np.pi
        
        Y_bfs_1 = nn_model.predict(new_faults_1)
        Y_bfs_1 = np.array([j.ravel() for j in Y_bfs_1])
        
        Y_bfs_2 = nn_model.predict(new_faults_2)
        Y_bfs_2 = np.array([j.ravel() for j in Y_bfs_2])
        
        Y_bfs_3 = nn_model.predict(new_faults_3)
        Y_bfs_3 = np.array([j.ravel() for j in Y_bfs_3])
        
        G = np.zeros([3,Y_bfs_1.shape[0],Y_bfs_1.shape[1]])
        G[0,:,:] = Y_bfs_1.copy()
        G[1,:,:] = Y_bfs_2.copy()
        G[2,:,:] = Y_bfs_3.copy()
        
        G_all_faults.append(G)
    G_all_faults = np.array(G_all_faults)
    
    
    
    ### Indexing for pulling orientations
    orientations = np.arange(G_all_faults.shape[1])[:,None]
        
    for j in range(G_all_faults.shape[0])[1:]:
        new_col = np.array([k*np.ones(orientations.shape[0]) for k in \
                            range(G_all_faults.shape[1])]).ravel()
        
        orientations = np.tile(orientations,(G_all_faults.shape[1],1))
        orientations = np.column_stack([new_col,orientations])
    orientations = orientations.astype(int)
    
    ### Indexing for pulling shapes
    shapes = np.arange(G_all_faults.shape[2])[:,None]
        
    for j in range(G_all_faults.shape[0])[1:]:
        new_col = np.array([k*np.ones(shapes.shape[0]) for k in \
                            range(G_all_faults.shape[2])]).ravel()
        
        shapes = np.tile(shapes,(G_all_faults.shape[2],1))
        shapes = np.column_stack([new_col,shapes])
    shapes = shapes.astype(int)
    
    
    mse_min = np.inf ## to be beaten
    best_orientation_shape = np.array([np.nan,np.nan])
    
    
    ### outer-loop making a G_all_faults_temp from G_all_faults and orientations combo
    for i in range(len(orientations)):
        print('Srike-Dip configuration ',str(i+1).zfill(3),' of ',\
              str(len(orientations)).zfill((3)))
        G_all_faults_temp = np.zeros([G_all_faults.shape[0],\
                                      G_all_faults.shape[2],\
                                      G_all_faults.shape[3]])
        for j in range(G_all_faults_temp.shape[0]):
            G_all_faults_temp[j,:,:] = G_all_faults[j,orientations[i,j],:,:]
    
        ### making a G from G_all_faults_temp and shapes
        G = np.zeros([shapes.shape[0],\
                                      G_all_faults.shape[3],\
                                      G_all_faults.shape[0]])
        for j in range(len(shapes)):
            G[j,:,:] = np.array([G_all_faults_temp[k,shapes[j,k]] \
                            for k in range(G_all_faults.shape[0])]).T
        
        ### now doing the inversion
        mse_all, m_all = do_many_inversions(G,Y.numpy().ravel())
        if mse_all.min() < mse_min:
            ii = np.arange(mse_all.size)[mse_all == mse_all.min()][0]
            best_orientation_shape[0] = i
            best_orientation_shape[1] = ii.copy()
            A_best = m_all[ii].ravel().copy()
            best_orientation_shape = best_orientation_shape.astype(int)
            mse_min = mse_all.min().copy()
            print(best_orientation_shape,mse_min,mse_all.min())
        
    
    ### Making the best faults from the strike-dip flipping exercise
    out_orientations = orientations[best_orientation_shape[0]]
    out_shapes = shapes[best_orientation_shape[1]]
        
    faults_flipped = best_faults.copy()
    for i in range(len(out_orientations)):
        ## flipping strike or dip if necessary
        if out_orientations[i] == 1:
            faults_flipped[i,3] += np.pi/2
            faults_flipped[i,5] += np.pi
        if out_orientations[i] == 2:
            faults_flipped[i,4] += np.pi/2
            faults_flipped[i,5] += np.pi
        
        ## changing size of fault
        faults_flipped[i,-2:] = all_wl_combs[out_shapes[i]]

    return faults_flipped,A_best



###############################################################################
###################             KERAS MODELS              #####################
###############################################################################


def return_v2i_me_conv2dT():  # returning vec2image with max entropy
    import tensorflow as tf
    from tensorflow import keras

    # Hyperparameters for log variance bounding and loss weighting
    var_weight = tf.constant(1e-3)
    std_range = tf.cast(tf.constant([0.001, 0.05]), 'float32')
    logvar_range = 2 * tf.math.log(std_range)

    # Model input
    inputs = keras.Input(shape=(8,))

    # Branching off toward a log_variance vector of same length
    x_trig = Layer_8_to_11()(inputs)

    x_logvar = keras.layers.Dense(11)(x_trig)
    x_logvar = keras.layers.LeakyReLU()(x_logvar)
    x_logvar = keras.layers.BatchNormalization()(x_logvar)

    x_logvar = keras.layers.Dense(11)(x_logvar)
    x_logvar = keras.layers.LeakyReLU()(x_logvar)
    x_logvar = keras.layers.BatchNormalization()(x_logvar)

    x_logvar = keras.layers.Dense(11)(x_logvar)
    x_logvar = keras.layers.LeakyReLU()(x_logvar)
    x_logvar = keras.layers.BatchNormalization()(x_logvar)

    x_logvar = keras.layers.Dense(11, activation='tanh')(x_logvar)
    x_logvar = Apply_LV_range()(x_logvar, logvar_range, var_weight)

    # Sampling
    x_sampled = Sampled()(x_trig, x_logvar)

    # Clipping layer
    x_sampled = ClipLayer()(x_sampled)

    # Now running through a vec2image type model
    x = keras.layers.Dense(128)(x_sampled)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(8 * 8 * 64)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Reshape((8, 8, 64))(x)

    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    # Replace UpSampling2D + Conv2D with Conv2DTranspose
    x = keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)  # 8x8 -> 16x16
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2DTranspose(16, kernel_size=7, strides=2, padding='same')(x)  # 16x16 -> 32x32
    x = keras.layers.LeakyReLU()(x)

    outputs = keras.layers.Conv2D(3, kernel_size=9, padding='same', activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=[outputs, x_logvar, x_trig, x_sampled])
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    return model

def return_v2i_me_conv2dT_variational_removed():# returning vec2image with max entropy
    
    ## Hyperparameters for log variance bounding and loss weighting
    var_weight = tf.constant(1e-3)
    std_range = tf.cast(tf.constant([0.001,0.05]),'float32') ### For bounding the logvar space
    logvar_range = 2*tf.math.log(std_range) ### do the math, this is correct
    
    ### Putting model here
    inputs = keras.Input(shape=(8,))

    ## Branching off toward a log_variance vector of same length
    x_trig = Layer_8_to_11()(inputs)

    
    ### Now running through a vec2image type model

    x = keras.layers.Dense(128)(x_trig)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(256)(x)  # Further expansion
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(8*8*64)(x)  # Prepare for reshaping into a smaller image
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Reshape((8, 8, 64))(x)  # Reshape into a smaller 3D tensor

    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)  # Convolution with small kernel
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    # Replace UpSampling2D + Conv2D with Conv2DTranspose
    x = keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)  # 8x8 -> 16x16
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2DTranspose(16, kernel_size=7, strides=2, padding='same')(x)  # 16x16 -> 32x32
    x = keras.layers.LeakyReLU()(x)

    outputs = keras.layers.Conv2D(3, kernel_size=9, padding='same', activation='linear')(x)  # Final convolution with large kernel and linear activation

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # Compile the model with a custom loss function
    model.compile(optimizer='adam', loss='mse')
    
    return model



def return_v2i_me_conv2dT_variational_removed_scale_added():# returning vec2image with max entropy
    
    ## Hyperparameters for log variance bounding and loss weighting
    var_weight = tf.constant(1e-3)
    std_range = tf.cast(tf.constant([0.001,0.05]),'float32') ### For bounding the logvar space
    logvar_range = 2*tf.math.log(std_range) ### do the math, this is correct
    
    ### Putting model here
    inputs = keras.Input(shape=(8,))

    ## Branching off toward a log_variance vector of same length
    x_trig = Layer_8_to_11()(inputs)

    
    ### Now running through a vec2image type model

    x = keras.layers.Dense(128)(x_trig)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(256)(x)  # Further expansion
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(8*8*64)(x)  # Prepare for reshaping into a smaller image
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Reshape((8, 8, 64))(x)  # Reshape into a smaller 3D tensor

    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)  # Convolution with small kernel
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    # Replace UpSampling2D + Conv2D with Conv2DTranspose
    x = keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)  # 8x8 -> 16x16
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2DTranspose(16, kernel_size=7, strides=2, padding='same')(x)  # 16x16 -> 32x32
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(3, kernel_size=9, padding='same', activation='linear')(x)  # Final convolution with large kernel and linear activation
    
    outputs = ScaleLayer()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # Compile the model with a custom loss function
    model.compile(optimizer='adam', loss='mse')
    
    return model

def return_v2i_me_conv2dT_variational_removed_scale_added_identity_input_added():# returning vec2image with max entropy
    
    ## Hyperparameters for log variance bounding and loss weighting
    var_weight = tf.constant(1e-3)
    std_range = tf.cast(tf.constant([0.001,0.05]),'float32') ### For bounding the logvar space
    logvar_range = 2*tf.math.log(std_range) ### do the math, this is correct
    
    # Model construction
    inputs = keras.Input(shape=(8, 8))  # Now expects (8,8) input

    x = MatMulInputLayer()(inputs)      # (batch, 8)

    ## Branching off toward a log_variance vector of same length
    x_trig = Layer_8_to_11()(x)

    
    ### Now running through a vec2image type model

    x = keras.layers.Dense(128)(x_trig)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(256)(x)  # Further expansion
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(8*8*64)(x)  # Prepare for reshaping into a smaller image
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Reshape((8, 8, 64))(x)  # Reshape into a smaller 3D tensor

    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)  # Convolution with small kernel
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    # Replace UpSampling2D + Conv2D with Conv2DTranspose
    x = keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)  # 8x8 -> 16x16
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2DTranspose(16, kernel_size=7, strides=2, padding='same')(x)  # 16x16 -> 32x32
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(3, kernel_size=9, padding='same', activation='linear')(x)  # Final convolution with large kernel and linear activation
    
    outputs = ScaleLayer()(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print_summary = False
    if print_summary == True:
        model.summary()

    # Compile the model with a custom loss function
    model.compile(optimizer='adam', loss='mse')
    
    return model







































































