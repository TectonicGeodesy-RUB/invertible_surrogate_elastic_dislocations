#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 17:31:51 2025

Moving a fault through the model space and predicting
slip

@author: jon
"""


model_path = './trained_models/v2i_max_entropy_C2DT_2025_09_15_2109.keras'


## import "standard" things
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
plt.close('all')
## import from modules
from surrogate_utils import *



### setting up list of models
n_frames = 800
rad_step = 0.1
corners = np.array([[-1,-1,-1],\
                    [1,1,0],\
                    [-1,1,0],\
                    [1,-1,-1],\
                    [1,1,-1],\
                    [-1,-1,0],\
                    [1,-1,0],\
                    [-1,1,-1],\
                        [-1,-1,-1]])
    
cycles_w = 10
cycles_l = 15
cycles_s = 20
cycles_d = 15



fpc = np.ceil(n_frames/(corners.shape[0]-1))
n_frames = int(fpc*(corners.shape[0]-1))

w = 0.1+0.9*(0.5+0.5*np.cos(np.linspace(0,cycles_w*2*np.pi,n_frames)))
l = 0.1+0.9*(0.5+0.5*np.cos(np.linspace(0,cycles_l*2*np.pi,n_frames)))
s = (np.linspace(0,cycles_s*2*np.pi,n_frames))%(2*np.pi)
d = (np.linspace(0,cycles_d*2*np.pi,n_frames))%(2*np.pi)
r = (np.pi/2)*np.ones(d.size)

xyz = np.array([]).reshape(0,3)
for i in range(len(corners))[1:]:
    xx = np.linspace(corners[i-1,0],corners[i,0],int(fpc+1))[1:]
    yy = np.linspace(corners[i-1,1],corners[i,1],int(fpc+1))[1:]
    zz = np.linspace(corners[i-1,2],corners[i,2],int(fpc+1))[1:]
    xyz = np.vstack([xyz,np.column_stack([xx,yy,zz])])
    
    
fig= plt.figure()   
ax = fig.add_subplot(1,1,1,projection='3d')
ax.plot3D(xyz[:,0],xyz[:,1],xyz[:,2],'.')

    
X = np.column_stack([xyz,s,d,r,w,l])
X,X_scaled = ORIG8_to_RS8_on_mat(X)

###  cleaning out old folder
save_folder = './showcase_model_space_predictions/'
if os.path.exists(save_folder) == False:
    os.mkdir(save_folder)
if os.path.exists(save_folder) == True:
    old_files = os.listdir(save_folder)
    for i in range(len(old_files)):
        rem_path = save_folder+old_files[i]
        os.remove(rem_path)
        
        
### loading in model

nn_model = load_nn_model(model_path)

### Preparing things for the plotting


vmini,vmaxi = -1,1

C = 1
LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 16

### Now running the predictions and plotting
fig,ax_all = mk_subplots_target_only()
ax_all[0].axis('equal')
ax_all[1].axis('equal')
ax_all[2].axis('equal')
ax_all[0].set_title('E',fontsize=20)
ax_all[1].set_title('N',fontsize=20)
ax_all[2].set_title('U',fontsize=20)

for i in range(X.shape[0]):
    Y_pred = nn_model(X_scaled[i,:][None,:])[0]
    
    if i == 0:
        XI = np.linspace(-1,1,Y_pred.shape[0])
        YI = np.linspace(-1,1,Y_pred.shape[1])
    
    if i > 0:
        ha_0.remove()
        ha_1.remove()
        ha_2.remove()
        ha_3[0].remove()
    
    ha_0 = ax_all[0].pcolor(XI,YI,Y_pred[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ha_1 = ax_all[1].pcolor(XI,YI,Y_pred[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ha_2 = ax_all[2].pcolor(XI,YI,Y_pred[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    
    pf = fault_geom_from_source(X[i,:])
    ha_3 = plot_faults_return_handle([pf],ax_all[3],'b')
    
    tit = 'x,y,z,s,d,r,w,l : '+\
        format(X[i][0],'.2f')+','+\
            format(X[i][1],'.2f')+','+\
                format(X[i][2],'.2f')+','+\
                    format(X[i][3],'.2f')+','+\
                        format(X[i][4],'.2f')+','+\
                            format(X[i][5],'.2f')+','+\
                                format(X[i][6],'.2f')+','+\
                                    format(X[i][7],'.2f')
    
    ax_all[3].set_title(tit,fontsize=20)
    
    
    if i == 0:
        # Add a colorbar axis in the 7th grid position (left column, bottom row)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.02, 0.20])  # [left, bottom, width, height], adjust as needed
            # Suppose im is the mappable object from your imshow/contourf/plot (e.g., im = ax_all[0].imshow(...))
        # Use the last im you plotted, or any representative one:
        cbar = fig.colorbar(ha_0, cax=cbar_ax, orientation='vertical')
        clab = 'Normalized'+'\n'+'Displacement '+'\n'+'(m)'
        cbar.set_label(clab, fontsize=LABEL_FONT_SIZE,rotation=0,labelpad=70)
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
        cbar.ax.set_yticklabels([f'{-C:.2f}', '0', f'{C:.2f}'])
    
    
    
    sname = save_folder+'out_'+str(i).zfill(5)+'.png'
    plt.savefig(fname=sname,dpi=200)
    
    print(i)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





























































