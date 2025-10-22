#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 17:58:32 2025

Script to plot the results of the blind source separation.

@author: Jonathan Bedford, Tectonic Geodesy, Ruhr University Bochum, DE
"""
## import "standard" things
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
plt.close('all')
## import from modules
from surrogate_utils import *


def return_things_from_sol(fpath):
    Y = np.load(fpath,allow_pickle=True)['Y']
    all_faults=np.load(fpath,allow_pickle=True)['all_faults']
    all_A=np.load(fpath,allow_pickle=True)['all_A']
    current_mf_history=np.load(fpath,allow_pickle=True)['current_mf_history']
    global_fault_list_history=np.load(fpath,allow_pickle=True)['global_fault_list_history']
    global_amplitudes_list_history=np.load(fpath,allow_pickle=True)['global_amplitudes_list_history']
    Y_pred_history=np.load(fpath,allow_pickle=True)['Y_pred_history']
    after_new_it_faults=np.load(fpath,allow_pickle=True)['after_new_it_faults']
    after_adjusting_faults=np.load(fpath,allow_pickle=True)['after_adjusting_faults']
    after_flipping_faults=np.load(fpath,allow_pickle=True)['after_flipping_faults']
    after_new_it_Y_pred=np.load(fpath,allow_pickle=True)['after_new_it_Y_pred']
    after_adjusting_Y_pred=np.load(fpath,allow_pickle=True)['after_adjusting_Y_pred']
    
    return Y,all_faults,all_A,current_mf_history,global_fault_list_history,\
    global_amplitudes_list_history,Y_pred_history,after_new_it_faults,\
    after_adjusting_faults,after_flipping_faults,after_new_it_Y_pred,\
    after_adjusting_Y_pred


def update_fig(Y_true,Y_pred,ft,fp,ax_all):
    
    vmini = -1*np.max(np.abs(Y_true))
    vmaxi = np.max(np.abs(Y_true))
    
    XI = np.linspace(-1,1,Y_true.shape[0])
    YI = np.linspace(-1,1,Y_true.shape[1])
    
    ha_0 = ax_all[0].pcolor(XI,YI,Y_true[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ha_1 = ax_all[1].pcolor(XI,YI,Y_true[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ha_2 = ax_all[2].pcolor(XI,YI,Y_true[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    ha_3 = ax_all[3].pcolor(XI,YI,Y_pred[:,:,0],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ha_4 = ax_all[4].pcolor(XI,YI,Y_pred[:,:,1],vmin=vmini,vmax=vmaxi,cmap='bwr')
    ha_5 = ax_all[5].pcolor(XI,YI,Y_pred[:,:,2],vmin=vmini,vmax=vmaxi,cmap='bwr')
    
    ha_6a = plot_faults_return_handle(ft,ax_all[6],'b')
    ha_6b = plot_faults_return_handle(fp,ax_all[6],'m')
    
    return ha_0,ha_1,ha_2,ha_3,ha_4,ha_5,ha_6a,ha_6b


###  cleaning out old folder
save_folder = './fault_optimization_figures/'
if os.path.exists(save_folder) == False:
    os.mkdir(save_folder)
if os.path.exists(save_folder) == True:
    old_files = os.listdir(save_folder)
    for i in range(len(old_files)):
        rem_path = save_folder+old_files[i]
        os.remove(rem_path)
        

folder = './converged_solutions/'
os.makedirs(folder, exist_ok=True)


all_sol_files = os.listdir(folder)


for i in range(len(all_sol_files)):
    fpath = folder+all_sol_files[i]
    
    Y,all_faults,all_A,current_mf_history,global_fault_list_history,\
    global_amplitudes_list_history,Y_pred_history,after_new_it_faults,\
    after_adjusting_faults,after_flipping_faults,after_new_it_Y_pred,\
    after_adjusting_Y_pred = return_things_from_sol(fpath)
    
    fig,ax_all = mk_subplots()
    for j in range(6):
        ax_all[j].axis('equal')
    
    
    faults_true = all_faults.astype('float')
    ft = [fault_geom_from_source(j.ravel()) for j in faults_true]

    Y = Y.astype('float')
    
    
    Y_pred = Y_pred_history[-1].astype('float')
    faults_pred = global_fault_list_history[-1].astype('float')
    fp = [fault_geom_from_source(j.ravel()) for j in faults_pred]

    ha_0,ha_1,ha_2,ha_3,ha_4,ha_5,ha_6a,ha_6b = \
        update_fig(Y,Y_pred,ft,fp,ax_all)


    # Parameters
    TIT_FONT_SIZE = 40
    LABEL_FONT_SIZE = 25
    TICK_FONT_SIZE = 16
    RIGHT_LABEL_PAD = 50  # adjust as needed
    BOTTOM_LABEL_PAD = 10  # adjust as needed

    # Titles
    ax_all[0].set_title('East', fontsize=TIT_FONT_SIZE)
    ax_all[1].set_title('North', fontsize=TIT_FONT_SIZE)
    ax_all[2].set_title('Vertical', fontsize=TIT_FONT_SIZE)

    # X labels
    for j in [3, 4, 5]:
        ax_all[j].set_xlabel('X', fontsize=LABEL_FONT_SIZE)

    # Y labels (left)
    ax_all[0].set_ylabel('Y', fontsize=LABEL_FONT_SIZE, rotation=0)
    ax_all[3].set_ylabel('Y', fontsize=LABEL_FONT_SIZE, rotation=0)

    # Y labels (right)
    ax_all[2].yaxis.set_label_position("right")
    ax_all[2].set_ylabel('Target', fontsize=TIT_FONT_SIZE, rotation=270, labelpad=RIGHT_LABEL_PAD)
    ax_all[5].yaxis.set_label_position("right")
    ax_all[5].set_ylabel('Prediction', fontsize=TIT_FONT_SIZE, rotation=270, labelpad=RIGHT_LABEL_PAD)

    # 3D axis labels
    ax_all[6].set_xlabel('X', fontsize=LABEL_FONT_SIZE, labelpad=BOTTOM_LABEL_PAD)
    ax_all[6].set_ylabel('Y', fontsize=LABEL_FONT_SIZE, labelpad=BOTTOM_LABEL_PAD)
    ax_all[6].set_zlabel('Z', fontsize=LABEL_FONT_SIZE, labelpad=BOTTOM_LABEL_PAD)

    # Set ticks and tick labels for all 2D axes
    for j in range(6):
        ax_all[j].set_xticks([-1, 0, 1])
        ax_all[j].set_yticks([-1, 0, 1])
        ax_all[j].tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # 3D axis ticks and tick labels
    ax_all[6].set_xticks([-1, 0,  1])
    ax_all[6].set_yticks([-1, 0, 1])
    ax_all[6].set_zticks([-2, -1, 0])
    ax_all[6].tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # 3D axis limits and view
    lim = 1.05
    ax_all[6].set_xlim(-lim, lim)
    ax_all[6].set_ylim(-lim, lim)
    ax_all[6].set_zlim(-2*lim, 0)
    ax_all[6].view_init(elev=40, azim=-70)

    # Making legend for the faults
    ha_6a[0].set_label('Target faults')
    ha_6b[0].set_label('Inverted faults')
    ax_all[6].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=LABEL_FONT_SIZE)

    # Add a colorbar axis in the 7th grid position (left column, bottom row)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.02, 0.20])  # [left, bottom, width, height], adjust as needed
        # Suppose im is the mappable object from your imshow/contourf/plot (e.g., im = ax_all[0].imshow(...))
    # Use the last im you plotted, or any representative one:
    cbar = fig.colorbar(ha_0, cax=cbar_ax, orientation='vertical')
    clab = 'Displacement '+'\n'+'(m)'
    cbar.set_label(clab, fontsize=LABEL_FONT_SIZE,rotation=0,labelpad=70)
    C = np.max(np.abs(Y))
    cbar.set_ticks([-C, 0, C])
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    cbar.ax.set_yticklabels([f'{-C:.2f}', '0', f'{C:.2f}'])

    
    ### saving
    save_name = save_folder+str(i).zfill(3)+'.png'
    plt.savefig(fname=save_name,dpi=180)
    
    plt.close('all')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    