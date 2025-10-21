#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:43:33 2025

Finding multiple faults from the summed displacement field fo known synthetic
faults.

@author: jon
"""
## import "standard" things
import numpy as np
import matplotlib.pyplot as plt
import itertools
plt.close('all')
### import from modules
from surrogate_utils import *


###############################################################################
# Things to control.
model_path = './trained_models/v2i_max_entropy_C2DT_2025_09_15_2109.keras'
N = 3 ## The true number of faults
MAX_LOOK = 3 ## we don't want more than this many faults to be found by our solution
A_min_max = 1,5 ## Setting range of fault slip magnitudes
std_mul = 0.0 ### noise level to add to the synthetic data example
AR_max = 5  ## maximum aspect ratio of the faults
AR_min = 2  ## minimum aspect ratio of the faults
side_min = 0.3 ## minimum size for width or length of a particular fault in synthetic example
side_min_solution = 0.2 ## minimum size for width or length of a particular fault in the inversion

plot_on = True ## argument to control some plotting

max_remove = 0 ## max number of faults to remove in the combinatorial analysis
max_add = 1 ## max number of faults to add in the combinatorial analysis
min_frac_improvement = 0.15 ### defined as 1-(new_misfit/old_misfit). Needed for convergence.

min_delta_explore = 5e-2 # Minimium improvement of loss for the initial inversion of the iteration
min_delta_refine = 1e-8 # Minimium improvement of loss for the final inversion of the iteration

add_noise_to_data = False
add_noise_to_resid = False
adjust_on = True ## for nudging
refine_on = True ## for final refinement

## search space (for search part of algorithm)
n_xyz_search = 5
n_sdr_search = 5
n_wl_search = 2


## Nudging parameters 
max_tries = 5# number of times to try to improve fit from a single faultduring nudging
std_loc = 0.05# standard deviation of the previous fault centroid location from where the nudging process starts
mfi_nudge = 0.0 # minimum fractional improvement for accepting a nudged set of fault parameters
### nudging mode 1: rotating 
n_sdr = 18#15
n_wl = 1#1
n_xyz = 1#1
mv_xyz = 1#0
xyz_sdr_wl_1 = gen_fault_params_xyz_sdr_wl(n_xyz,mv_xyz,n_sdr,n_wl)
if n_wl == 1:
    xyz_sdr_wl_1[:,-2:] = 0.05 ## small source
### mode 2: moving
n_sdr = 1#1
n_wl = 4#9
n_xyz = 5#7
mv_xyz = 0.03#0.03
xyz_sdr_wl_2 = gen_fault_params_xyz_sdr_wl(n_xyz,mv_xyz,n_sdr,n_wl)


############################### (1) ###########################################
###############################################################################
##                          Initialization                                   ##
###############################################################################
###############################################################################


## loading in the neural network model that has an extra scaling layer at the end
nn_model = load_nn_model(model_path)

## With this nn_model, getting the basis functions [as vectors] (y_bfs) 
## a multiple parameter model space (fault: location, orientation, size)
## and corresponing fault params to be used each time  to find the 
## best two candidate new fault locations
X,y_bfs = gen_BFs_xyz_sdr_wl_chunked(n_xyz_search,n_sdr_search,n_wl_search,nn_model,model_path)


### making new faults
all_faults, all_faults_scaled, all_A, Y = \
    synthesize_faults_and_Y_smarter(N,A_min_max,nn_model,AR_min,AR_max,\
                                    side_min)

# Initializing Y_resid, "current misfit" and converged flag.
Y_resid = Y.numpy().copy()
current_mf = tf.linalg.norm(Y).numpy() ### the initial residual / misfit / "m.f."/"mf"
current_mf_history = [current_mf]

# global_fault_list will contain the list of faults at the end of the optimization
global_fault_list = np.array([]).reshape(0,8)
global_fault_list_history = []## to be updatd after each iteration so that we can see how the optimization evolved.

global_amplitudes_list = np.array([]) ### we want to preserve the amplitudes of optimized faults in the current (global) list
global_amplitudes_list_history = []

Y_pred_history = []

### making some more lists 
after_new_it_faults = []
after_adjusting_faults = []
after_flipping_faults = []

after_new_it_Y_pred = []
after_adjusting_Y_pred = []

############################### (2) ###########################################
###############################################################################
##                      Looping (with "while") until converged               ##
###############################################################################
###############################################################################

converged = False

while converged == False:
    
    print('Statting whole new iteration. Last misfit:',current_mf_history[-1])
    print('Statting whole new iteration. Misfit to beat:',\
          current_mf_history[-1]*(1-min_frac_improvement))
    
    ## getting the best two faults that together fit the residual
    two_faults,A_prior = find_two_faults(Y_resid,X,y_bfs) ### later I want to add a condition that these faults do not intersect...
    testing_fault_list = np.append(global_fault_list,two_faults,axis=0)  ## outputs non-scaled fault params
    testing_amplitudes_list = np.append(global_amplitudes_list,A_prior)
    
    if plot_on == True:
        see_testing_faults = False #### for debugging
        if see_testing_faults == True:
            faults_temp = [fault_geom_from_source(j.ravel()) for j in testing_fault_list]
            plot_faults(faults_temp,ax_out[6],'m')
        
        
    ## From the testing_list, making fault_combinations
    fault_combos = get_fault_combos(testing_fault_list,max_remove)
    
    ## Sorting by smallest number of faults (maybe stuff into above function later)
    len_combos = np.array([len(i) for i in fault_combos])
    ii_lc = np.column_stack([np.arange(len_combos.size),len_combos])
    ii_lc = my_sortrows(ii_lc, [1], [1])
    fault_combos = [fault_combos[ii_lc[i,0]] for i in range(ii_lc.shape[0])]
    len_combos = np.array([len(i) for i in fault_combos])
    unq_combos = np.unique(len_combos)
    
    ### First making a list of fault combos for next fewest number of faults
    for h in range(unq_combos.size):
        sub_combos = []
        for hh in range(len_combos.size):
            if len_combos[hh]==unq_combos[h]:
                sub_combos.append(fault_combos[hh])
    
        ## for each fault combination, doing an initial inversion 
        ## to get a-priori estimates of amplitude with basis functions
        ### then feeding this model into the gradient descent and storing outputs of each combo
        fault_params_all = []
        amplitudes_all = []
        misfits_all = []
        Y_pred_all = []
            
        for i in range(len(sub_combos)):
            print(' ')
            print('# faults being tested: ',str(len(testing_fault_list)))
            print('Starting sub-combo ',str(i+1),' of ',str(len(sub_combos)))
            print('Combo contains ',str(len(sub_combos[i])),' faults')
            print('Combo is: ',sub_combos[i])
            fault_params,amplitudes,Y_pred, model_misfit = optimize_fault_combo_2(\
               testing_fault_list,testing_amplitudes_list,sub_combos[i],\
                                            Y.numpy(),nn_model,model_path,\
                                                side_min_solution,\
                                                min_delta_explore,0)
            fault_params_all.append(fault_params)
            amplitudes_all.append(amplitudes)
            misfits_all.append(model_misfit)
            Y_pred_all.append(Y_pred)
            
        
        ### Depending on number of faults, setting m_f_a
        if unq_combos[h] <= len(global_fault_list):
            m_f_a = 0 ### a reshuffle or loss of fault must at least be equal or better performance
        else:
            m_f_a = min_frac_improvement  ## adding one or more faults must exceed min_frac_improvement
        
        ### Now seeing if we have a min_frac_improvement (convergence checking)
        n_faults = np.array([len(i) for i in fault_params_all])
        ii_n_mfi = np.column_stack(\
                    [np.arange(len(misfits_all)),n_faults,\
                     1-(np.array(misfits_all)/current_mf)])## index fault combo, number of faults, min_frac_improvement
        ii_n_mfi = my_sortrows(ii_n_mfi, [1,2], [1,-1])### sorting appropriately
        ii_n_mfi = ii_n_mfi[ii_n_mfi[:,-1]>=m_f_a,:]
        
        if len(ii_n_mfi)>0:### leaving the loop if a significant improvement is found because advancing in loop adds another fault...
            break
    
    if (len(ii_n_mfi)>0):
        ii = ii_n_mfi[0][0].astype(int)
        print(ii_n_mfi)
        
        after_new_it_faults.append(fault_params_all[ii].copy())
        after_new_it_Y_pred.append(Y_pred_all[ii].copy())
        
        if adjust_on == True:
            print('')
            print('************************************************')
            print('Starting adjustment with mf:',misfits_all[ii])
            #### Now looping through each fault, removing fixed signal of other
            ###  faults, and adjusting the fault to fit remainder, Y_rem
                        
            
            ## params to be moved elsewhere laterafter devel stage
            keras_verbose_flag = 0
            ## running the nudging
            fault_params,amplitudes,Y_pred, model_misfit = \
                run_nudging_2(fault_params_all[ii],amplitudes_all[ii],\
                                    Y.numpy(),nn_model,model_path,\
                                        side_min_solution,\
                            keras_verbose_flag,std_loc,max_tries,mfi_nudge,\
                                xyz_sdr_wl_1,xyz_sdr_wl_2)
            
            ## updating parameters
            fault_params_all[ii] = fault_params.copy()
            amplitudes_all[ii] = amplitudes.copy()
            Y_pred_all[ii] = Y_pred.copy()
            misfits_all[ii] = model_misfit.copy()
            print('Ending adjustment with mf:',misfits_all[ii])
            
            after_adjusting_faults.append(fault_params_all[ii].copy())
            after_adjusting_Y_pred.append(Y_pred_all[ii].copy())
        
        
        if refine_on == True:### maybe remove the conditional if we ALWAYS need to refine here...
            ## Refinement of best faults to avoid underfitting and needing more
            ## faults to catch the locally underfit signal.
            print('**************************************************\
                  Refining the fits...')
            
            do_flip = False### searching for flipping faults perpendicularly. Resets sizes too
            if do_flip == True:
                fault_params_refine,amplitudes_refine = flip_search(fault_params_all[ii],\
                            Y,nn_model)
            else:
                fault_params_refine = fault_params_all[ii]
                amplitudes_refine = amplitudes_all[ii]
            
            after_flipping_faults.append(fault_params_refine.copy())
            
            
            fault_params,amplitudes,Y_pred, model_misfit = optimize_fault_combo_2(\
                    fault_params_refine,amplitudes_refine,\
                        np.arange(len(fault_params_refine)),\
                            Y.numpy(),nn_model,model_path,\
                                side_min_solution,\
                                min_delta_refine,0)
            
            global_fault_list = fault_params.copy()
            global_amplitudes_list = amplitudes.copy()
            current_mf = model_misfit.copy()
            Y_pred = Y_pred.copy()
            
        else:
            global_fault_list = fault_params_all[ii]
            global_amplitudes_list = amplitudes_all[ii]
            current_mf = misfits_all[ii]
            Y_pred = Y_pred_all[ii]
        
        current_mf_history.append(current_mf.copy())
        global_fault_list_history.append(global_fault_list)
        global_amplitudes_list_history.append(global_amplitudes_list)
        Y_pred_history.append(Y_pred)
        
        ### updating the residuals
        Y_resid = Y.numpy()-Y_pred
        
        ## Putting some noise into the residual field
        
        if add_noise_to_resid == True:
            Y_resid_out = add_noise_to_field(Y_resid)
            #see_two_fields_temp(Y_resid,Y_resid_out)
            Y_resid = Y_resid_out
        
        print('*************************************************')
        print('iteration done: old mf, new mf',current_mf_history\
              [len(current_mf_history)-2],\
              current_mf)
        
        n_faults = len(global_fault_list)
        if n_faults == MAX_LOOK:
            converged = True
        
    else:
        converged = True
        print('*********************************************\
              ****** CONVERGED')
    
   
###############################################################################
#### saving
if os.path.exists('./converged_solutions/') == False:
    os.mkdir('converged_solutions')
    
out_num = len(os.listdir('./converged_solutions/'))

out_path = './converged_solutions/sol_'+str(out_num).zfill(3)+'.npz'
np.savez(out_path,Y=Y,
all_faults=np.array(all_faults, dtype=object),
all_A=np.array(all_A, dtype=object),
current_mf_history=np.array(current_mf_history, dtype=object),
global_fault_list_history=np.array(global_fault_list_history, dtype=object),
global_amplitudes_list_history=np.array(global_amplitudes_list_history, dtype=object),
Y_pred_history=np.array(Y_pred_history, dtype=object),
after_new_it_faults=np.array(after_new_it_faults, dtype=object),
after_adjusting_faults=np.array(after_adjusting_faults, dtype=object),
after_flipping_faults=np.array(after_flipping_faults, dtype=object),
after_new_it_Y_pred=np.array(after_new_it_Y_pred, dtype=object),
after_adjusting_Y_pred=np.array(after_adjusting_Y_pred, dtype=object))











































