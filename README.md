# Invertible Surrogate Elastic Dislocations

The paper associated with this repository is currently submitted and under review at __Geophysical Research Letters__:

> Jonathan Bedford and Kaan Çökerim. *An Invertible Surrogate Model for Elastic Half-Space Dislocations*

### What this repository contains:
This repository contains a minimal example to reproduce the results presented in the paper.

We run this code in a conda environment which can be replicated `environment.yml`.  We do not recommend recreating this environment, rather we share it to show you which versions of packages we have used. 

To ensure that custom classes run properly, we recommend that you use the same Keras and Tensorflow versions that are listed in `environment.yml`.

The required python package versions are:
- Keras Version: `3.6.0`
- Tensorflow Version: `2.18.0`
<br/><br/>

The repository contains the following codes:
- Octave version is 8.4.0 and no additional Octave packages are necessary.
- Octave script to generate synthetic examples in `sythetic_training_data_generation/generate_training_samples.m`.
- Matlab function `sythetic_training_data_generation/TDdispHS.m` (than can be run in Octave 8.4.0).  This function is the copyright of Mehdi Nikhoo (Copyright (c) 2014 Mehdi Nikkhoo).
- Python script `general_model_trainer.py` trains the model.
- Python script `inspect_trained_model.py` plots the model loss as a function of the model parameters.
- Python script `animate_surrogate_predictions.py` plots the model predictions for gradually varying model parameters.
- Python script `find_multiple_faults.py` performs the blind source separation optimization of faults.
- Python script `see_BSS_results.py` plots the results of the converged blind source separation solution (after running `find_multiple_faults.py`).
- Python module `surrogate_utils.py` contains the functions and classes necessary to train the surrogate, do inference and inversion with the surrogate, and do blind source separation.
