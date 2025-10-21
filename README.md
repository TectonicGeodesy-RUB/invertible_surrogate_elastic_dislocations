# Invertible Surrogate Elastic Dislocations

The paper associated with this repositorry is currently submitted and under review at __Geophysical Research Letters__:

Jonathan Bedford and Kaan Çökerim. *An Invertible Surrogate Model for Elastic Half-Space Dislocations*

### What this repository contains:
This repository contains a minimal example to reproduce the results presented in the paper:

- it is recomended to set up a python environment using the provided `environment.yml` to ensure all scripts run properly
- a matlab scripts to generate synthetic examples in `sythetic_training_data_generation/generate_training_samples.m`
- a python script `general_model_trainer.py` which trains the model
- a python script `inspect_trained_model.py` which plots the model loss as a function of the model parameters
- a python script `find_multiple_faults.py` to perform the prediction of faults