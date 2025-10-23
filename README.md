
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# Invertible Surrogate Elastic Dislocations

The paper associated with this repository is currently submitted and under review at __Geophysical Research Letters__:

> Jonathan Bedford and Kaan Çökerim. *Finding Multiple Faults with Invertible Surrogates and Blind Source Separation*

### What this repository contains:
This repository contains a minimal example to reproduce the results presented in the paper.

> Due to the size of the `.mat`-file with the synthetic data we provide the synthetic samples in a `sciebo`-folder [here](). If you want to use these data, download and place the `.mat`-file in the `./synthetic_training_data_generation/` folder. However, you can (and are encouraged to) generate the sythetic samples yourself using the script mentioned below. 

We run this code in a conda environment which can be replicated `environment.yml`.  We do not recommend recreating this environment, rather we share it to show you which versions of packages we have used. 

To ensure that custom classes run properly, we recommend that you use the same Keras and Tensorflow versions that are listed in `environment.yml`.

The required software versions are:
- Keras Version: `3.6.0`
- Tensorflow Version: `2.18.0`
- Octave version: `8.4.0`; no additional Octave packages are necessary
<br/><br/>


The repository contains the following codes:
- Octave script to generate synthetic examples in `sythetic_training_data_generation/generate_training_samples.m`.
- Matlab function `sythetic_training_data_generation/TDdispHS.m` (than can be run in Octave 8.4.0).  This function is the copyright of Mehdi Nikkhoo (Copyright (c) 2015 Mehdi Nikkhoo). The full codes by Mehdi Nikkho are published as [https://doi.org/10.1093/gji/ggv035](https://doi.org/10.1093/gji/ggv035) and availiable for download at [https://www.volcanodeformation.com/software](https://www.volcanodeformation.com/software)
- Python script `general_model_trainer.py` trains the model.
- Python script `inspect_trained_model.py` plots the model loss as a function of the model parameters.
- Python script `animate_surrogate_predictions.py` plots the model predictions for gradually varying model parameters.
- Python script `find_multiple_faults.py` performs the blind source separation optimization of faults.
- Python script `see_BSS_results.py` plots the results of the converged blind source separation solution (after running `find_multiple_faults.py`).
- Python module `surrogate_utils.py` contains the functions and classes necessary to train the surrogate, do inference and inversion with the surrogate, and do blind source separation.
