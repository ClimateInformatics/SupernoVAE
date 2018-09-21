# SupernoVAE: VAE based Kernel-PCA for analysis of spatio temporal earth data

This git repository contains the python code for the SupernoVAE framework presented on the **Climate Informatic Workshop 2018 in Boulder**. For explanations please refer to the paper in the proceedings of that workshop. Please cite this paper if you use SupernoVAE.

## Content of the Repository

This repository contains three files:

[Supernovae_class](Supernovae_class.py)

[SupernoVAE_functions](SupernoVAE_functions.py)

[Supernovae_tutorial](Supernovae_tutorial.ipynb)

## Getting started
To get started clone the repository and follow the instructions in the [Supernovae_tutorial](Supernovae_tutorial.ipynb) to create and run a toy example. This toy example will teach you how to apply SupernoVAE to your own tasks.

## Description of the Files

### Config_Svae.ini
This file contains the runtime configurations for the SupernoVAE class. It can be created using the build function included in the Supernovae_class.py.

### Supernovae_class.py
This file contains the supernovae class. It also contains all needed functions to create a configuration file and the needed folder structure as well as the functions to load and store data.

### SupernoVAE_functions.py
This file contains the model function of the neural network used as well as the construction of the data pipeline. If you want to alter the structure of the neural network, or the input data format, change this file.

## Requirements
We tested the code using Python 3.6 and Tensorflow 1.8
