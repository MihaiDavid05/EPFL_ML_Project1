# EPFL_ML_Project1
Implementation of Project 1 for ML course, by team CMT.

Paper: [Learning to discover: the Higgs
boson machine learning challenge](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf)


## Environment
NOTE: You can skip step 1 if you have next libraries installed: `matplotlib`, `numpy`, `pyyaml`, `seaborn`.

##### 1. Import environment
There is an exported conda environment under `environment.yaml`and a `requirements.txt` with all the libraries.

Use one of the following command to get all the libraries:
```bash
conda env create -f environment.yml
```
or 
```bash
pip install -r requirements.txt
```

NOTE: For conda importing you should have Anaconda installed.

##### 2. Set path
Please add the project root folder to `$PYTHONPATH` using following command:
```bash
export PYTHONPATH=$PYTHONPATH:<path_to_project_folder>
```
We used PyCharm with Python 3.7. 
## Data
Please download the data and store all the `.csv` files under the `data` folder.


## Configs
Check `config` folder for different configs and experiments descriptions.

Configs follow a YAML format.
 
A `_3models` suffix for configuration files mean that it contains parameters for 
3 separate models used to train 3 subsets of the dataset. These subsets were created by splitting the 
entire dataset by the column `PRI_jet_num`, which is an ordinal feature.

## Train and test

For training run the following commands:
```bash
cd src
python run.py <config_filename> [OPTIONAL_ARGUMENTS]
``` 
Example:
```bash
cd src
python run.py experiment_1 --see_hist
``` 

For training and testing run the following commands
```bash
cd src
python run.py <config_filename> --test [OPTIONAL_ARGUMENTS]
``` 
Example:
```bash
cd src
python run.py experiment_2 --test --see_hist
``` 

## Results

All submissions will be stored under `results` folder in the form `<config_filename>_submission`.

## Code structure

1.Folder `utils`:
* `algo.py`: Helpers for functions in implementations.py, predicting labels and metrics.
* `config.py`: Config related helpers.
* `data.py`: All functions related do data loading, pre processing and cross validation.
* `implementations.py`: Different methods for finding weights.
* `optimizations.py`: Functions for finding optimal parameters through grid
searches and cross-validation.
* `visualization.py`: Functions for plots.

2.Folder `visualizations`: All the plots will be stored here.

3.Folder `src`:
* `run.py`: Main file for running the training and testing + an argument parser.
* `optimize.py`: Script for running different experiments from optimizations.py file.

4.Folder `results`: All submission files will be stored here.

5.Folder `data`: All the data should be stored here.

6.Folder `configs`: All configs should be stored here.


