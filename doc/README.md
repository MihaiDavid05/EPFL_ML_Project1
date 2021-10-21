# EPFL_ML_Project1
Implementation of Project 1 for ML course, by team CMT.

Paper: [Learning to discover: the Higgs
boson machine learning challenge](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf)


## Environment
If you have libraries such as `matplotlib`, `numpy`, `pyyaml` and `seaborn` already installed,
step 1 can be omitted.

##### 1. Import environment
There is an exported conda environment under `environment.yaml`.
To import it please run the following command:
```bash
conda env create -f environment.yml
```

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

## Train and test

For training run the following commands:
```bash
cd src
python run.py <config_filename> [OPTIONAL_ARGUMENTS]
``` 
For training and testing run the following commands
```bash
cd src
python run.py <config_filename> --test [OPTIONAL_ARGUMENTS]
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


