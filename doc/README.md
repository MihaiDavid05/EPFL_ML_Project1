# EPFL_ML_Project1
Implementation of Project 1 for ML course, by team CMT.

Paper: [Learning to discover: the Higgs
boson machine learning challenge](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf)


## Environment
*NOTE*: You can skip step 1 if you have next libraries installed: `matplotlib`, `numpy`, `pyyaml`, `seaborn`.

#### 1. Import environment
There is an exported conda environment under `environment.yaml`and a `requirements.txt` with all the libraries.

Use one of the following command to get all the libraries:
```bash
conda env create -f environment.yml
```
or 
```bash
pip install -r requirements.txt
```

*NOTE*: For conda importing you should have Anaconda installed.

#### 2. Set path
Please add the project root folder to `$PYTHONPATH` using following command:
```bash
export PYTHONPATH=$PYTHONPATH:<path_to_project_folder>
```
We used PyCharm IDE and Python 3.7.
 
## Data
Please download the data and store all the `.csv` files under the `data` folder.


## Configs
Check `config` folder for different configs and experiments descriptions.
Configs follow a YAML format. We used them in order to keep track of our experiments with different parameters.
 
A `_3models` suffix for configuration files mean that it contains parameters for 
3 separate models used to train 3 subsets of the dataset. These subsets were created by splitting the 
entire dataset by the column `PRI_jet_num`, which is an ordinal feature.

A `_6models` suffix for configuration files mean that it contains parameters for 
6 separate models used to train 6 subsets of the dataset. These subsets were created by splitting the 
entire dataset by jet number, as described above, and further by the value of the first feature (-999 vs other).

## Train and test

For training and testing run the following commands:
```bash
cd src
python run.py <config_filename> [OPTIONAL_ARGUMENTS]
``` 
Example for obtaining the BEST RESULTS:

```bash
cd src
python run.py experiment_23_3models
``` 

## Results

All submissions will be stored under `results` folder in the form `<config_filename>_submission`.

Prediction files were submitted to AICrowd platform.
Our best results were **0.834 accuracy** and **0.751 F1-score**.

These results correspond to `experiment_23_3models` configuration file.

## Code structure

1.Folder `utils`:
* `algo.py`: Helpers for functions in implementations.py, predicting labels and metrics.
* `config.py`: Config related helpers.
* `data.py`: All functions related do data loading, pre processing and cross validation.
* `implementations.py`: Different methods for finding weights.
* `optimizations.py`: Functions for finding optimal parameters through grid
searches and cross-validation.
* `pipelines.py`: Functions for entire pipelines of data processing, training and testing 
divided into 2 groups: a pipeline that uses all the data and a pipeline that uses sub sets, thus sub models.
* `visualization.py`: Functions for plots.

2.Folder `src`:
* `run.py`: Main file for running an experiment + an argument parser.
* `optimize.py`: Script for running grid searches using functions from optimizations.py file.

3.Folder `visualizations`: All the plots are stored here.

4.Folder `results`: All submission files will be stored here.

5.Folder `data`: All the data should be stored here.

6.Folder `configs`: All configs should be stored here.


