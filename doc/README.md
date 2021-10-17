# EPFL_ML_Project1
Implementation of Project 1 for ML course, by team CMT.

Paper: [Learning to discover: the Higgs
boson machine learning challenge](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf)


## Environment
If you have basic libraries such as `matplotlib` and `numpy` already installed,
step 1 can be omitted.

#####1. Import environment
There is an exported conda environment under `environment.yaml`.
To import it please run the following command:
```bash
conda env create -f environment.yml
```

#####2. Set path
Please add the project root folder to `$PYTHONPATH` using following command:
```bash
export PYTHONPATH=$PYTHONPATH:<path_to_project_folder>
```
We used PyCharm with Python 3.7. 
## Data
Please download the data and store all the `.csv` files under the `data` folder.


## Configs
Check `config` folder for different configs and experiments descriptions.

## Train and test

For training and testing run the following commands:
```bash
cd src
python run.py <config_filenme> [OPTIONAL_ARGUMENTS]
``` 
BE CAREFUL: `--test` is an optional argument, therefore you MUST provided it for building a submission file.

## Results

All submissions will be stored under `results` folder in the form `<config_filename>_submission`.


