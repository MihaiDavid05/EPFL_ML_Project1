# EPFL_ML_Project1
Implementation of Project 1 - ML course, by team CMT.

## Environment
There is an exported conda environment under `environment.yaml`.
To import it please run the following command:
```bash
conda env create -f environment.yml
```
We used Python 3.7. 
## Data
Please download the data and store all the `.csv` files under the `data` folder.


## Configs
Check `config` folder for different configs and experiments descriptions.

## Train and test

For training and testing run the following commands:
```bash
cd src
python train_test.py <config_filenme> [OPTIONAL_ARGUMENTS]
``` 
