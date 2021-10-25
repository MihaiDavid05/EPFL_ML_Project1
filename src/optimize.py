import argparse
import numpy as np
from src.run import read_config
from utils.data import load_csv_data, split_data_by_jet, remove_useless_columns, drop_correlated
from utils.optimizations import find_best_poly_lambda, find_best_reg_threshold

CONFIGS_PATH = '../configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('--test', action='store_true', help='Also test and create submission file')
    parser.add_argument('--see_hist', action='store_true', help='See features histogram panel')
    parser.add_argument('--see_loss', action='store_true', help='See training loss plot')
    parser.add_argument('--see_pca', action='store_true', help='See PCA with 2 components')
    parser.add_argument('--sub_models', type=str, default='0,1,2', help='Choose which sub-model to run. 0 - '
                                                                        'jet_zero, 1- jet_one, 2 - more than 1 jet')
    parser.add_argument('--search_type', type=str, default='lambda-degree', help='Type of grid search')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    cli_args = parse_arguments()
    config_path = CONFIGS_PATH + cli_args.config_filename + '.yaml'
    config = read_config(config_path)
    output_filename = config['output_path'] + cli_args.config_filename + '_submission'
    by_jet = cli_args.config_filename.split('_')[-1] == '3models'
    cli_args.sub_models = [int(i) for i in cli_args.sub_models.split(',')]

    # If there are 3 subsets, split by jet number, predict on each of them
    if by_jet:
        # Load data
        labels_tr, x_tr, _, x_name_tr = load_csv_data(config['train_data'])

        # Define lists for test indexes, predictions and metric
        idxs, preds, total_f1, total_acc = [], [], [], []

        # Split data according to jet number
        data_dict_tr = split_data_by_jet(x_tr, labels_tr, np.zeros(x_tr.shape[0]))

        # Remove columns full of useless values
        data_dict_tr = remove_useless_columns(data_dict_tr, x_name_tr)

        # Iterate through each subset
        for i, k in enumerate(data_dict_tr.keys()):
            if i in cli_args.sub_models:
                # Drop correlated features
                if config[k]["drop_corr"]:
                    data_dict_tr, tr_corr_idxs = drop_correlated(data_dict_tr, k, config)

                # Get test indices, training and testing data and labels for a subset
                _, x_tr, labels_tr, x_name_tr = data_dict_tr[k]

                # Optimizations
                if cli_args.search_type == 'lambda_degree':
                    find_best_poly_lambda(x_tr, labels_tr, config, cli_args)
                elif cli_args.search_type == 'reg_thresh':
                    find_best_reg_threshold(x_tr, labels_tr, config, cli_args)
                else:
                    print("No parameter search given")
    else:
        # Load data
        labels_tr, x_tr, _, x_name_tr = load_csv_data(config['train_data'])
        # Optimizations
        if cli_args.search_type == 'lambda_degree':
            find_best_poly_lambda(x_tr, labels_tr, config, cli_args)
        elif cli_args.search_type == 'reg_thresh':
            find_best_reg_threshold(x_tr, labels_tr, config, cli_args)
        else:
            print("No parameter search given")
