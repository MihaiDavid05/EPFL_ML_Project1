import argparse
import numpy as np
from src.run import read_config
from utils.data import load_csv_data, split_data_by_jet, remove_useless_columns, drop_correlated, split_data_by_ffeat
from utils.optimizations import find_best_poly_lambda, find_best_reg_threshold

CONFIGS_PATH = '../configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('--test', action='store_true', help='Also test and create submission file')
    parser.add_argument('--see_hist', action='store_true', help='See features histogram panel')
    parser.add_argument('--see_loss', action='store_true', help='See training loss plot')
    parser.add_argument('--see_pca', action='store_true', help='See PCA with 2 components')
    parser.add_argument('--sub_models_by_jet', type=str, default='0,1,2', help='Choose which sub-model to run. 0-jet'
                                                                               '_zero, 1-jet_one, 2-more than 1 jet')
    parser.add_argument('--sub_models_by_jet_and_ffeat', type=str, default='0,1,2,3,4,5',
                        help='Choose which sub-model to run. 0-jet_zero and no mmcder, 1-jet_zero and mmcder,'
                             ' 2-jet_one and no mmcder, 3-jet_one and mmcder, 4-more_than_one and no mmcder,'
                             ' 5-more_than_one and mmcder,')
    parser.add_argument('--search_type', type=str, default='lambda-degree', help='Type of grid search')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    cli_args = parse_arguments()
    config_path = CONFIGS_PATH + cli_args.config_filename + '.yaml'
    config = read_config(config_path)
    output_filename = config['output_path'] + cli_args.config_filename + '_submission'
    by_jet = cli_args.config_filename.split('_')[-1] == '3models'
    by_jet_and_ffeat = cli_args.config_filename.split('_')[-1] == '6models'
    model_ids = None
    if by_jet:
        model_ids = [int(i) for i in cli_args.sub_models_by_jet.split(',')]
    elif by_jet_and_ffeat:
        model_ids = [int(i) for i in cli_args.sub_models_by_jet_and_ffeat.split(',')]

    # If there are subsets, apply grid search depending on their numbers
    if by_jet or by_jet_and_ffeat:
        # Define grids
        if by_jet:
            lambdas = [np.logspace(-15, -3, 13), np.logspace(-15, -3, 13), np.logspace(-15, -3, 13)]
            degrees = [range(4, 12), range(4, 12), range(4, 12)]
            thresholds = [np.linspace(0.01, 0.05, 5), np.linspace(0.01, 0.05, 5), np.linspace(0.01, 0.05, 5)]
        else:
            lambdas = [np.logspace(-15, -3, 13), np.logspace(-15, -3, 13), np.logspace(-14, -3, 12),
                       np.logspace(-16, -3, 14), np.logspace(-15, -3, 13), np.logspace(-15, -3, 13)]
            degrees = [range(3, 12), range(3, 12), range(3, 12), range(3, 12), range(3, 12), range(3, 12)]
            thresholds = [np.linspace(0.01, 0.05, 5), np.linspace(0.01, 0.05, 5), np.linspace(0.01, 0.05, 5),
                          np.linspace(0.01, 0.05, 5), np.linspace(0.01, 0.05, 5), np.linspace(0.01, 0.05, 5)]

        # Load data
        labels_tr, x_tr, _, x_name_tr = load_csv_data(config['train_data'])

        # Define lists for test indexes, predictions and metric
        idxs, preds, total_f1, total_acc = [], [], [], []

        # Split data according to jet number
        data_dict_tr = split_data_by_jet(x_tr, labels_tr, np.zeros(x_tr.shape[0]))

        if by_jet_and_ffeat:
            # Further split data according to first feature
            data_dict_tr = split_data_by_ffeat(data_dict_tr)

        # Remove columns full of useless values
        data_dict_tr = remove_useless_columns(data_dict_tr, x_name_tr)

        # Iterate through each subset
        res_dict = {}
        for i, k in enumerate(data_dict_tr.keys()):
            if i in model_ids:
                # Drop correlated features
                if config[k]["drop_corr"]:
                    data_dict_tr, tr_corr_idxs = drop_correlated(data_dict_tr, k, config)

                # Get test indices, training and testing data and labels for a subset
                _, x_tr, labels_tr, x_name_tr = data_dict_tr[k]

                # Optimizations
                if cli_args.search_type == 'lambda_degree':
                    tr_best, te_best = find_best_poly_lambda(x_tr, labels_tr, degrees[i], lambdas[i], config[k],
                                                             cli_args, x_name_tr, model_key=k)
                    res_dict[k] = [tr_best, te_best]
                elif cli_args.search_type == 'reg_thresh':
                    find_best_reg_threshold(x_tr, labels_tr, thresholds[i], config[k], cli_args, x_name_tr, model_key=k)
                else:
                    print("No parameter search given")
        print(res_dict)
    else:
        # Define grids
        lambdas = np.logspace(-4, 0, 5)
        degrees = list(range(3, 8))
        thresholds = np.linspace(0.01, 0.05, 5)
        # Load data
        labels_tr, x_tr, _, x_name_tr = load_csv_data(config['train_data'])
        # Optimizations
        if cli_args.search_type == 'lambda_degree':
            tr_best, te_best = find_best_poly_lambda(x_tr, labels_tr, degrees, lambdas, config, cli_args, x_name_tr)
        elif cli_args.search_type == 'reg_thresh':
            find_best_reg_threshold(x_tr, labels_tr, thresholds, config, cli_args, x_name_tr)
        else:
            print("No parameter search given")
