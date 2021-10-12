import argparse
import numpy as np
from utils.config import read_config
from utils.data import load_csv_data, create_csv_submission, build_model_data, standardize, build_poly, replace_values, \
    prepare_train_data, prepare_test_data
from utils.algo import do_cross_validation, predict_labels, accuracy, compute_pca
from utils.implementations import logistic_regression, reg_logistic_regression
from utils.vizualization import plot_hist_panel

CONFIGS_PATH = '../configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('--test', action='store_true', help='Also test and create submission file')
    parser.add_argument('--see_hist', action='store_true', help='See features histogram panel')

    return parser.parse_args()


def train(config, args):
    """
    Pipeline for training.
    :param config: Configuration parameters.
    :param args: Command line arguments
    :return:
    """
    feats, labels, tr_mean, tr_std = prepare_train_data(config, args)

    if config['cross_val']:
        do_cross_validation(feats, labels, logistic_regression, config)

    # Train on whole data set and find weights
    if config['lambda'] is not None:
        weights, tr_loss = reg_logistic_regression(labels, feats, config['lambda'], np.zeros((feats.shape[1], 1)),
                                                   config['max_iters'], config['gamma'])
    else:
        weights, tr_loss = logistic_regression(labels, feats, np.zeros((feats.shape[1], 1)), config['max_iters'],
                                               config['gamma'])

    tr_preds = predict_labels(weights, feats)
    tr_acc = accuracy(tr_preds, labels)
    print("Training accuracy is {:.2f} % ".format(tr_acc * 100))

    return tr_mean, tr_std, weights


def test(config, tr_mean, tr_std, tr_weights, output):
    """
    Pipeline for testing.
    :param config: Configuration parameters.
    :param tr_mean: Feature-wise training mean.
    :param tr_std: Feature-wise training standard deviation.
    :param tr_weights: Weights of the model.
    :param output: Output file_name
    :return:
    """

    test_feats, test_index = prepare_test_data(config, tr_mean, tr_std)
    # Predictions
    test_preds = predict_labels(tr_weights, test_feats)
    # Create submission file
    create_csv_submission(test_index, test_preds, output)


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    cli_args = parse_arguments()
    config_path = CONFIGS_PATH + cli_args.config_filename + '.yaml'
    c = read_config(config_path)
    output_filename = c['output_path'] + cli_args.config_filename + '_submission'

    # Train
    mean, std, w = train(c, cli_args)
    # Test
    if cli_args.test:
        test(c, mean, std, w, output_filename)
