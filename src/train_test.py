import argparse
import numpy as np
from utils.config import read_config
from utils.data import load_csv_data, create_csv_submission, build_model_data, standardize, build_poly, replace_values
from utils.algo import do_cross_validation, predict_labels, accuracy
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
    # Load data
    labels, feats, index, feats_name = load_csv_data(config['train_data'])

    if config["replace_with"] is not None:
        feats = replace_values(config, feats)

    if args.see_hist:
        name = 'hist_panel'
        if config["replace_with"] is not None:
            name += '_replaced_with_' + str(config["replace_with"])
        plot_hist_panel(feats, feats_name, config['viz_path'] + name)
        # These seems like good features to me, but selecting only them does not help apparently
        # feats = feats[:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 29]]

    if config["build_poly"]:
        # Build polynomial features
        feats = build_poly(feats, config["degree"])
    else:
        # Create x 'tilda'
        feats = build_model_data(feats)
    labels = labels.reshape((labels.shape[0], 1))

    # Feature standardization: we should not standardize feature 23 because it is categorical
    feats, tr_mean, tr_std = standardize(feats, without=[23])
    # plot_hist_panel(feats[:, 1:], feats_name, config['viz_path'] + 'hist_panel_after_standardization')

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
    # Load test data
    _, test_feats, test_index, _ = load_csv_data(config['test_data'])
    if config["replace_with"] is not None:
        test_feats = replace_values(config, test_feats)
    if config["build_poly"]:
        # Build polynomial features
        test_feats = build_poly(test_feats, config["degree"])
    else:
        # Create x 'tilda'
        test_feats = build_model_data(test_feats)

    # Normalize features
    test_feats = standardize(test_feats, tr_mean, tr_std, without=[23])
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
