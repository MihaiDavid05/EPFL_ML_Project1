import argparse
import numpy as np
from utils.config import read_config
from utils.data import create_csv_submission, prepare_train_data, prepare_test_data
from utils.algo import do_cross_validation, predict_labels, accuracy
from utils.implementations import logistic_regression, reg_logistic_regression
from utils.vizualization import plot_loss

CONFIGS_PATH = '../configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('--test', action='store_true', help='Also test and create submission file')
    parser.add_argument('--see_hist', action='store_true', help='See features histogram panel')
    parser.add_argument('--see_loss', action='store_true', help='See training loss plot')

    return parser.parse_args()


def train(config, args):
    """
    Pipeline for training.
    :param config: Configuration parameters.
    :param args: Command line arguments
    :return:
    """
    feats, labels, stat = prepare_train_data(config, args)

    if config['cross_val']:
        do_cross_validation(feats, labels, logistic_regression, config)

    # Train on whole data set and find weights
    if config['lambda'] is not None:
        weights, tr_loss = reg_logistic_regression(labels, feats, config['lambda'], np.zeros((feats.shape[1], 1)),
                                                   config['max_iters'], config['gamma'])
    else:
        weights, tr_loss = logistic_regression(labels, feats, np.zeros((feats.shape[1], 1)), config['max_iters'],
                                               config['gamma'])
    if args.see_loss:
        output_path = config["viz_path"] + 'loss_plot_' + args.config_filename
        plot_loss(range(config['max_iters']), np.ravel(tr_loss), output_path=output_path)

    tr_preds = predict_labels(weights, feats)
    tr_acc = accuracy(tr_preds, labels)
    print("Training accuracy is {:.2f} % ".format(tr_acc * 100))

    return stat, weights


def test(config, stat1, stat2, tr_weights, output):
    """
    Pipeline for testing.
    :param config: Configuration parameters.
    :param stat1: Feature-wise training mean or max - min
    :param stat2: Feature-wise training standard deviation or minimum
    :param tr_weights: Weights of the model.
    :param output: Output file_name
    :return:
    """

    test_feats, test_index = prepare_test_data(config, stat1, stat2)
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
    stats, w = train(c, cli_args)
    # Test
    if cli_args.test:
        test(c, stats[0], stats[1], w, output_filename)
